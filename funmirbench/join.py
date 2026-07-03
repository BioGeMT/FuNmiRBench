"""Join experiment DE tables with prediction tool scores."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Collection

import pandas as pd

from funmirbench import DatasetMeta
from funmirbench.benchmark_config import resolve_predictor_output_path
from funmirbench.de_table import find_gene_id_column, read_de_table


PREDICTOR_CHUNK_SIZE = 1_000_000


def _emit_log(logger, message: str) -> None:
    if logger is not None:
        logger(message)


def _elapsed(start: float) -> float:
    return time.perf_counter() - start


def _strip_ensembl_version(value: object) -> str:
    return str(value).strip().split(".", 1)[0]


def _compute_global_rank_percentile(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    ranks = values.rank(method="dense", ascending=True)
    max_rank = ranks.max(skipna=True)
    if pd.isna(max_rank):
        return pd.Series(float("nan"), index=series.index)
    if float(max_rank) <= 1.0:
        return pd.Series(1.0, index=series.index, dtype=float)
    return (ranks - 1.0) / (float(max_rank) - 1.0)


def _global_rank_percentile_map(scores: pd.Series) -> dict[float, float]:
    values = pd.to_numeric(scores, errors="coerce").dropna().astype(float)
    if values.empty:
        return {}
    ordered = pd.Series(values.unique()).sort_values(kind="mergesort").reset_index(drop=True)
    if len(ordered) <= 1:
        return {float(value): 1.0 for value in ordered}
    denominator = float(len(ordered) - 1)
    return {float(value): float(index / denominator) for index, value in enumerate(ordered)}


def _normalize_scores(scores: pd.Series, *, score_direction: str, tool_id: str) -> pd.Series:
    normalized = pd.to_numeric(scores, errors="coerce").astype(float)
    if score_direction == "lower_is_stronger":
        return -normalized
    if score_direction != "higher_is_stronger":
        raise ValueError(
            f"Unsupported score_direction {score_direction!r} for tool {tool_id!r}."
        )
    return normalized


def load_experiment_table(
    meta: DatasetMeta,
    *,
    protein_coding_gene_ids: Collection[str] | None = None,
    logger=None,
) -> pd.DataFrame:
    start = time.perf_counter()
    de = read_de_table(meta.full_path)
    gene_src = find_gene_id_column(de)
    if gene_src == "__index__":
        de = de.copy()
        de.insert(0, "gene_id", de.index.astype(str))
    else:
        de = de.rename(columns={gene_src: "gene_id"})
    de["gene_id"] = de["gene_id"].map(_strip_ensembl_version)
    missing = [col for col in ("logFC", "FDR") if col not in de.columns]
    if missing:
        raise ValueError(f"{meta.full_path} missing required columns: {missing}")
    if de["gene_id"].duplicated().any():
        raise ValueError(f"Duplicate gene_id values found in {meta.full_path}")
    keep = ["gene_id", "logFC", "FDR"]
    for optional in (
        "PValue",
        "plot_FDR",
        "benchmark_FDR",
    ):
        if optional in de.columns:
            keep.append(optional)
    out = de[keep].copy()
    if protein_coding_gene_ids is not None:
        before = len(out)
        protein_coding_gene_ids = {
            _strip_ensembl_version(gene_id)
            for gene_id in protein_coding_gene_ids
        }
        out = out.loc[
            out["gene_id"].isin(protein_coding_gene_ids)
        ].copy()
        _emit_log(
            logger,
            f"    Protein-coding filter | DE rows={before:,} -> {len(out):,}",
        )
    out.insert(0, "mirna", meta.miRNA)
    out.insert(0, "dataset_id", meta.id)
    out.insert(2, "perturbation", meta.perturbation)
    _emit_log(logger, f"    Join timing | DE table | rows={len(out):,} | { _elapsed(start):.2f}s")
    return out


def _read_relevant_tool_scores(
    path: Path,
    *,
    tool_id: str,
    score_direction: str,
    dataset_id: str,
    mirna: str,
    min_score: float | None,
) -> tuple[pd.DataFrame, dict[float, float], int]:
    header = pd.read_csv(path, sep="\t", nrows=0)
    columns = [str(c).strip() for c in header.columns]
    column_lookup = {str(c).strip(): c for c in header.columns}
    required = ("Ensembl_ID", "miRNA_Name", "Score")
    missing = [col for col in required if col not in columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    has_dataset_id = "Dataset_ID" in columns
    usecols = [column_lookup[col] for col in required]
    if has_dataset_id:
        usecols.append(column_lookup["Dataset_ID"])

    chunks = []
    score_values = []
    rows_read = 0
    reader = pd.read_csv(path, sep="\t", usecols=usecols, chunksize=PREDICTOR_CHUNK_SIZE)
    for chunk in reader:
        chunk.columns = [str(c).strip() for c in chunk.columns]
        rows_read += len(chunk)
        normalized_score = _normalize_scores(
            chunk["Score"],
            score_direction=score_direction,
            tool_id=tool_id,
        )
        score_values.append(normalized_score.dropna())

        mask = chunk["miRNA_Name"].astype(str) == str(mirna)
        if has_dataset_id:
            mask &= chunk["Dataset_ID"].astype(str) == str(dataset_id)
        if min_score is not None:
            mask &= normalized_score >= float(min_score)
        if not bool(mask.any()):
            continue
        selected = chunk.loc[mask, ["Ensembl_ID"]].copy()
        selected["Score"] = normalized_score.loc[mask].astype(float)
        chunks.append(selected)

    all_scores = pd.concat(score_values, ignore_index=True) if score_values else pd.Series(dtype=float)
    rank_map = _global_rank_percentile_map(all_scores)
    relevant = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["Ensembl_ID", "Score"])
    return relevant, rank_map, rows_read


def load_tool_scores(
    tool_id: str,
    tool_meta: dict,
    root: Path,
    dataset_id: str,
    mirna: str,
    col_name: str,
    rank_col_name: str,
    min_score: float | None = None,
    *,
    logger=None,
) -> tuple[pd.DataFrame, Path]:
    start = time.perf_counter()
    path = resolve_predictor_output_path(tool_meta["predictor_output_path"], root)
    score_direction = str(tool_meta.get("score_direction", "higher_is_stronger") or "higher_is_stronger")

    df, rank_map, rows_read = _read_relevant_tool_scores(
        path,
        tool_id=tool_id,
        score_direction=score_direction,
        dataset_id=dataset_id,
        mirna=mirna,
        min_score=min_score,
    )
    df[rank_col_name] = df["Score"].map(rank_map)
    df["gene_id"] = df["Ensembl_ID"].map(_strip_ensembl_version)
    if df["gene_id"].duplicated().any():
        keep_idx = df.groupby("gene_id")["Score"].idxmax()
        df = df.loc[keep_idx, ["gene_id", "Score", rank_col_name]].reset_index(drop=True)
    else:
        df = df[["gene_id", "Score", rank_col_name]].copy()
    out = df.rename(columns={"Score": col_name})
    _emit_log(
        logger,
        (
            f"    Join timing | {tool_id} | file_rows={rows_read:,} | "
            f"matched_rows={len(out):,} | unique_scores={len(rank_map):,} | {_elapsed(start):.2f}s"
        ),
    )
    return out, path


def build_joined(
    meta,
    tool_ids,
    predictions,
    root,
    min_score: float | None = None,
    *,
    protein_coding_gene_ids: Collection[str] | None = None,
    logger=None,
):
    total_start = time.perf_counter()
    joined = load_experiment_table(
        meta,
        protein_coding_gene_ids=protein_coding_gene_ids,
        logger=logger,
    )
    paths = {}
    for tool_id in tool_ids:
        if tool_id not in predictions:
            raise ValueError(f"Unknown tool {tool_id!r}. Known: {sorted(predictions)}")
        tool_start = time.perf_counter()
        scores, predictor_output_path = load_tool_scores(
            tool_id,
            predictions[tool_id],
            root,
            meta.id,
            meta.miRNA,
            f"score_{tool_id}",
            f"global_rank_{tool_id}",
            min_score=min_score,
            logger=logger,
        )
        joined = joined.merge(scores, on="gene_id", how="left")
        paths[tool_id] = str(predictor_output_path)
        scored = int(joined[f"score_{tool_id}"].notna().sum())
        _emit_log(
            logger,
            f"    Join timing | {tool_id} merge | scored_genes={scored:,} | {_elapsed(tool_start):.2f}s",
        )
    _emit_log(
        logger,
        f"    Join timing | {meta.id} total | rows={len(joined):,} | {_elapsed(total_start):.2f}s",
    )
    return joined, paths
