"""Join experiment DE tables with prediction tool scores."""

from pathlib import Path

import pandas as pd

from funmirbench import DatasetMeta
from funmirbench.de_table import find_gene_id_column, read_de_table


def load_experiment_table(meta: DatasetMeta) -> pd.DataFrame:
    de = read_de_table(meta.full_path)
    gene_src = find_gene_id_column(de)
    if gene_src == "__index__":
        de = de.copy()
        de.insert(0, "gene_id", de.index.astype(str))
    else:
        de = de.rename(columns={gene_src: "gene_id"})
    de["gene_id"] = de["gene_id"].astype(str)
    missing = [col for col in ("logFC", "FDR") if col not in de.columns]
    if missing:
        raise ValueError(f"{meta.full_path} missing required columns: {missing}")
    if de["gene_id"].duplicated().any():
        raise ValueError(f"Duplicate gene_id values found in {meta.full_path}")
    keep = ["gene_id", "logFC", "FDR"]
    if "PValue" in de.columns:
        keep.append("PValue")
    out = de[keep].copy()
    out.insert(0, "mirna", meta.miRNA)
    out.insert(0, "dataset_id", meta.id)
    return out


def load_tool_scores(
    tool_id: str,
    tool_meta: dict,
    root: Path,
    mirna: str,
    col_name: str,
    min_score: float | None = None,
) -> tuple[pd.DataFrame, Path]:
    path = Path(tool_meta["predictor_output_path"])
    if not path.is_absolute():
        path = root / path
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [col for col in ("mirna", "gene_id", "score") if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df = df[df["mirna"].astype(str) == mirna].copy()
    if min_score is not None:
        df = df[df["score"].astype(float) >= float(min_score)].copy()
    df["gene_id"] = df["gene_id"].astype(str)
    if df["gene_id"].duplicated().any():
        raise ValueError(
            f"Duplicate mirna+gene scores found for tool {tool_id} in {path}"
        )
    return df[["gene_id", "score"]].rename(columns={"score": col_name}), path


def build_joined(meta, tool_ids, predictions, root, min_score: float | None = None):
    joined = load_experiment_table(meta)
    paths = {}
    for tool_id in tool_ids:
        if tool_id not in predictions:
            raise ValueError(f"Unknown tool {tool_id!r}. Known: {sorted(predictions)}")
        scores, predictor_output_path = load_tool_scores(
            tool_id,
            predictions[tool_id],
            root,
            meta.miRNA,
            f"score_{tool_id}",
            min_score=min_score,
        )
        joined = joined.merge(scores, on="gene_id", how="left")
        paths[tool_id] = str(predictor_output_path)
    return joined, paths
