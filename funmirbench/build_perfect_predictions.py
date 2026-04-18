"""Generate a dataset-aware perfect demo predictor from benchmark labels."""

import argparse
import math
import pathlib

import pandas as pd

from funmirbench.build_cheating_predictions import DEMO_DATASET_IDS, _expected_effect_from_logfc
from funmirbench.build_predictions import write_tsv
from funmirbench.de_table import find_gene_id_column, read_de_table
from funmirbench.logger import parse_log_level, setup_logging


def _resolve_table_path(root, value):
    path = pathlib.Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def _rank_scale(series):
    values = series.astype(float)
    ranks = values.rank(method="dense", ascending=True)
    max_rank = ranks.max(skipna=True)
    if pd.isna(max_rank):
        return pd.Series(float("nan"), index=series.index)
    if float(max_rank) <= 1.0:
        return pd.Series(1.0, index=series.index, dtype=float)
    return (ranks - 1.0) / (float(max_rank) - 1.0)


def build_perfect_scores(
    experiments_tsv,
    root,
    *,
    dataset_ids=None,
    fdr_threshold=0.05,
    abs_logfc_threshold=1.0,
):
    df = pd.read_csv(experiments_tsv, sep="\t")
    if dataset_ids:
        df = df[df["id"].isin(dataset_ids)].copy()
    if df.empty:
        raise ValueError("No experiments selected for perfect predictor generation.")

    scores = {}
    for _, row in df.iterrows():
        dataset_id = str(row["id"])
        mirna = str(row["mirna_name"])
        experiment_type = str(row.get("experiment_type", "") or "").upper()
        path = _resolve_table_path(root, row["de_table_path"])
        de = read_de_table(path)
        gene_src = find_gene_id_column(de)
        if gene_src == "__index__":
            de = de.copy()
            de.insert(0, "gene_id", de.index.astype(str))
        else:
            de = de.rename(columns={gene_src: "gene_id"})

        required_cols = {"gene_id", "logFC", "FDR"}
        missing = [col for col in required_cols if col not in de.columns]
        if missing:
            raise ValueError(f"{dataset_id} is missing required columns: {missing}")
        if de["gene_id"].duplicated().any():
            raise ValueError(f"{dataset_id} contains duplicate gene_id values.")

        keep = de[["gene_id", "logFC", "FDR"]].copy()
        keep = keep[keep["gene_id"].notna() & keep["logFC"].notna() & keep["FDR"].notna()].copy()
        keep["gene_id"] = keep["gene_id"].astype(str)
        keep["logFC"] = keep["logFC"].astype(float)
        keep["FDR"] = keep["FDR"].astype(float)
        keep = keep[keep["FDR"] > 0].copy()
        if keep.empty:
            continue

        expected_effect = _expected_effect_from_logfc(keep["logFC"], experiment_type)
        significance_signal = (-keep["FDR"].clip(lower=1e-300).map(math.log10)).clip(lower=0.0, upper=6.0) / 6.0
        effect_signal = expected_effect.clip(lower=0.0, upper=3.0) / 3.0
        keep["quality"] = 0.70 * effect_signal + 0.30 * significance_signal
        keep["is_positive"] = (
            (keep["FDR"] < fdr_threshold) & (expected_effect > abs_logfc_threshold)
        ).astype(int)
        keep["score"] = 0.0

        positive_mask = keep["is_positive"].astype(bool)
        if bool(positive_mask.any()):
            positive_rank = _rank_scale(keep.loc[positive_mask, "quality"])
            keep.loc[positive_mask, "score"] = 0.5 + 0.5 * positive_rank
        if bool((~positive_mask).any()):
            negative_rank = _rank_scale(keep.loc[~positive_mask, "quality"])
            keep.loc[~positive_mask, "score"] = 0.499999 * negative_rank

        for gene_id, score in zip(keep["gene_id"], keep["score"]):
            scores[(dataset_id, mirna, gene_id)] = float(score)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Build the perfect demo predictor.")
    parser.add_argument("--experiments-tsv", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, default=None)
    parser.add_argument(
        "--dataset-id",
        dest="dataset_ids",
        action="append",
        default=None,
        help="Restrict to specific dataset IDs. Defaults to the shipped demo datasets.",
    )
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--abs-logfc-threshold", type=float, default=1.0)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    setup_logging(parse_log_level(args.log_level))

    root = (args.root or args.experiments_tsv.parent).resolve()
    dataset_ids = args.dataset_ids or DEMO_DATASET_IDS
    scores = build_perfect_scores(
        args.experiments_tsv.resolve(),
        root,
        dataset_ids=dataset_ids,
        fdr_threshold=args.fdr_threshold,
        abs_logfc_threshold=args.abs_logfc_threshold,
    )
    write_tsv(scores, args.out.resolve())


if __name__ == "__main__":
    main()
