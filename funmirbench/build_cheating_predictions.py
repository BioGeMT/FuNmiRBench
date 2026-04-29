"""Generate a strong demo-only predictor from DE labels."""

import argparse
import logging
import math
import pathlib

import pandas as pd

from funmirbench.build_predictions import stable_hash_float, write_tsv
from funmirbench.de_table import find_gene_id_column, read_de_table
from funmirbench.logger import parse_log_level, setup_logging

logger = logging.getLogger(__name__)
DEFAULT_FDR_THRESHOLD = 0.05
DEFAULT_ABS_LOGFC_THRESHOLD = 1.0

DEMO_DATASET_IDS = [
    "GSE109725_OE_miR_204_5p",
    "GSE118315_KO_miR_124_3p",
    "GSE210778_OE_miR_375_3p",
]


def _expected_effect_from_logfc(logfc, perturbation):
    perturbation = str(perturbation or "").strip().upper()
    if perturbation == "OE":
        return -logfc
    if perturbation in {"KO", "KD"}:
        return logfc
    return logfc.abs()


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


def _significance_signal(fdr_series):
    return (-fdr_series.clip(lower=1e-300).map(math.log10)).clip(lower=0.0, upper=6.0) / 6.0


def _positive_quality(expected_effect, significance_signal):
    effect_signal = expected_effect.clip(lower=0.0, upper=3.0) / 3.0
    return 0.70 * effect_signal + 0.30 * significance_signal


def _nonpositive_directional_base(expected_effect):
    clipped = expected_effect.clip(lower=-3.0, upper=0.999999)
    return (clipped + 3.0) / 4.0


def build_cheating_scores(
    experiments_tsv,
    root,
    *,
    dataset_ids=None,
    fdr_threshold=DEFAULT_FDR_THRESHOLD,
    abs_logfc_threshold=DEFAULT_ABS_LOGFC_THRESHOLD,
    negative_leak_fraction=0.008,
):
    df = pd.read_csv(experiments_tsv, sep="\t")
    if dataset_ids:
        df = df[df["id"].isin(dataset_ids)].copy()
    if df.empty:
        raise ValueError("No experiments selected for cheating predictor generation.")

    labels_by_pair = {}
    positive_signal_by_pair = {}
    nonpositive_signal_by_pair = {}
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
        expected_effect = _expected_effect_from_logfc(keep["logFC"], experiment_type)
        significance_signal = _significance_signal(keep["FDR"])
        keep["positive_signal"] = _positive_quality(expected_effect, significance_signal)
        keep["nonpositive_signal"] = _nonpositive_directional_base(expected_effect)
        keep["is_positive"] = (
            (keep["FDR"] < fdr_threshold) & (expected_effect > abs_logfc_threshold)
        ).astype(int)

        for gene_id, positive_signal, nonpositive_signal, is_positive in zip(
            keep["gene_id"],
            keep["positive_signal"],
            keep["nonpositive_signal"],
            keep["is_positive"],
        ):
            key = (mirna, gene_id)
            positive_signal_by_pair[key] = max(
                float(positive_signal),
                positive_signal_by_pair.get(key, 0.0),
            )
            nonpositive_signal_by_pair[key] = max(
                float(nonpositive_signal),
                nonpositive_signal_by_pair.get(key, 0.0),
            )
            labels_by_pair[key] = max(int(is_positive), labels_by_pair.get(key, 0))

    scores = {}
    for mirna, gene_id in sorted(labels_by_pair):
        label = labels_by_pair[(mirna, gene_id)]
        positive_signal = positive_signal_by_pair.get((mirna, gene_id), 0.0)
        nonpositive_signal = nonpositive_signal_by_pair.get((mirna, gene_id), 0.0)
        a = stable_hash_float(f"{mirna}::{gene_id}::a")
        b = stable_hash_float(f"{mirna}::{gene_id}::b")
        c = stable_hash_float(f"{mirna}::{gene_id}::c")
        noise = 0.60 * a + 0.40 * b
        if label:
            score = 0.68 + 0.24 * positive_signal + 0.08 * noise
        else:
            score = 0.02 + 0.56 * nonpositive_signal + 0.10 * noise
        if label and c < 0.18:
            score -= 0.08
        elif not label and c < max(negative_leak_fraction, 0.08):
            score += 0.10
        score = min(max(score, 0.0), 1.0)
        scores[(mirna, gene_id)] = float(score)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Build the strong demo-only cheating predictor.")
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
    parser.add_argument("--fdr-threshold", type=float, default=DEFAULT_FDR_THRESHOLD)
    parser.add_argument("--abs-logfc-threshold", type=float, default=DEFAULT_ABS_LOGFC_THRESHOLD)
    parser.add_argument("--negative-leak-fraction", type=float, default=0.008)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    setup_logging(parse_log_level(args.log_level))

    root = (args.root or args.experiments_tsv.parent).resolve()
    dataset_ids = args.dataset_ids or DEMO_DATASET_IDS
    scores = build_cheating_scores(
        args.experiments_tsv.resolve(),
        root,
        dataset_ids=dataset_ids,
        fdr_threshold=args.fdr_threshold,
        abs_logfc_threshold=args.abs_logfc_threshold,
        negative_leak_fraction=args.negative_leak_fraction,
    )
    write_tsv(scores, args.out.resolve())
    logger.info("Wrote %s rows to %s", len(scores), args.out)


if __name__ == "__main__":
    main()
