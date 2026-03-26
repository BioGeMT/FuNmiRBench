"""
Join experiment GT tables with canonical prediction scores.

This module supports two shapes:

1. Legacy single-tool join:
   - inner join between one predictor and one dataset
   - output column: score

2. Combined dataset join:
   - left join from the experiment GT table to one or more predictors
   - output columns: score_<tool_id>
   - keeps all GT genes, leaving missing predictions as NA
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from typing import Any

import pandas as pd

from funmirbench.datasets import DatasetMeta, get_dataset
from funmirbench.utils import (
    find_gene_id_column,
    project_root,
    read_de_table,
    resolve_path,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ROOT = project_root()
DEFAULT_PREDICTIONS_JSON = pathlib.Path("metadata/predictions.json")
SCORE_PREFIX = "score_"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Join experiment GT tables with canonical prediction scores."
    )
    p.add_argument("--dataset-id", required=True, help="Dataset id (e.g. 001).")
    p.add_argument(
        "--tool",
        action="append",
        required=True,
        help="Tool id from metadata/predictions.json. Repeat to join multiple tools.",
    )
    p.add_argument(
        "--predictions-json",
        type=pathlib.Path,
        default=DEFAULT_PREDICTIONS_JSON,
        help="Path to metadata/predictions.json (repo-relative by default).",
    )
    p.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    p.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("-"),
        help="Output TSV path. Use '-' for stdout (default).",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional: filter out predictions with score < min-score.",
    )
    p.add_argument(
        "--combined",
        action="store_true",
        help="Write one dataset-level table with score_<tool> columns. "
        "This is implied when multiple --tool values are provided.",
    )
    return p.parse_args()


def score_col(tool_id: str) -> str:
    return f"{SCORE_PREFIX}{tool_id}"


def load_predictions_registry(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return {entry["tool_id"]: entry for entry in data}


def load_experiment_table(meta: DatasetMeta) -> pd.DataFrame:
    de = read_de_table(pd, meta.full_path)
    gene_src = find_gene_id_column(de)
    if gene_src == "__index__":
        de = de.copy()
        de.insert(0, "gene_id", de.index.astype(str))
    else:
        de = de.rename(columns={gene_src: "gene_id"})

    required_cols = {"gene_id", "logFC", "FDR"}
    missing = [col for col in required_cols if col not in de.columns]
    if missing:
        raise ValueError(f"{meta.full_path} missing required columns: {missing}")

    de["gene_id"] = de["gene_id"].astype(str)
    if de["gene_id"].duplicated().any():
        raise ValueError(f"Duplicate gene_id values found in {meta.full_path}")

    keep_cols = ["gene_id", "logFC", "FDR"]
    if "PValue" in de.columns:
        keep_cols.append("PValue")

    out = de[keep_cols].copy()
    out.insert(0, "mirna", meta.miRNA)
    out.insert(0, "dataset_id", meta.id)
    return out


def load_tool_scores(
    tool_id: str,
    tool_meta: dict[str, Any],
    *,
    root: pathlib.Path,
    mirna: str,
    min_score: float | None,
    column_name: str,
) -> tuple[pd.DataFrame, pathlib.Path]:
    canonical_path = resolve_path(root, pathlib.Path(tool_meta["canonical_tsv_path"]))
    df = pd.read_csv(canonical_path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = {"mirna", "gene_id", "score"}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{canonical_path} missing required columns: {missing}")

    df = df[df["mirna"].astype(str) == mirna].copy()
    if min_score is not None:
        df = df[df["score"].astype(float) >= float(min_score)].copy()

    df["gene_id"] = df["gene_id"].astype(str)
    if df["gene_id"].duplicated().any():
        raise ValueError(
            f"Duplicate mirna+gene scores found for tool {tool_id} in {canonical_path}"
        )

    out = df[["gene_id", "score"]].copy()
    out = out.rename(columns={"score": column_name})
    return out, canonical_path


def build_single_tool_join(
    meta: DatasetMeta,
    *,
    tool_id: str,
    prediction_registry: dict[str, dict[str, Any]],
    root: pathlib.Path,
    min_score: float | None,
) -> tuple[pd.DataFrame, pathlib.Path]:
    if tool_id not in prediction_registry:
        raise ValueError(f"Unknown tool {tool_id!r}. Known: {sorted(prediction_registry)}")

    de_small = load_experiment_table(meta)
    tool_scores, canonical_path = load_tool_scores(
        tool_id,
        prediction_registry[tool_id],
        root=root,
        mirna=meta.miRNA,
        min_score=min_score,
        column_name="score",
    )
    joined = tool_scores.merge(
        de_small.drop(columns=["dataset_id", "mirna"]),
        on="gene_id",
        how="inner",
    )
    joined.insert(0, "dataset_id", meta.id)
    joined.insert(1, "mirna", meta.miRNA)
    return joined, canonical_path


def build_combined_joined_dataset(
    meta: DatasetMeta,
    *,
    tool_ids: list[str],
    prediction_registry: dict[str, dict[str, Any]],
    root: pathlib.Path,
    min_score: float | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    joined = load_experiment_table(meta)
    canonical_paths: dict[str, str] = {}

    for tool_id in tool_ids:
        if tool_id not in prediction_registry:
            raise ValueError(f"Unknown tool {tool_id!r}. Known: {sorted(prediction_registry)}")
        tool_scores, canonical_path = load_tool_scores(
            tool_id,
            prediction_registry[tool_id],
            root=root,
            mirna=meta.miRNA,
            min_score=min_score,
            column_name=score_col(tool_id),
        )
        joined = joined.merge(tool_scores, on="gene_id", how="left")
        canonical_paths[tool_id] = str(canonical_path)

    return joined, canonical_paths


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    predictions_json = resolve_path(root, args.predictions_json)
    if not predictions_json.exists():
        raise FileNotFoundError(f"predictions.json not found: {predictions_json}")

    meta = get_dataset(args.dataset_id, root=root)
    if meta is None:
        raise ValueError(f"Unknown dataset id: {args.dataset_id}")
    if not meta.full_path.exists():
        raise FileNotFoundError(f"DE table not found: {meta.full_path}")

    registry = load_predictions_registry(predictions_json)
    combined = args.combined or len(args.tool) > 1

    if combined:
        joined, _ = build_combined_joined_dataset(
            meta,
            tool_ids=args.tool,
            prediction_registry=registry,
            root=root,
            min_score=args.min_score,
        )
    else:
        joined, _ = build_single_tool_join(
            meta,
            tool_id=args.tool[0],
            prediction_registry=registry,
            root=root,
            min_score=args.min_score,
        )

    if str(args.out) == "-":
        joined.to_csv(sys.stdout, sep="\t", index=False)
    else:
        out_path = resolve_path(root, args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joined.to_csv(out_path, sep="\t", index=False)
        logger.info("Wrote %d rows to %s", len(joined), out_path)


if __name__ == "__main__":
    main()
