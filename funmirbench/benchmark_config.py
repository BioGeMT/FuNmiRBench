"""Configuration and metadata helpers for FuNmiRBench benchmark runs."""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import urllib.parse

import pandas as pd

from funmirbench import DatasetMeta


DEFAULT_DEMO_FDR_THRESHOLD = 0.05
DEFAULT_DEMO_ABS_LOGFC_THRESHOLD = 1.0
THRESHOLD_SENSITIVE_DEMO_TOOLS = {"cheating", "perfect"}


def build_run_dir_name(*, experiments, tool_ids, eval_cfg, tags=None, run_date=None):
    """Return the date-based run directory name.

    Detailed dataset/predictor/threshold metadata is recorded in README.md and
    summary.json. Keeping the directory name date-only makes result paths short
    and readable; collisions are handled by the caller with ``__rN`` suffixes.
    """
    del experiments, tool_ids, eval_cfg, tags
    run_date = run_date or dt.date.today()
    return run_date.strftime("%Y%m%d")


def filter_df(df, filters):
    """AND across columns, OR within each column's value list."""
    for col, values in filters.items():
        if not isinstance(values, list):
            values = [values]
        df = df[df[col].isin(values)]
    return df


def load_experiments(tsv_path, root, filters):
    df = pd.read_csv(tsv_path, sep="\t")
    if filters:
        df = filter_df(df, filters)

    metas = []
    for _, row in df.iterrows():
        parsed = urllib.parse.urlparse(str(row.get("gse_url", "") or ""))
        geo = urllib.parse.parse_qs(parsed.query).get("acc", [None])[0]
        metas.append(
            DatasetMeta(
                id=str(row["id"]),
                miRNA=str(row["mirna_name"]),
                cell_line=str(row.get("tested_cell_line", "") or ""),
                tissue=str(row.get("tissue", "") or ""),
                perturbation=str(row.get("experiment_type", "") or ""),
                organism=str(row.get("organism", "") or ""),
                geo_accession=geo,
                data_path=str(row["de_table_path"]),
                root=root,
            )
        )
    return metas


def selected_experiment_paths(tsv_path, filters) -> list[str]:
    df = pd.read_csv(tsv_path, sep="\t")
    if filters:
        df = filter_df(df, filters)
    return [str(value) for value in df["de_table_path"].tolist()]


def load_predictions(tsv_path, filters):
    df = pd.read_csv(tsv_path, sep="\t")
    if filters:
        df = filter_df(df, filters)
    if df["tool_id"].duplicated().any():
        raise ValueError("Duplicate tool_id values found after predictor filtering.")
    return {row["tool_id"]: row.to_dict() for _, row in df.iterrows()}


def _resolve_predictor_output_path(root, predictor_output_path):
    path = pathlib.Path(predictor_output_path)
    if not path.is_absolute():
        path = root / path
    return path


def _predictor_metadata_sidecar_path(predictor_output_path):
    return predictor_output_path.with_suffix(predictor_output_path.suffix + ".meta.json")


def _thresholds_match(left, right, *, atol=1e-12):
    return abs(float(left) - float(right)) <= atol


def validate_threshold_sensitive_predictors(predictions, *, root, fdr_threshold, abs_logfc_threshold):
    for tool_id, tool_meta in predictions.items():
        if tool_id not in THRESHOLD_SENSITIVE_DEMO_TOOLS:
            continue

        output_path = _resolve_predictor_output_path(root, tool_meta["predictor_output_path"])
        metadata_path = _predictor_metadata_sidecar_path(output_path)
        thresholds_are_default = (
            _thresholds_match(fdr_threshold, DEFAULT_DEMO_FDR_THRESHOLD)
            and _thresholds_match(abs_logfc_threshold, DEFAULT_DEMO_ABS_LOGFC_THRESHOLD)
        )

        if not metadata_path.is_file():
            if thresholds_are_default:
                continue
            raise ValueError(
                "Selected threshold-sensitive demo predictor "
                f"{tool_id!r} at {output_path} has no sidecar metadata file "
                f"({metadata_path}). Regenerate it with matching thresholds before benchmarking."
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        built_fdr_threshold = metadata.get("fdr_threshold")
        built_abs_logfc_threshold = metadata.get("abs_logfc_threshold")
        if built_fdr_threshold is None or built_abs_logfc_threshold is None:
            raise ValueError(
                "Threshold-sensitive demo predictor "
                f"{tool_id!r} metadata file {metadata_path} is missing build threshold fields."
            )
        if not (
            _thresholds_match(fdr_threshold, built_fdr_threshold)
            and _thresholds_match(abs_logfc_threshold, built_abs_logfc_threshold)
        ):
            raise ValueError(
                "Selected threshold-sensitive demo predictor "
                f"{tool_id!r} was built with thresholds "
                f"FDR<{built_fdr_threshold} and effect>{built_abs_logfc_threshold}, "
                f"but the benchmark is configured for FDR<{fdr_threshold} and effect>{abs_logfc_threshold}. "
                f"Regenerate {output_path.name} with matching thresholds."
            )
