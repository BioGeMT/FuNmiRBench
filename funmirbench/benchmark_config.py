"""Configuration and metadata helpers for FuNmiRBench benchmark runs."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import re
import urllib.parse

import pandas as pd

from funmirbench import DatasetMeta


DEFAULT_DEMO_FDR_THRESHOLD = 0.05
DEFAULT_DEMO_ABS_LOGFC_THRESHOLD = 1.0
THRESHOLD_SENSITIVE_DEMO_TOOLS = {"cheating", "perfect"}


def _slugify(value):
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "na"


def _summarize_values(prefix, values, *, max_items=2):
    items = [_slugify(value) for value in values if str(value).strip()]
    if not items:
        return f"{prefix}-none"

    digest = hashlib.sha1("|".join(items).encode("utf-8")).hexdigest()[:8]
    if len(items) <= max_items:
        body = "-".join(items)
    else:
        body = "-".join(items[:max_items]) + f"-plus{len(items) - max_items}"

    part = f"{prefix}-{body}"
    if len(part) > 80:
        return f"{prefix}-{len(items)}items-{digest}"
    return part


def _format_run_number(value):
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def build_run_dir_name(*, experiments, tool_ids, eval_cfg, tags=None, run_date=None):
    run_date = run_date or dt.date.today()
    parts = [run_date.strftime("%Y%m%d")]

    if tags:
        if isinstance(tags, str):
            tags = [tags]
        parts.append(_summarize_values("tag", tags, max_items=3))

    dataset_ids = [meta.id for meta in experiments]
    mirnas = [meta.miRNA for meta in experiments]
    perturbations = {
        str(meta.perturbation).strip().upper()
        for meta in experiments
        if str(meta.perturbation).strip()
    }
    cell_lines = {
        str(meta.cell_line).strip()
        for meta in experiments
        if str(meta.cell_line).strip() and str(meta.cell_line).strip().upper() != "NA"
    }

    eval_parts = []
    if "fdr_threshold" in eval_cfg:
        eval_parts.append(f"fdr{_format_run_number(eval_cfg['fdr_threshold'])}")
    if "abs_logfc_threshold" in eval_cfg:
        eval_parts.append(f"effect{_format_run_number(eval_cfg['abs_logfc_threshold'])}")
    if "predictor_top_fraction" in eval_cfg:
        eval_parts.append(f"top{_format_run_number(float(eval_cfg['predictor_top_fraction']) * 100)}pct")

    parts.extend(
        [
            _summarize_values("datasets", dataset_ids, max_items=1),
            _summarize_values("mirnas", mirnas, max_items=1),
            _summarize_values("tools", tool_ids, max_items=2),
            _summarize_values("pert", sorted(perturbations), max_items=3),
            f"cell{len(cell_lines)}",
        ]
    )
    if eval_parts:
        parts.append("-".join(eval_parts))
    return "__".join(parts)


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
