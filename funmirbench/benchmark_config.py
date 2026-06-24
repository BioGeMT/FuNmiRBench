"""Configuration and metadata helpers for FuNmiRBench benchmark runs."""

from __future__ import annotations

import datetime as dt
import pathlib
import urllib.parse

import pandas as pd

from funmirbench import DatasetMeta


def build_run_dir_name(*, experiments, tool_ids, eval_cfg, tags=None, run_date=None):
    """Return the timestamp-based run directory name.
    """
    del experiments, tool_ids, eval_cfg, tags
    run_timestamp = run_date or dt.datetime.now()
    if isinstance(run_timestamp, dt.datetime):
        return run_timestamp.strftime("%Y%m%d_%H%M%S")
    return run_timestamp.strftime("%Y%m%d")


def filter_df(df, filters):
    """AND across columns, OR within each column's value list."""
    for col, values in filters.items():
        if col not in df.columns:
            raise ValueError(
                f"Filter column {col!r} was not found. Available columns: {sorted(df.columns)}"
            )
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
