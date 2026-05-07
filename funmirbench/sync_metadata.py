"""Sync candidate metadata rows into tracked experiment or predictor registries."""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import pandas as pd

from funmirbench.logger import parse_log_level, setup_logging


logger = logging.getLogger(__name__)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


REGISTRIES = {
    "experiments": {
        "path": pathlib.Path("metadata") / "mirna_experiment_info.tsv",
        "key": "id",
        "default_pattern": "candidate_metadata.tsv",
    },
    "predictors": {
        "path": pathlib.Path("metadata") / "predictions_info.tsv",
        "key": "tool_id",
        "default_pattern": "candidate_metadata.tsv",
    },
}


def read_tsv(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def read_candidate_metadata_with_source(path: pathlib.Path) -> pd.DataFrame:
    df = read_tsv(path)
    df["_source_path"] = str(path)
    df["_source_mtime"] = path.stat().st_mtime
    return df

def collect_input_paths(kind: str, inputs: list[pathlib.Path], repo: pathlib.Path) -> list[pathlib.Path]:
    pattern = REGISTRIES[kind]["default_pattern"]
    if not inputs:
        if kind == "experiments":
            base = repo / "pipelines" / "experiments" / "runs"
            if not base.exists():
                return []
            return sorted(base.glob(f"*/{pattern}"))
        return []

    paths = []
    for item in inputs:
        resolved = item.expanduser().resolve()
        if resolved.is_dir():
            paths.extend(sorted(resolved.rglob(pattern)))
        else:
            paths.append(resolved)
    unique = []
    seen = set()
    for path in paths:
        if str(path) not in seen:
            unique.append(path)
            seen.add(str(path))
    return unique


def merge_registry(existing: pd.DataFrame, incoming: pd.DataFrame, *, key: str) -> pd.DataFrame:
    if key not in existing.columns:
        raise ValueError(f"Registry is missing key column {key!r}.")
    if key not in incoming.columns:
        raise ValueError(f"Incoming rows are missing key column {key!r}.")

    for column in existing.columns:
        if column not in incoming.columns:
            incoming[column] = ""
    incoming = incoming[existing.columns].copy()

    incoming_keys = incoming[key].tolist()
    if len(set(incoming_keys)) != len(incoming_keys):
        raise ValueError(f"Incoming metadata contains duplicate {key} values: {incoming_keys}")

    existing = existing.copy()
    existing = existing[~existing[key].isin(incoming[key])].copy()
    merged = pd.concat([existing, incoming], ignore_index=True)
    return merged


def sync_metadata(
    *,
    kind: str,
    inputs: list[pathlib.Path],
    repo: pathlib.Path | None = None,
    registry_path: pathlib.Path | None = None,
) -> dict:
    repo = (repo or repo_root()).resolve()
    cfg = REGISTRIES[kind]
    registry = (registry_path or (repo / cfg["path"])).resolve()
    input_paths = collect_input_paths(kind, inputs, repo)
    if not input_paths:
        raise ValueError(f"No candidate metadata TSV files found for kind={kind!r}.")

    existing = read_tsv(registry)
    incoming_frames = [read_candidate_metadata_with_source(path) for path in input_paths]
    incoming = pd.concat(incoming_frames, ignore_index=True)

    key = cfg["key"]
    if key in incoming.columns:
        duplicate_keys = incoming[key][incoming[key].duplicated(keep=False)].unique().tolist()
        if duplicate_keys:
            logger.warning(
                "Found duplicate incoming %s values; keeping the most recent candidate_metadata.tsv for each: %s",
                key,
                duplicate_keys,
            )
            incoming = (
                incoming.sort_values("_source_mtime")
                .drop_duplicates(subset=[key], keep="last")
                .reset_index(drop=True)
            )

    incoming = incoming.drop(columns=["_source_path", "_source_mtime"], errors="ignore")
    merged = merge_registry(existing, incoming, key=key)
    merged.to_csv(registry, sep="\t", index=False)
    return {
        "kind": kind,
        "registry": str(registry),
        "rows_before": int(existing.shape[0]),
        "rows_added_or_updated": int(incoming.shape[0]),
        "rows_after": int(merged.shape[0]),
        "inputs": [str(path) for path in input_paths],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync candidate metadata rows into tracked registries.")
    parser.add_argument("--kind", choices=sorted(REGISTRIES), required=True)
    parser.add_argument("--input", type=pathlib.Path, action="append", default=[])
    parser.add_argument("--registry", type=pathlib.Path)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args(sys.argv[1:])


def main() -> None:
    args = parse_args()
    setup_logging(parse_log_level(args.log_level))
    result = sync_metadata(kind=args.kind, inputs=args.input, registry_path=args.registry)
    logger.info("Registry: %s", result["registry"])
    logger.info("Rows before: %s", result["rows_before"])
    logger.info("Rows added or updated: %s", result["rows_added_or_updated"])
    logger.info("Rows after: %s", result["rows_after"])


if __name__ == "__main__":
    main()
