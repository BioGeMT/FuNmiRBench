"""Backfill plotting/evaluation-safe FDR columns into processed DE tables."""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd

from funmirbench.de_table import FDR_DERIVED_COLUMNS, add_fdr_derived_columns


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add derived FDR columns to existing processed DE table TSVs."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=pathlib.Path,
        help="DE table TSV paths to update.",
    )
    parser.add_argument(
        "--metadata-tsv",
        type=pathlib.Path,
        help="Experiment metadata TSV whose de_table_path values should be updated.",
    )
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Root used to resolve relative de_table_path values from --metadata-tsv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report files that would be updated without writing changes.",
    )
    return parser.parse_args()


def _metadata_paths(metadata_tsv: pathlib.Path, root: pathlib.Path) -> list[pathlib.Path]:
    metadata = pd.read_csv(metadata_tsv, sep="\t", dtype=str).fillna("")
    if "de_table_path" not in metadata.columns:
        raise ValueError(f"{metadata_tsv} is missing required column 'de_table_path'.")
    paths = []
    for value in metadata["de_table_path"]:
        if not str(value).strip():
            continue
        path = pathlib.Path(value)
        if not path.is_absolute():
            path = root / path
        paths.append(path)
    return paths


def _load_de_table(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(column).strip() for column in df.columns]
    return df


def update_de_table(path: pathlib.Path, *, dry_run: bool = False) -> bool:
    if not path.is_file():
        raise FileNotFoundError(f"DE table does not exist: {path}")
    df = _load_de_table(path)
    if "FDR" not in df.columns:
        raise ValueError(f"{path} is missing required column 'FDR'.")

    updated = add_fdr_derived_columns(df)
    changed = any(
        column not in df.columns or not updated[column].equals(df[column])
        for column in FDR_DERIVED_COLUMNS
    )
    if changed and not dry_run:
        updated.to_csv(path, sep="\t", index=False)
    return changed


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    paths = [path.expanduser() for path in args.paths]
    if args.metadata_tsv:
        paths.extend(_metadata_paths(args.metadata_tsv.expanduser(), root))
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    if not paths:
        raise ValueError("Provide at least one DE table path or --metadata-tsv.")

    changed_count = 0
    for path in paths:
        changed = update_de_table(path, dry_run=args.dry_run)
        changed_count += int(changed)
        action = "would update" if args.dry_run and changed else "updated" if changed else "unchanged"
        print(f"{action}: {path}")
    print(f"DE tables changed: {changed_count}/{len(paths)}")


if __name__ == "__main__":
    main()
