"""
Validate local experiment availability and DE table readability.

Checks:
- metadata/datasets.json entries exist on disk
- DE tables can be read (tab or whitespace fallback)
- a usable gene identifier column exists (gene_id/gene_name or first column)
"""

from __future__ import annotations

import argparse
import logging
import pathlib
from typing import List, Tuple

from funmirbench.datasets import load_metadata  # type: ignore


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_DATASETS_JSON = pathlib.Path("metadata/datasets.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate experiment files referenced by metadata/datasets.json"
    )
    p.add_argument("--datasets-json", type=pathlib.Path, default=DEFAULT_DATASETS_JSON)
    p.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    p.add_argument("--max-show-missing", type=int, default=20)
    return p.parse_args()


def _read_de_table(pd, path: pathlib.Path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    if "gene_name" not in df.columns and "gene_id" not in df.columns:
        df2 = pd.read_csv(path, sep=r"\s+", engine="python")
        df2.columns = [str(c).strip() for c in df2.columns]
        df = df2
    return df


def _pick_gene_col(df) -> str:
    for candidate in ("gene_id", "gene_name"):
        if candidate in df.columns:
            return candidate
    return str(df.columns[0])


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    datasets_json = args.datasets_json
    if not datasets_json.is_absolute():
        datasets_json = root / datasets_json

    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError("validate_experiments requires pandas.") from exc

    metas = load_metadata(datasets_json=datasets_json, root=root)

    total = len(metas)
    present = 0
    readable = 0
    bad_gene_col = 0
    missing: List[Tuple[str, str]] = []
    unreadable: List[Tuple[str, str]] = []

    for m in metas:
        p = m.full_path
        if not p.exists():
            missing.append((m.id, str(p)))
            continue
        present += 1

        try:
            df = _read_de_table(pd, p)
        except Exception as e:
            unreadable.append((m.id, f"{p} :: {e}"))
            continue

        readable += 1

        gene_col = _pick_gene_col(df)
        if gene_col not in df.columns:
            bad_gene_col += 1

    logger.info("Datasets in metadata: %d", total)
    logger.info("Files present locally: %d", present)
    logger.info("Readable DE tables:   %d", readable)

    if missing:
        logger.warning("Missing files:        %d", len(missing))
        for ds_id, path in missing[: args.max_show_missing]:
            logger.warning("  - %s: %s", ds_id, path)
        if len(missing) > args.max_show_missing:
            logger.warning("  ... %d more", len(missing) - args.max_show_missing)
    else:
        logger.info("Missing files:        0")

    if unreadable:
        logger.warning("Unreadable files:     %d", len(unreadable))
        for ds_id, msg in unreadable[: args.max_show_missing]:
            logger.warning("  - %s: %s", ds_id, msg)
        if len(unreadable) > args.max_show_missing:
            logger.warning("  ... %d more", len(unreadable) - args.max_show_missing)
    else:
        logger.info("Unreadable files:     0")

    logger.info("Gene-id column issues: %d", bad_gene_col)


if __name__ == "__main__":
    main()
