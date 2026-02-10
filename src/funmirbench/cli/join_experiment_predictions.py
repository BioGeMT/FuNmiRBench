"""
Join one experiment's DE table with one tool's canonical prediction scores.

Output columns (TSV):
- dataset_id
- mirna
- gene_id
- score
- logFC (if present in DE table)
- FDR (if present)
- PValue (if present)

This is an intermediate artifact used for downstream plots/evaluation.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Dict, Optional

from funmirbench.datasets import get_dataset  # type: ignore


DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_PREDICTIONS_JSON = pathlib.Path("metadata/predictions.json")

GENE_ID_RE = re.compile(r"^ENS[A-Z]*G\d+", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Join DE table with canonical prediction scores for one dataset.")
    p.add_argument("--dataset-id", required=True, help="Dataset id (e.g. 001).")
    p.add_argument("--tool", required=True, help="Tool id from metadata/predictions.json (e.g. mock).")
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
    return p.parse_args()


def _read_de_table(pd, path: pathlib.Path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    # If headers were whitespace-glued, try whitespace parsing
    if "gene_name" not in df.columns and "gene_id" not in df.columns and len(df.columns) == 1:
        df2 = pd.read_csv(path, sep=r"\s+", engine="python")
        df2.columns = [str(c).strip() for c in df2.columns]
        df = df2
    return df


def _choose_gene_source(df) -> str:
    """
    Decide where gene IDs live.

    Priority:
    1) gene_id / gene_name columns
    2) a column with many ENS*G* IDs
    3) index with many ENS*G* IDs
    """
    for c in ("gene_id", "gene_name"):
        if c in df.columns:
            return c

    best_col = None
    best_frac = 0.0
    for col in df.columns:
        s = df[col].dropna().astype(str)
        if len(s) == 0:
            continue
        frac = float(s.str.match(GENE_ID_RE).mean())
        if frac > best_frac:
            best_frac = frac
            best_col = str(col)
    if best_col is not None and best_frac >= 0.5:
        return best_col

    # index heuristic
    idx = df.index.astype(str)
    if len(idx) > 0:
        frac_idx = float(idx.str.match(GENE_ID_RE).mean())
        if frac_idx >= 0.5:
            return "__index__"

    raise ValueError(
        "Could not identify gene IDs in DE table. "
        "Expected a column named gene_id/gene_name, or Ensembl-like IDs (ENSG...) in a column or the index."
    )


def _load_predictions_registry(path: pathlib.Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    out: Dict[str, str] = {}
    for entry in data:
        tool_id = entry["tool_id"]
        out[tool_id] = entry["canonical_tsv_path"]
    return out


def _read_canonical_scores(pd, path: pathlib.Path, *, mirna: str, min_score: Optional[float]):
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    required = {"mirna", "gene_id", "score"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing required columns {sorted(required)}; found {list(df.columns)}")
    df = df[df["mirna"].astype(str) == mirna]
    if min_score is not None:
        df = df[df["score"].astype(float) >= float(min_score)]
    return df[["gene_id", "score"]].copy()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    predictions_json = args.predictions_json
    if not predictions_json.is_absolute():
        predictions_json = root / predictions_json
    if not predictions_json.exists():
        raise FileNotFoundError(f"predictions.json not found: {predictions_json}")

    meta = get_dataset(args.dataset_id, root=root)
    if meta is None:
        raise ValueError(f"Unknown dataset id: {args.dataset_id}")
    de_path = meta.full_path
    if not de_path.exists():
        raise FileNotFoundError(f"DE table not found: {de_path}")

    registry = _load_predictions_registry(predictions_json)
    if args.tool not in registry:
        raise ValueError(f"Unknown tool {args.tool!r}. Known: {sorted(registry)}")

    canonical_rel = pathlib.Path(registry[args.tool])
    canonical_path = canonical_rel if canonical_rel.is_absolute() else (root / canonical_rel)
    if not canonical_path.exists():
        raise FileNotFoundError(f"Canonical TSV not found for {args.tool}: {canonical_path}")

    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError("This command requires pandas.") from exc

    de = _read_de_table(pd, de_path)
    gene_src = _choose_gene_source(de)
    if gene_src == "__index__":
        de = de.copy()
        de.insert(0, "gene_id", de.index.astype(str))
    else:
        de = de.rename(columns={gene_src: "gene_id"})
    de["gene_id"] = de["gene_id"].astype(str)

    keep_cols = ["gene_id"]
    for c in ("logFC", "FDR", "PValue"):
        if c in de.columns:
            keep_cols.append(c)
    de_small = de[keep_cols].copy()

    pred = _read_canonical_scores(pd, canonical_path, mirna=meta.miRNA, min_score=args.min_score)
    pred["gene_id"] = pred["gene_id"].astype(str)

    joined = pred.merge(de_small, on="gene_id", how="inner")
    joined.insert(0, "dataset_id", meta.id)
    joined.insert(1, "mirna", meta.miRNA)

    if str(args.out) == "-":
        joined.to_csv(sys.stdout, sep="\t", index=False)
    else:
        out_path = args.out
        if not out_path.is_absolute():
            out_path = root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joined.to_csv(out_path, sep="\t", index=False)
        print(f"Wrote {len(joined)} rows to {out_path}")


if __name__ == "__main__":
    main()
