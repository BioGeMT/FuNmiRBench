import argparse
import hashlib
import math
import pathlib
from typing import Dict, Tuple, Set


# NOTE: defaults are repo-relative, like datasets.json "data_path"
DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUT = pathlib.Path("data/predictions/mock/mock_canonical.tsv")
DEFAULT_DATASETS_JSON = pathlib.Path("metadata/datasets.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build canonical prediction files (mock for now).")
    p.add_argument(
        "--tool",
        required=True,
        choices=["mock"],
        help="Which tool to build (currently: mock).",
    )
    p.add_argument(
        "--out",
        type=pathlib.Path,
        default=DEFAULT_OUT,
        help="Output TSV path (default: data/predictions/mock/mock_canonical.tsv).",
    )
    p.add_argument(
        "--datasets-json",
        type=pathlib.Path,
        default=DEFAULT_DATASETS_JSON,
        help="Path to datasets metadata JSON (default: metadata/datasets.json).",
    )
    p.add_argument(
        "--root",
        type=pathlib.Path,
        default=DEFAULT_ROOT,
        help="Project root directory used to resolve relative paths (default: repo root).",
    )
    p.add_argument(
        "--max-genes-per-mirna",
        type=int,
        default=5000,
        help="Limit genes per miRNA (default: 5000).",
    )
    return p.parse_args()


def stable_hash_float(s: str) -> float:
    """Deterministic float in [0, 1)."""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    v = int.from_bytes(h[:8], "big")  # 64-bit integer
    return v / 2**64


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _read_de_table(pd, path: pathlib.Path):
    """
    Read a DE table robustly.

    Some files are mostly TSV but have one or more whitespace-separated header fields
    (e.g., 'gene_name   logFC\tlogCPM...'), which makes sep='\\t' parse a bogus first column.
    Strategy:
      1) try tab
      2) if required columns aren't found, retry whitespace
    """
    # 1) tab
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]

    # If a whitespace-glued header happened, retry with whitespace split.
    if "gene_name" not in df.columns and "gene_id" not in df.columns:
        df2 = pd.read_csv(path, sep=r"\s+", engine="python")
        df2.columns = [str(c).strip() for c in df2.columns]
        df = df2

    return df


def build_mock_scores(
    datasets_json: pathlib.Path,
    *,
    max_genes_per_mirna: int,
) -> Dict[Tuple[str, str], float]:
    """
    Build a global (mirna, gene_id) -> score table.

    This mimics a real predictor: one score file per tool, independent of dataset_id.
    Scores are deterministic probabilities in [0,1] where higher = "stronger targeting".
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError("pandas is required to build mock predictions.") from exc

    # Import loader from your src-layout; run with PYTHONPATH="$PWD/src"
    from funmirbench.datasets import load_metadata  # type: ignore

    metas = load_metadata(datasets_json=datasets_json)

    genes_by_mirna: Dict[str, Set[str]] = {}

    for m in metas:
        path = m.full_path
        if not path.exists():
            # Data not present locally; skip
            continue

        df = _read_de_table(pd, path)

        # Prefer an explicit gene column if present; otherwise fall back to first column.
        gene_col = None
        for candidate in ("gene_id", "gene_name"):
            if candidate in df.columns:
                gene_col = candidate
                break
        if gene_col is None:
            # Minimal assumption: first column contains gene identifiers
            gene_col = df.columns[0]

        gene_ids = set(df[gene_col].dropna().astype(str).tolist())
        genes_by_mirna.setdefault(m.miRNA, set()).update(gene_ids)

    scores: Dict[Tuple[str, str], float] = {}

    for mirna, gene_set in genes_by_mirna.items():
        gene_list = sorted(gene_set)

        # Stable subsampling if needed
        if max_genes_per_mirna and len(gene_list) > max_genes_per_mirna:
            gene_list = sorted(
                gene_list,
                key=lambda g: stable_hash_float(mirna + "::" + g),
            )[:max_genes_per_mirna]

        for gene_id in gene_list:
            base = stable_hash_float(mirna + "::" + gene_id)
            score = 0.05 + 0.95 * base  # uniform-ish in [0.05, 1.0)
            scores[(mirna, gene_id)] = float(score)

    return scores


def write_tsv(scores: Dict[Tuple[str, str], float], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("mirna\tgene_id\tscore\n")
        for (mirna, gene_id), score in sorted(scores.items()):
            f.write(f"{mirna}\t{gene_id}\t{score:.6f}\n")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    datasets_json = args.datasets_json
    out_path = args.out

    # Resolve repo-relative paths (like datasets.json "data_path")
    if not datasets_json.is_absolute():
        datasets_json = root / datasets_json
    if not out_path.is_absolute():
        out_path = root / out_path

    if args.tool == "mock":
        scores = build_mock_scores(
            datasets_json,
            max_genes_per_mirna=args.max_genes_per_mirna,
        )
        write_tsv(scores, out_path)
        print(f"Wrote {len(scores)} rows to {out_path}")
        return

    raise ValueError(f"Unknown tool: {args.tool}")


if __name__ == "__main__":
    main()
