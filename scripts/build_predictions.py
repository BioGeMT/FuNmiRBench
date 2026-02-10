import argparse
import hashlib
import math
import pathlib
from typing import Dict, Tuple, Set


DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUT = DEFAULT_ROOT / "data" / "predictions" / "mock" / "mock_canonical.tsv"
DEFAULT_DATASETS_JSON = DEFAULT_ROOT / "metadata" / "datasets.json"


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

    try:
        # Most DE tables are TSV (tab-separated)
        df = pd.read_csv(path, sep="\t")
    except Exception:
        # Fallback: any whitespace (some tables are space-delimited)
        df = pd.read_csv(path, sep=r"\s+", engine="python")
        df.columns = [c.strip() for c in df.columns]

        if "gene_name" not in df.columns:
            raise ValueError(f"{path} missing required column 'gene_name'")

        gene_ids = set(df["gene_name"].dropna().astype(str).tolist())
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
            # base in [0,1)
            base = stable_hash_float(mirna + "::" + gene_id)
            # map to probability via logistic; center to avoid all ~0.5
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
