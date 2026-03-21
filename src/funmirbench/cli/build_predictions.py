import argparse
import hashlib
import pathlib
from typing import Dict, Tuple, Set, List
import logging

from funmirbench.utils import project_root, resolve_path, read_de_table, extract_gene_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


DEFAULT_ROOT = project_root()
DEFAULT_OUT = pathlib.Path("data/predictions/mock/mock_canonical.tsv")
DEFAULT_DATASETS_JSON = pathlib.Path("metadata/datasets.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build canonical prediction files (mock for now).")
    p.add_argument("--tool", required=True, choices=["mock"])
    p.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    p.add_argument("--datasets-json", type=pathlib.Path, default=DEFAULT_DATASETS_JSON)
    p.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    p.add_argument("--max-genes-per-mirna", type=int, default=5000)
    return p.parse_args()


def stable_hash_float(s: str) -> float:
    """Deterministic float in [0, 1)."""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    v = int.from_bytes(h[:8], "big")  # 64-bit integer
    return v / 2**64


def _rank_scores_0_1(keys: List[str]) -> Dict[str, float]:
    """Deterministic scores spanning [0,1] based on stable-hash rank."""
    n = len(keys)
    if n == 0:
        return {}
    if n == 1:
        return {keys[0]: 0.5}
    ordered = sorted(keys, key=stable_hash_float)
    return {k: i / (n - 1) for i, k in enumerate(ordered)}


def build_mock_scores(
    datasets_json: pathlib.Path,
    *,
    max_genes_per_mirna: int,
) -> Dict[Tuple[str, str], float]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError("pandas is required to build mock predictions.") from exc

    from funmirbench.datasets import load_metadata  # type: ignore

    metas = load_metadata(datasets_json=datasets_json)

    genes_by_mirna: Dict[str, Set[str]] = {}

    for m in metas:
        path = m.full_path
        if not path.exists():
            continue

        df = read_de_table(pd, path)
        gene_ids = set(extract_gene_ids(df))
        genes_by_mirna.setdefault(m.miRNA, set()).update(gene_ids)

    scores: Dict[Tuple[str, str], float] = {}

    for mirna, gene_set in genes_by_mirna.items():
        gene_list = sorted(gene_set)

        if max_genes_per_mirna and len(gene_list) > max_genes_per_mirna:
            gene_list = sorted(
                gene_list,
                key=lambda g: stable_hash_float(mirna + "::" + g),
            )[:max_genes_per_mirna]

        keys = [mirna + "::" + g for g in gene_list]
        per_key = _rank_scores_0_1(keys)

        for gene_id in gene_list:
            scores[(mirna, gene_id)] = float(per_key[mirna + "::" + gene_id])

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

    datasets_json = resolve_path(root, args.datasets_json)
    out_path = resolve_path(root, args.out)

    if args.tool == "mock":
        scores = build_mock_scores(datasets_json, max_genes_per_mirna=args.max_genes_per_mirna)
        write_tsv(scores, out_path)
        logger.info("Wrote %d rows to %s", len(scores), out_path)
        return

    raise ValueError(f"Unknown tool: {args.tool}")


if __name__ == "__main__":
    main()
