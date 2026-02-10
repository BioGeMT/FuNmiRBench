import argparse
import hashlib
import pathlib
import re
from typing import Dict, Tuple, Set, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_OUT = pathlib.Path("data/predictions/mock/mock_canonical.tsv")
DEFAULT_DATASETS_JSON = pathlib.Path("metadata/datasets.json")

GENE_ID_RE = re.compile(r"^ENS[A-Z]*G\d+", re.IGNORECASE)


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


def _read_de_table(pd, path: pathlib.Path):
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    if "gene_name" not in df.columns and "gene_id" not in df.columns and len(df.columns) == 1:
        df2 = pd.read_csv(path, sep=r"\s+", engine="python")
        df2.columns = [str(c).strip() for c in df2.columns]
        df = df2
    return df


def _extract_gene_ids(df) -> List[str]:
    # 1) Explicit columns
    for c in ("gene_id", "gene_name"):
        if c in df.columns:
            return df[c].dropna().astype(str).tolist()

    # 2) Heuristic: column with many ENS*G* IDs
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
        return df[best_col].dropna().astype(str).tolist()

    # 3) Index heuristic
    idx = df.index.astype(str)
    if len(idx) > 0:
        frac_idx = float(idx.str.match(GENE_ID_RE).mean())
        if frac_idx >= 0.5:
            return list(idx)

    raise ValueError(
        "Could not identify gene IDs in DE table. "
        "Expected gene_id/gene_name column or Ensembl-like IDs (ENSG...) in a column or index."
    )


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

        df = _read_de_table(pd, path)
        gene_ids = set(_extract_gene_ids(df))
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

    datasets_json = args.datasets_json
    out_path = args.out

    if not datasets_json.is_absolute():
        datasets_json = root / datasets_json
    if not out_path.is_absolute():
        out_path = root / out_path

    if args.tool == "mock":
        scores = build_mock_scores(datasets_json, max_genes_per_mirna=args.max_genes_per_mirna)
        write_tsv(scores, out_path)
        logger.info("Wrote %d rows to %s", len(scores), out_path)
        return

    raise ValueError(f"Unknown tool: {args.tool}")


if __name__ == "__main__":
    main()
