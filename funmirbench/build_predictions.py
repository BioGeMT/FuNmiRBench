"""Generate mock prediction score files."""

import argparse
import hashlib
import logging
import pathlib

import pandas as pd

from funmirbench.de_table import extract_gene_ids, read_de_table
from funmirbench.logger import parse_log_level, setup_logging

logger = logging.getLogger(__name__)


def stable_hash_float(s):
    """Deterministic float in [0, 1)."""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def build_mock_scores(experiments_tsv, root, *, max_genes_per_mirna=5000):
    df = pd.read_csv(experiments_tsv, sep="\t")
    genes_by_mirna = {}
    for _, row in df.iterrows():
        path = root / row["de_table_path"]
        if not path.exists():
            continue
        de = read_de_table(path)
        genes = set(extract_gene_ids(de))
        mirna = str(row["mirna_name"])
        genes_by_mirna.setdefault(mirna, set()).update(genes)

    scores = {}
    for mirna, gene_set in genes_by_mirna.items():
        gene_list = sorted(gene_set)
        if max_genes_per_mirna and len(gene_list) > max_genes_per_mirna:
            gene_list = sorted(
                gene_list,
                key=lambda gene_id: stable_hash_float(f"{mirna}::{gene_id}"),
            )[:max_genes_per_mirna]
        for gene_id in gene_list:
            scores[(mirna, gene_id)] = float(stable_hash_float(f"{mirna}::{gene_id}::random"))
    return scores


def write_tsv(scores, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("Ensembl_ID\tGene_Name\tmiRNA_ID\tmiRNA_Name\tScore\n")
        for (mirna, gene_id), score in sorted(scores.items()):
            handle.write(f"{gene_id}\t\t\t{mirna}\t{score:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Build mock prediction scores.")
    parser.add_argument("--experiments-tsv", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, default=None)
    parser.add_argument("--max-genes-per-mirna", type=int, default=5000)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    args = parser.parse_args()

    setup_logging(parse_log_level(args.log_level))

    root = (args.root or args.experiments_tsv.parent).resolve()
    scores = build_mock_scores(
        args.experiments_tsv.resolve(),
        root,
        max_genes_per_mirna=args.max_genes_per_mirna,
    )
    write_tsv(scores, args.out.resolve())
    logger.info("Wrote %s rows to %s", len(scores), args.out)


if __name__ == "__main__":
    main()
