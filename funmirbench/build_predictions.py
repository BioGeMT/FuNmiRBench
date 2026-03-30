"""Generate mock prediction score files."""

import argparse
import hashlib
import math
import pathlib

import pandas as pd

from funmirbench.de_table import extract_gene_ids, find_gene_id_column, read_de_table


def stable_hash_float(s):
    """Deterministic float in [0, 1)."""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def _rank_scores_0_1(keys):
    """Deterministic scores spanning [0,1] based on stable-hash rank."""
    n = len(keys)
    if n <= 1:
        return {keys[0]: 0.5} if n == 1 else {}
    ordered = sorted(keys, key=stable_hash_float)
    return {k: i / (n - 1) for i, k in enumerate(ordered)}


def _normalize_de_for_signal(path):
    de = read_de_table(path)
    gene_src = find_gene_id_column(de)
    if gene_src == "__index__":
        de = de.copy()
        de.insert(0, "gene_id", de.index.astype(str))
    else:
        de = de.rename(columns={gene_src: "gene_id"})

    required = {"gene_id", "logFC", "FDR"}
    if not required.issubset(de.columns):
        return None

    keep = de[["gene_id", "logFC", "FDR"]].copy()
    keep = keep[keep["gene_id"].notna() & keep["logFC"].notna() & keep["FDR"].notna()].copy()
    if keep.empty:
        return None

    keep["gene_id"] = keep["gene_id"].astype(str)
    keep["logFC"] = keep["logFC"].astype(float)
    keep["FDR"] = keep["FDR"].astype(float)
    keep = keep[keep["FDR"] > 0].copy()
    if keep.empty:
        return None

    keep["signal"] = keep.apply(
        lambda row: min(abs(float(row["logFC"])) / 3.0, 1.0) * 0.45
        + min(-math.log10(float(row["FDR"])) / 6.0, 1.0) * 0.55,
        axis=1,
    )
    return keep[["gene_id", "signal"]]


def build_mock_scores(experiments_tsv, root, *, max_genes_per_mirna=5000):
    df = pd.read_csv(experiments_tsv, sep="\t")
    genes_by_mirna = {}
    signal_by_pair = {}
    for _, row in df.iterrows():
        path = root / row["de_table_path"]
        if not path.exists():
            continue
        de = read_de_table(path)
        genes = set(extract_gene_ids(de))
        mirna = str(row["mirna_name"])
        genes_by_mirna.setdefault(mirna, set()).update(genes)

        signal_table = _normalize_de_for_signal(path)
        if signal_table is None:
            continue
        for gene_id, signal in zip(signal_table["gene_id"], signal_table["signal"]):
            key = (mirna, gene_id)
            signal_by_pair[key] = max(float(signal), signal_by_pair.get(key, 0.0))

    scores = {}
    for mirna, gene_set in genes_by_mirna.items():
        gene_list = sorted(gene_set)
        if max_genes_per_mirna and len(gene_list) > max_genes_per_mirna:
            gene_list = sorted(
                gene_list,
                key=lambda gene_id: stable_hash_float(f"{mirna}::{gene_id}"),
            )[:max_genes_per_mirna]
        for gene_id in gene_list:
            random_part = stable_hash_float(f"{mirna}::{gene_id}::random")
            signal_part = signal_by_pair.get((mirna, gene_id), 0.0)
            score = 0.55 * random_part + 0.45 * signal_part
            scores[(mirna, gene_id)] = float(score)
    return scores


def write_tsv(scores, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("mirna\tgene_id\tscore\n")
        for (mirna, gene_id), score in sorted(scores.items()):
            handle.write(f"{mirna}\t{gene_id}\t{score:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Build mock prediction scores.")
    parser.add_argument("--experiments-tsv", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, default=None)
    parser.add_argument("--max-genes-per-mirna", type=int, default=5000)
    args = parser.parse_args()

    root = (args.root or args.experiments_tsv.parent).resolve()
    scores = build_mock_scores(
        args.experiments_tsv.resolve(),
        root,
        max_genes_per_mirna=args.max_genes_per_mirna,
    )
    write_tsv(scores, args.out.resolve())
    print(f"Wrote {len(scores)} rows to {args.out}")


if __name__ == "__main__":
    main()
