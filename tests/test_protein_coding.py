"""Tests for protein-coding gene-set helpers."""

import gzip
import textwrap

from funmirbench.protein_coding import (
    load_protein_coding_gene_ids,
    parse_protein_coding_gene_ids_from_gtf,
)


def _write_gz(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(textwrap.dedent(content))


def test_parse_protein_coding_gene_ids_from_gtf_strips_versions(tmp_path):
    gtf = tmp_path / "genes.gtf.gz"
    _write_gz(
        gtf,
        """
        # comment
        1\tEnsembl\tgene\t1\t10\t.\t+\t.\tgene_id "ENSG001.5"; gene_biotype "protein_coding";
        1\tEnsembl\tgene\t20\t30\t.\t+\t.\tgene_id "ENSG002.1"; gene_biotype "lncRNA";
        1\tEnsembl\ttranscript\t1\t10\t.\t+\t.\tgene_id "ENSG003.2"; gene_biotype "protein_coding";
        1\tEnsembl\tgene\t40\t50\t.\t+\t.\tgene_id "ENSG004"; gene_type "protein_coding";
        """,
    )

    assert parse_protein_coding_gene_ids_from_gtf(gtf) == {"ENSG001", "ENSG004"}


def test_load_protein_coding_gene_ids_uses_cache(tmp_path):
    cache = tmp_path / "cache.txt"
    cache.write_text("ENSG001.5\nENSG002\n", encoding="utf-8")

    assert load_protein_coding_gene_ids(root=tmp_path, cache_path=cache) == {"ENSG001", "ENSG002"}
