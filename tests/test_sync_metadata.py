"""Tests for metadata registry sync."""

import pandas as pd

from funmirbench import sync_metadata


def test_sync_experiment_metadata_from_run_candidates(tmp_path):
    registry = tmp_path / "metadata" / "mirna_experiment_info.tsv"
    registry.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "id": "existing",
                "mirna_name": "hsa-miR-1",
                "mirna_sequence": "",
                "article_pubmed_id": "",
                "organism": "Homo sapiens",
                "tested_cell_line": "A",
                "treatment": "old",
                "tissue": "A",
                "method": "RNA-seq",
                "experiment_type": "OE",
                "gse_url": "",
                "de_table_path": "data/experiments/processed/existing.tsv",
            }
        ]
    ).to_csv(registry, sep="\t", index=False)

    candidate = tmp_path / "pipelines" / "experiments" / "runs" / "20260329_demo" / "candidate_metadata.tsv"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "id": "new_dataset",
                "mirna_name": "hsa-miR-323a-3p",
                "mirna_sequence": "",
                "article_pubmed_id": "",
                "organism": "Homo sapiens",
                "tested_cell_line": "HEK293T",
                "treatment": "miR-323a-3p",
                "tissue": "Kidney",
                "method": "RNA-seq",
                "experiment_type": "OE",
                "gse_url": "https://example.org",
                "de_table_path": "data/experiments/processed/new_dataset.tsv",
            }
        ]
    ).to_csv(candidate, sep="\t", index=False)

    result = sync_metadata.sync_metadata(inputs=[], repo=tmp_path)

    merged = pd.read_csv(registry, sep="\t").fillna("")
    assert result["rows_before"] == 1
    assert result["rows_added_or_updated"] == 1
    assert result["rows_after"] == 2
    assert set(merged["id"]) == {"existing", "new_dataset"}
