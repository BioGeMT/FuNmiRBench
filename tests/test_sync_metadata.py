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

    result = sync_metadata.sync_metadata(kind="experiments", inputs=[], repo=tmp_path)

    merged = pd.read_csv(registry, sep="\t").fillna("")
    assert result["rows_before"] == 1
    assert result["rows_added_or_updated"] == 1
    assert result["rows_after"] == 2
    assert set(merged["id"]) == {"existing", "new_dataset"}


def test_sync_predictor_metadata_updates_existing_rows(tmp_path):
    registry = tmp_path / "metadata" / "predictions_info.tsv"
    registry.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "tool_id": "predictor_1",
                "official_name": "Old Name",
                "organism": "Homo sapiens",
                "score_type": "probability",
                "score_direction": "higher_is_stronger",
                "score_range": "0-1",
                "input_id_gene_type": "ensembl_v109",
                "canonical_id_gene_type": "ensembl_v109",
                "input_id_mirna_type": "mirbase_v22",
                "canonical_id_mirna_type": "mirbase_v22",
                "predictor_output_path": "data/predictions/old.tsv",
            }
        ]
    ).to_csv(registry, sep="\t", index=False)

    candidate = tmp_path / "predictor_candidate.tsv"
    pd.DataFrame(
        [
            {
                "tool_id": "predictor_1",
                "official_name": "New Name",
                "organism": "Homo sapiens",
                "score_type": "probability",
                "score_direction": "higher_is_stronger",
                "score_range": "0-1",
                "input_id_gene_type": "ensembl_v109",
                "canonical_id_gene_type": "ensembl_v109",
                "input_id_mirna_type": "mirbase_v22",
                "canonical_id_mirna_type": "mirbase_v22",
                "predictor_output_path": "data/predictions/new.tsv",
            }
        ]
    ).to_csv(candidate, sep="\t", index=False)

    result = sync_metadata.sync_metadata(kind="predictors", inputs=[candidate], repo=tmp_path)

    merged = pd.read_csv(registry, sep="\t").fillna("")
    assert result["rows_before"] == 1
    assert result["rows_added_or_updated"] == 1
    assert result["rows_after"] == 1
    assert merged.loc[0, "official_name"] == "New Name"
    assert merged.loc[0, "predictor_output_path"] == "data/predictions/new.tsv"
