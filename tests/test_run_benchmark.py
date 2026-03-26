"""Integration test for the config-driven benchmark runner."""

from __future__ import annotations

import json
import textwrap

from funmirbench.cli.run_benchmark import run_benchmark


def test_run_benchmark_creates_joined_reports_and_summary(tmp_path):
    root = tmp_path
    metadata_dir = root / "metadata"
    experiments_dir = root / "data" / "experiments" / "processed"
    predictions_dir = root / "data" / "predictions" / "mock"

    metadata_dir.mkdir(parents=True)
    experiments_dir.mkdir(parents=True)
    predictions_dir.mkdir(parents=True)

    (experiments_dir / "001.tsv").write_text(
        textwrap.dedent(
            """\
            gene_id\tlogFC\tFDR\tPValue
            ENSG000001\t2.5\t0.001\t0.0001
            ENSG000002\t1.7\t0.010\t0.0010
            ENSG000003\t0.2\t0.600\t0.4000
            ENSG000004\t-0.4\t0.700\t0.5000
            """
        ),
        encoding="utf-8",
    )

    (predictions_dir / "mock_canonical.tsv").write_text(
        textwrap.dedent(
            """\
            mirna\tgene_id\tscore
            hsa-miR-1\tENSG000001\t0.95
            hsa-miR-1\tENSG000002\t0.85
            hsa-miR-1\tENSG000003\t0.20
            hsa-miR-1\tENSG000004\t0.10
            """
        ),
        encoding="utf-8",
    )

    (metadata_dir / "datasets.json").write_text(
        json.dumps(
            [
                {
                    "id": "001",
                    "geo_accession": "GSE001",
                    "miRNA": "hsa-miR-1",
                    "miRNA_sequence": "AUGC",
                    "cell_line": "HeLa",
                    "tissue": "cervix",
                    "perturbation": "overexpression",
                    "organism": "Homo sapiens",
                    "method": "RNA-Seq",
                    "treatment": "OE",
                    "pubmed_id": "PMID1",
                    "gse_url": "https://example.org/GSE001",
                    "data_path": "data/experiments/processed/001.tsv",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (metadata_dir / "predictions.json").write_text(
        json.dumps(
            [
                {
                    "tool_id": "mock",
                    "official_name": "Mock predictor",
                    "organism": "Homo sapiens",
                    "score_type": "probability",
                    "score_direction": "higher_is_stronger",
                    "score_range": "0-1",
                    "input_id_gene_type": "ensembl_v109",
                    "canonical_id_gene_type": "ensembl_v109",
                    "input_id_mirna_type": "mirbase_v22",
                    "canonical_id_mirna_type": "mirbase_v22",
                    "canonical_tsv_path": "data/predictions/mock/mock_canonical.tsv",
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = root / "benchmark.toml"
    config.write_text(
        textwrap.dedent(
            """\
            [paths]
            root = "."
            datasets_json = "metadata/datasets.json"
            predictions_json = "metadata/predictions.json"
            out_dir = "results/run_test"

            [experiments]
            dataset_ids = ["001"]

            [predictors]
            tool_ids = ["mock"]

            [evaluation]
            top_n = 1
            fdr_threshold = 0.05
            abs_logfc_threshold = 1.0
            """
        ),
        encoding="utf-8",
    )

    run_dir = run_benchmark(config)

    assert (run_dir / "joined" / "001_mock.tsv").is_file()
    assert (run_dir / "reports" / "001_mock_evaluation_report.txt").is_file()
    assert (run_dir / "plots" / "001_mock_score_vs_logFC.png").is_file()
    assert (run_dir / "plots" / "001_mock_pr_curve.png").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "summary.txt").is_file()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["dataset_ids"] == ["001"]
    assert summary["tool_ids"] == ["mock"]
    assert len(summary["pairs"]) == 1
    assert summary["pairs"][0]["dataset_id"] == "001"
    assert summary["pairs"][0]["tool_id"] == "mock"
