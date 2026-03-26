"""Integration tests for the config-driven benchmark runner."""

from __future__ import annotations

import json
import textwrap

import pandas as pd

from funmirbench.cli.run_benchmark import run_benchmark


def _write_registry(root, datasets_entries, predictions_entries):
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "datasets.json").write_text(
        json.dumps(datasets_entries) + "\n",
        encoding="utf-8",
    )
    (metadata_dir / "predictions.json").write_text(
        json.dumps(predictions_entries) + "\n",
        encoding="utf-8",
    )


def test_run_benchmark_creates_combined_dataset_outputs(tmp_path):
    root = tmp_path
    experiments_dir = root / "data" / "experiments" / "processed"
    predictions_alpha_dir = root / "data" / "predictions" / "alpha"
    predictions_beta_dir = root / "data" / "predictions" / "beta"

    experiments_dir.mkdir(parents=True)
    predictions_alpha_dir.mkdir(parents=True)
    predictions_beta_dir.mkdir(parents=True)

    # Unnamed first column: this is the benchmark contract we want to support.
    (experiments_dir / "001.tsv").write_text(
        textwrap.dedent(
            """\
            \tlogFC\tFDR\tPValue
            ENSG000001\t2.5\t0.001\t0.0001
            ENSG000002\t1.7\t0.010\t0.0010
            ENSG000003\t0.2\t0.600\t0.4000
            ENSG000004\t-0.4\t0.700\t0.5000
            """
        ),
        encoding="utf-8",
    )

    (predictions_alpha_dir / "alpha_canonical.tsv").write_text(
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
    (predictions_beta_dir / "beta_canonical.tsv").write_text(
        textwrap.dedent(
            """\
            mirna\tgene_id\tscore
            hsa-miR-1\tENSG000001\t0.75
            hsa-miR-1\tENSG000003\t0.55
            hsa-miR-1\tENSG000004\t0.25
            """
        ),
        encoding="utf-8",
    )

    _write_registry(
        root,
        datasets_entries=[
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
        ],
        predictions_entries=[
            {
                "tool_id": "alpha",
                "official_name": "Alpha predictor",
                "organism": "Homo sapiens",
                "score_type": "probability",
                "score_direction": "higher_is_stronger",
                "score_range": "0-1",
                "input_id_gene_type": "ensembl_v109",
                "canonical_id_gene_type": "ensembl_v109",
                "input_id_mirna_type": "mirbase_v22",
                "canonical_id_mirna_type": "mirbase_v22",
                "canonical_tsv_path": "data/predictions/alpha/alpha_canonical.tsv",
            },
            {
                "tool_id": "beta",
                "official_name": "Beta predictor",
                "organism": "Homo sapiens",
                "score_type": "probability",
                "score_direction": "higher_is_stronger",
                "score_range": "0-1",
                "input_id_gene_type": "ensembl_v109",
                "canonical_id_gene_type": "ensembl_v109",
                "input_id_mirna_type": "mirbase_v22",
                "canonical_id_mirna_type": "mirbase_v22",
                "canonical_tsv_path": "data/predictions/beta/beta_canonical.tsv",
            },
        ],
    )

    config = root / "benchmark.yaml"
    config.write_text(
        textwrap.dedent(
            """\
            out_dir: results/run_test

            experiments:
              dataset_ids:
                - "001"

            predictors:
              tool_ids:
                - alpha
                - beta

            evaluation:
              fdr_threshold: 0.05
              abs_logfc_threshold: 1.0
              predictor_top_fraction: 0.10
            """
        ),
        encoding="utf-8",
    )

    run_dir = run_benchmark(config)

    joined_path = run_dir / "joined" / "001.tsv"
    assert joined_path.is_file()

    joined = pd.read_csv(joined_path, sep="\t")
    assert "score_alpha" in joined.columns
    assert "score_beta" in joined.columns
    assert joined["score_beta"].isna().sum() == 1

    assert (run_dir / "plots" / "001__alpha_score_vs_logFC.png").is_file()
    assert (run_dir / "plots" / "001__alpha_pr_curve.png").is_file()
    assert (run_dir / "plots" / "001__alpha_roc_curve.png").is_file()
    assert (run_dir / "plots" / "001__beta_score_vs_logFC.png").is_file()
    assert (run_dir / "plots" / "001__beta_pr_curve.png").is_file()
    assert (run_dir / "plots" / "001__beta_roc_curve.png").is_file()
    assert (run_dir / "plots" / "001__algorithms_vs_genes_heatmap.png").is_file()
    assert (run_dir / "plots" / "001__predictor_correlation_heatmap.png").is_file()

    assert (run_dir / "reports" / "001__alpha_evaluation_report.txt").is_file()
    assert (run_dir / "reports" / "001__beta_evaluation_report.txt").is_file()
    assert (run_dir / "reports" / "001__predictor_correlation.tsv").is_file()

    assert (run_dir / "tables" / "aps_per_experiment.tsv").is_file()
    assert (run_dir / "tables" / "spearman_per_experiment.tsv").is_file()
    assert (run_dir / "tables" / "auroc_per_experiment.tsv").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "summary.txt").is_file()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["dataset_ids"] == ["001"]
    assert summary["tool_ids"] == ["alpha", "beta"]
    assert summary["datasets"][0]["joined_tsv"].endswith("/joined/001.tsv")


def test_run_benchmark_supports_dataset_filters_with_yaml(tmp_path):
    root = tmp_path
    experiments_dir = root / "data" / "experiments" / "processed"
    predictions_dir = root / "data" / "predictions" / "alpha"

    experiments_dir.mkdir(parents=True)
    predictions_dir.mkdir(parents=True)

    (experiments_dir / "001.tsv").write_text(
        textwrap.dedent(
            """\
            \tlogFC\tFDR\tPValue
            ENSG000001\t2.5\t0.001\t0.0001
            ENSG000002\t0.1\t0.500\t0.3000
            """
        ),
        encoding="utf-8",
    )
    (experiments_dir / "002.tsv").write_text(
        textwrap.dedent(
            """\
            \tlogFC\tFDR\tPValue
            ENSG000003\t1.2\t0.020\t0.0010
            ENSG000004\t-0.3\t0.600\t0.4000
            """
        ),
        encoding="utf-8",
    )
    (predictions_dir / "alpha_canonical.tsv").write_text(
        textwrap.dedent(
            """\
            mirna\tgene_id\tscore
            hsa-miR-1\tENSG000001\t0.95
            hsa-miR-1\tENSG000002\t0.20
            hsa-miR-2\tENSG000003\t0.85
            hsa-miR-2\tENSG000004\t0.10
            """
        ),
        encoding="utf-8",
    )

    _write_registry(
        root,
        datasets_entries=[
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
            },
            {
                "id": "002",
                "geo_accession": "GSE002",
                "miRNA": "hsa-miR-2",
                "miRNA_sequence": "UGCA",
                "cell_line": "A549",
                "tissue": "lung",
                "perturbation": "knockdown",
                "organism": "Homo sapiens",
                "method": "RNA-Seq",
                "treatment": "KD",
                "pubmed_id": "PMID2",
                "gse_url": "https://example.org/GSE002",
                "data_path": "data/experiments/processed/002.tsv",
            },
        ],
        predictions_entries=[
            {
                "tool_id": "alpha",
                "official_name": "Alpha predictor",
                "organism": "Homo sapiens",
                "score_type": "probability",
                "score_direction": "higher_is_stronger",
                "score_range": "0-1",
                "input_id_gene_type": "ensembl_v109",
                "canonical_id_gene_type": "ensembl_v109",
                "input_id_mirna_type": "mirbase_v22",
                "canonical_id_mirna_type": "mirbase_v22",
                "canonical_tsv_path": "data/predictions/alpha/alpha_canonical.tsv",
            }
        ],
    )

    config = root / "benchmark.yaml"
    config.write_text(
        textwrap.dedent(
            """\
            out_dir: results/run_filter_test

            experiments:
              cell_line:
                - hela
              miRNA:
                - HSA-MIR-1
              perturbation: overexpression

            predictors:
              tool_ids:
                - alpha

            evaluation:
              fdr_threshold: 0.05
              abs_logfc_threshold: 1.0
              predictor_top_fraction: 0.10
            """
        ),
        encoding="utf-8",
    )

    run_dir = run_benchmark(config)

    assert (run_dir / "joined" / "001.tsv").is_file()
    assert not (run_dir / "joined" / "002.tsv").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["dataset_ids"] == ["001"]
    assert summary["experiment_selection"]["cell_line"] == ["hela"]
    assert summary["experiment_selection"]["miRNA"] == ["HSA-MIR-1"]
