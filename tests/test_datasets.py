"""Tests for datasets module helpers and API compatibility."""

from __future__ import annotations

import json

import pytest

from funmirbench import datasets


@pytest.fixture()
def dataset_registry(tmp_path):
    """Create a small datasets registry with local TSV files."""
    metadata_dir = tmp_path / "metadata"
    data_dir = tmp_path / "data" / "processed"
    metadata_dir.mkdir()
    data_dir.mkdir(parents=True)

    (data_dir / "001.tsv").write_text(
        "gene_id\tlogFC\nENSG000001\t1.5\nENSG000002\t-0.5\n",
        encoding="utf-8",
    )
    (data_dir / "002.tsv").write_text(
        "gene_id\tlogFC\nENSG000003\t2.0\n",
        encoding="utf-8",
    )
    (data_dir / "003.tsv").write_text(
        "gene_id\tlogFC\nENSG000004\t0.25\n",
        encoding="utf-8",
    )

    datasets_json = metadata_dir / "datasets.json"
    datasets_json.write_text(
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
                    "data_path": "data/processed/001.tsv",
                },
                {
                    "id": "002",
                    "geo_accession": "GSE002",
                    "miRNA": "hsa-miR-2",
                    "miRNA_sequence": "UGCA",
                    "cell_line": "HeLa",
                    "tissue": "cervix",
                    "perturbation": "knockdown",
                    "data_path": "data/processed/002.tsv",
                },
                {
                    "id": "003",
                    "geo_accession": None,
                    "miRNA": "hsa-miR-1",
                    "miRNA_sequence": "AUGC",
                    "cell_line": None,
                    "tissue": "liver",
                    "perturbation": None,
                    "data_path": "data/processed/003.tsv",
                },
            ]
        ),
        encoding="utf-8",
    )

    return tmp_path, datasets_json


def test_star_import_only_exposes_public_dataset_api():
    namespace = {}
    exec("from funmirbench.datasets import *", {}, namespace)

    assert "DatasetMeta" in namespace
    assert "get_dataset" in namespace
    assert "load_dataset" in namespace
    assert "_load_raw_metadata" not in namespace
    assert "_matches" not in namespace


def test_get_dataset_supports_dataset_id_and_legacy_id(dataset_registry):
    root, datasets_json = dataset_registry

    meta = datasets.get_dataset(
        dataset_id="001",
        root=root,
        datasets_json=datasets_json,
    )

    assert meta is not None
    assert meta.id == "001"

    with pytest.deprecated_call(
        match=r"get_dataset\(id=\.\.\.\) is deprecated",
    ):
        legacy_meta = datasets.get_dataset(
            id="001",
            root=root,
            datasets_json=datasets_json,
        )

    assert legacy_meta == meta


def test_load_dataset_supports_dataset_id_and_legacy_id(dataset_registry):
    root, datasets_json = dataset_registry

    df = datasets.load_dataset(
        dataset_id="001",
        root=root,
        datasets_json=datasets_json,
    )
    assert df["gene_id"].tolist() == ["ENSG000001", "ENSG000002"]

    with pytest.deprecated_call(
        match=r"load_dataset\(id=\.\.\.\) is deprecated",
    ):
        legacy_df = datasets.load_dataset(
            id="001",
            root=root,
            datasets_json=datasets_json,
        )

    assert legacy_df.equals(df)


def test_list_and_summarize_helpers_return_sorted_values(dataset_registry):
    root, datasets_json = dataset_registry

    assert datasets.list_cell_lines(root=root, datasets_json=datasets_json) == [
        "HeLa"
    ]
    assert datasets.list_mirnas(root=root, datasets_json=datasets_json) == [
        "hsa-miR-1",
        "hsa-miR-2",
    ]
    assert datasets.list_tissues(root=root, datasets_json=datasets_json) == [
        "cervix",
        "liver",
    ]
    assert datasets.list_geo_accessions(root=root, datasets_json=datasets_json) == [
        "GSE001",
        "GSE002",
    ]
    assert datasets.list_perturbations(root=root, datasets_json=datasets_json) == [
        "knockdown",
        "overexpression",
    ]

    assert datasets.summarize_cell_lines(
        root=root,
        datasets_json=datasets_json,
    ) == {"HeLa": 2}
    assert datasets.summarize_mirnas(
        root=root,
        datasets_json=datasets_json,
    ) == {"hsa-miR-1": 2, "hsa-miR-2": 1}
    assert datasets.summarize_tissues(
        root=root,
        datasets_json=datasets_json,
    ) == {"cervix": 2, "liver": 1}
    assert datasets.summarize_perturbations(
        root=root,
        datasets_json=datasets_json,
    ) == {"knockdown": 1, "overexpression": 1}


def test_list_datasets_accepts_list_filters(dataset_registry):
    root, datasets_json = dataset_registry

    results = datasets.list_datasets(
        miRNA=["HSA-MIR-1", "hsa-miR-2"],
        cell_line=["hela", "missing"],
        root=root,
        datasets_json=datasets_json,
    )

    assert [meta.id for meta in results] == ["001", "002"]


def test_load_all_datasets_accepts_list_filters(dataset_registry):
    root, datasets_json = dataset_registry

    df = datasets.load_all_datasets(
        miRNA=["hsa-miR-1", "hsa-miR-2"],
        cell_line=["HeLa"],
        root=root,
        datasets_json=datasets_json,
    )

    assert sorted(df["dataset_id"].unique().tolist()) == ["001", "002"]
    assert sorted(df["miRNA"].unique().tolist()) == ["hsa-miR-1", "hsa-miR-2"]


def test_list_datasets_by_cell_line_is_deprecated(dataset_registry):
    root, datasets_json = dataset_registry

    with pytest.deprecated_call(
        match="list_datasets_by_cell_line is deprecated",
    ):
        results = datasets.list_datasets_by_cell_line(
            "HeLa",
            root=root,
            datasets_json=datasets_json,
        )

    assert [meta.id for meta in results] == ["001", "002"]
