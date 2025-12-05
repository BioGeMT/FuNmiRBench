"""
Dataset utilities for FuNmiRBench.

This module:

- Reads `metadata/datasets.json`
- Lets you list/filter datasets by miRNA, cell line, perturbation, etc.
- Loads one or many TSV files into pandas DataFrames.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


# Project root (FuNmiRBench/)
ROOT = pathlib.Path(__file__).resolve().parents[2]
DATASETS_JSON = ROOT / "metadata" / "datasets.json"


@dataclass
class DatasetMeta:
    """Metadata for a single experiment/dataset."""

    id: str
    geo_accession: Optional[str]
    miRNA: str
    miRNA_sequence: str
    cell_line: Optional[str]
    tissue: Optional[str]
    perturbation: Optional[str]  # "overexpression" / "knockdown"
    organism: str
    method: Optional[str]
    treatment: Optional[str]
    pubmed_id: Optional[str]
    gse_url: Optional[str]
    data_path: str

    @property
    def full_path(self) -> pathlib.Path:
        """Absolute path to the TSV file on disk."""
        return ROOT / self.data_path


def _load_raw_metadata() -> List[Dict[str, Any]]:
    """Load raw JSON objects from metadata/datasets.json."""
    if not DATASETS_JSON.exists():
        raise FileNotFoundError(f"Metadata file not found: {DATASETS_JSON}")

    with DATASETS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {DATASETS_JSON}, got {type(data)}")

    return data


def load_metadata() -> List[DatasetMeta]:
    """
    Return a list of DatasetMeta objects for all experiments.

    This is the main entry point to inspect available datasets.
    """
    raw = _load_raw_metadata()
    metas: List[DatasetMeta] = []

    for entry in raw:
        metas.append(
            DatasetMeta(
                id=entry["id"],
                geo_accession=entry.get("geo_accession"),
                miRNA=entry["miRNA"],
                miRNA_sequence=entry["miRNA_sequence"],
                cell_line=entry.get("cell_line"),
                tissue=entry.get("tissue"),
                perturbation=entry.get("perturbation"),
                organism=entry.get("organism", "Homo sapiens"),
                method=entry.get("method"),
                treatment=entry.get("treatment"),
                pubmed_id=entry.get("pubmed_id"),
                gse_url=entry.get("gse_url"),
                data_path=entry["data_path"],
            )
        )

    return metas


def _matches(field_value: Optional[str], target: Optional[str]) -> bool:
    """Case-insensitive exact match helper."""
    if target is None:
        return True
    if field_value is None:
        return False
    return field_value.lower() == target.lower()


def list_datasets(
    *,
    miRNA: Optional[str] = None,
    cell_line: Optional[str] = None,
    perturbation: Optional[str] = None,
    tissue: Optional[str] = None,
    geo_accession: Optional[str] = None,
) -> List[DatasetMeta]:
    """
    Return a list of DatasetMeta filtered by the given criteria.

    Examples
    --------
    - list_datasets(perturbation="overexpression")
    - list_datasets(miRNA="hsa-miR-124-3p", cell_line="HeLa")
    """
    metas = load_metadata()
    results: List[DatasetMeta] = []

    for m in metas:
        if not _matches(m.miRNA, miRNA):
            continue
        if not _matches(m.cell_line, cell_line):
            continue
        if not _matches(m.perturbation, perturbation):
            continue
        if not _matches(m.tissue, tissue):
            continue
        if not _matches(m.geo_accession, geo_accession):
            continue
        results.append(m)

    return results


def get_dataset(id: str) -> Optional[DatasetMeta]:
    """Return the DatasetMeta with the given ID, or None if not found."""
    for m in load_metadata():
        if m.id == id:
            return m
    return None


def load_dataset(id: str, *, sep: str = "\t"):
    """
    Load a single dataset's TSV as a pandas DataFrame.

    Parameters
    ----------
    id : str
        The dataset ID (e.g. "001", "017", ...).
    sep : str
        Column separator (default: tab).

    Returns
    -------
    pandas.DataFrame
    """
    meta = get_dataset(id)
    if meta is None:
        raise ValueError(f"No dataset with id {id!r}")

    try:
        import pandas as pd  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pandas is required to use load_dataset; "
            "install it with `pip install pandas`."
        ) from exc

    path = meta.full_path
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path}")

    return pd.read_csv(path, sep=sep)


def load_all_datasets(
    *,
    miRNA: Optional[str] = None,
    cell_line: Optional[str] = None,
    perturbation: Optional[str] = None,
    tissue: Optional[str] = None,
    geo_accession: Optional[str] = None,
    sep: str = "\t",
):
    """
    Load and concatenate multiple datasets into a single DataFrame.

    The same filters as list_datasets() are applied. The resulting DataFrame
    will include 'dataset_id' and 'miRNA' columns to let you track origin.
    """
    try:
        import pandas as pd  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pandas is required to use load_all_datasets; "
            "install it with `pip install pandas`."
        ) from exc

    metas = list_datasets(
        miRNA=miRNA,
        cell_line=cell_line,
        perturbation=perturbation,
        tissue=tissue,
        geo_accession=geo_accession,
    )

    frames = []
    for m in metas:
        path = m.full_path
        if not path.exists():
            raise FileNotFoundError(f"Data file not found at {path}")
        df = pd.read_csv(path, sep=sep)
        df["dataset_id"] = m.id
        df["miRNA"] = m.miRNA
        frames.append(df)

    if not frames:
        # Return empty DataFrame with no columns
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Convenience listing / summarizing helpers
# ---------------------------------------------------------------------------

def list_cell_lines() -> List[str]:
    """Return a sorted list of all unique cell lines."""
    metas = load_metadata()
    cell_lines = {m.cell_line for m in metas if m.cell_line is not None}
    return sorted(cell_lines)


def list_mirnas() -> List[str]:
    """Return a sorted list of all unique miRNAs."""
    metas = load_metadata()
    mirnas = {m.miRNA for m in metas if m.miRNA is not None}
    return sorted(mirnas)


def list_tissues() -> List[str]:
    """Return a sorted list of all unique tissues."""
    metas = load_metadata()
    tissues = {m.tissue for m in metas if m.tissue is not None}
    return sorted(tissues)


def list_geo_accessions() -> List[str]:
    """Return a sorted list of all unique GEO accessions."""
    metas = load_metadata()
    geos = {m.geo_accession for m in metas if m.geo_accession is not None}
    return sorted(geos)


def list_perturbations() -> List[str]:
    """Return a sorted list of all unique perturbation types."""
    metas = load_metadata()
    kinds = {m.perturbation for m in metas if m.perturbation is not None}
    return sorted(kinds)


def summarize_cell_lines() -> Dict[str, int]:
    """Return a dict: cell_line -> number of datasets."""
    metas = load_metadata()
    counts: Dict[str, int] = {}
    for m in metas:
        if m.cell_line is None:
            continue
        counts[m.cell_line] = counts.get(m.cell_line, 0) + 1
    return dict(sorted(counts.items()))


def summarize_mirnas() -> Dict[str, int]:
    """Return a dict: miRNA -> number of datasets."""
    metas = load_metadata()
    counts: Dict[str, int] = {}
    for m in metas:
        counts[m.miRNA] = counts.get(m.miRNA, 0) + 1
    return dict(sorted(counts.items()))


def summarize_tissues() -> Dict[str, int]:
    """Return a dict: tissue -> number of datasets."""
    metas = load_metadata()
    counts: Dict[str, int] = {}
    for m in metas:
        if m.tissue is None:
            continue
        counts[m.tissue] = counts.get(m.tissue, 0) + 1
    return dict(sorted(counts.items()))


def summarize_perturbations() -> Dict[str, int]:
    """Return a dict: perturbation -> number of datasets."""
    metas = load_metadata()
    counts: Dict[str, int] = {}
    for m in metas:
        if m.perturbation is None:
            continue
        counts[m.perturbation] = counts.get(m.perturbation, 0) + 1
    return dict(sorted(counts.items()))


def group_by_mirna() -> Dict[str, List[DatasetMeta]]:
    """Return a dict: miRNA -> list of DatasetMeta."""
    metas = load_metadata()
    grouped: Dict[str, List[DatasetMeta]] = {}
    for m in metas:
        grouped.setdefault(m.miRNA, []).append(m)
    return grouped


def group_by_geo() -> Dict[str, List[DatasetMeta]]:
    """Return a dict: GEO accession -> list of DatasetMeta."""
    metas = load_metadata()
    grouped: Dict[str, List[DatasetMeta]] = {}
    for m in metas:
        if m.geo_accession is None:
            continue
        grouped.setdefault(m.geo_accession, []).append(m)
    return grouped


def list_ids() -> List[str]:
    """Return a sorted list of all dataset IDs."""
    metas = load_metadata()
    return sorted(m.id for m in metas)


def list_datasets_by_cell_line(cell_line: str) -> List[DatasetMeta]:
    """Return all datasets associated with a specific cell line."""
    return list_datasets(cell_line=cell_line)

def summarize_datasets() -> List[Dict[str, Any]]:
    """
    Return a list of lightweight summaries, one per dataset.

    Each summary is a dict with keys:
    - id
    - miRNA
    - perturbation
    - cell_line
    - tissue
    - geo_accession
    - pubmed_id
    """
    metas = load_metadata()
    summaries: List[Dict[str, Any]] = []

    for m in metas:
        summaries.append(
            {
                "id": m.id,
                "miRNA": m.miRNA,
                "perturbation": m.perturbation,
                "cell_line": m.cell_line,
                "tissue": m.tissue,
                "geo_accession": m.geo_accession,
                "pubmed_id": m.pubmed_id,
            }
        )

    return summaries
