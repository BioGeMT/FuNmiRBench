"""
Dataset utilities for FuNmiRBench.

This module:

- Reads `metadata/datasets.json`
- Lets you list/filter datasets by miRNA, cell line, perturbation, etc.
- Loads one or many TSV files into pandas DataFrames.
"""

from __future__ import annotations

import json
import os
import pathlib
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, cast

from funmirbench.utils.paths import project_root


DEFAULT_ROOT = project_root()
DEFAULT_DATASETS_JSON = DEFAULT_ROOT / "metadata" / "datasets.json"
_DATASET_ID_MISSING = object()

__all__ = [
    "DatasetMeta",
    "get_root",
    "get_datasets_json",
    "load_metadata",
    "list_datasets",
    "get_dataset",
    "load_dataset",
    "load_all_datasets",
    "list_cell_lines",
    "list_mirnas",
    "list_tissues",
    "list_geo_accessions",
    "list_perturbations",
    "list_ids",
    "summarize_cell_lines",
    "summarize_mirnas",
    "summarize_tissues",
    "summarize_perturbations",
    "summarize_datasets",
    "group_by_mirna",
    "group_by_geo",
    "list_datasets_by_cell_line",
]


def get_root(root: Optional[pathlib.Path] = None) -> pathlib.Path:
    """Resolve project root, allowing override via arg or FUNMIRBENCH_ROOT."""
    return project_root(root)


def get_datasets_json(
    *,
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
) -> pathlib.Path:
    """Resolve datasets.json path, allowing override via arg or FUNMIRBENCH_DATASETS_JSON."""
    if datasets_json is not None:
        return datasets_json
    env_path = os.getenv("FUNMIRBENCH_DATASETS_JSON")
    if env_path:
        return pathlib.Path(env_path).expanduser().resolve()
    return get_root(root) / "metadata" / "datasets.json"


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
    root: pathlib.Path = DEFAULT_ROOT  # resolved root for this dataset

    @property
    def full_path(self) -> pathlib.Path:
        """Absolute path to the TSV file on disk."""
        return self.root / self.data_path


FieldExtractor = Callable[[DatasetMeta], Optional[str]]
StringListFilter = Optional[str | List[str]]


def _resolve_dataset_id_arg(
    func_name: str,
    dataset_id: object,
    kwargs: Dict[str, Any],
) -> str:
    legacy_id = kwargs.pop("id", _DATASET_ID_MISSING)

    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(
            f"{func_name}() got an unexpected keyword argument {unexpected!r}"
        )

    if dataset_id is _DATASET_ID_MISSING:
        if legacy_id is _DATASET_ID_MISSING:
            raise TypeError(
                f"{func_name}() missing 1 required positional argument: "
                "'dataset_id'"
            )
        warnings.warn(
            f"{func_name}(id=...) is deprecated; use dataset_id=... instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return cast(str, legacy_id)

    if legacy_id is not _DATASET_ID_MISSING:
        raise TypeError(
            f"{func_name}() got multiple values for argument 'dataset_id'"
        )

    return cast(str, dataset_id)


def _load_raw_metadata(
    *,
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
) -> List[Dict[str, Any]]:
    """Load raw JSON objects from metadata/datasets.json."""
    path = get_datasets_json(root=root, datasets_json=datasets_json)
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)}")

    return data


def load_metadata(
    *,
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
) -> List[DatasetMeta]:
    """
    Return a list of DatasetMeta objects for all experiments.

    This is the main entry point to inspect available datasets.
    """
    resolved_root = get_root(root)
    raw = _load_raw_metadata(root=resolved_root, datasets_json=datasets_json)
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
                root=resolved_root,
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


def _normalize_text_filter(target: StringListFilter) -> Optional[set[str]]:
    """Normalize a string-or-list filter into lowercase match values."""
    if target is None:
        return None

    values = [target] if isinstance(target, str) else target
    normalized = {value.lower() for value in values if value}
    return normalized or None


def list_datasets(
    *,
    miRNA: StringListFilter = None,
    cell_line: StringListFilter = None,
    perturbation: Optional[str] = None,
    tissue: Optional[str] = None,
    geo_accession: Optional[str] = None,
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
) -> List[DatasetMeta]:
    """
    Return a list of DatasetMeta filtered by the given criteria.

    Examples
    --------
    - list_datasets(perturbation="overexpression")
    - list_datasets(miRNA="hsa-miR-124-3p", cell_line="HeLa")
    - list_datasets(miRNA=["hsa-miR-124-3p", "hsa-miR-1"], cell_line=["HeLa", "HEK293"])
    """
    metas = load_metadata(root=root, datasets_json=datasets_json)
    results: List[DatasetMeta] = []
    miRNA_filters = _normalize_text_filter(miRNA)
    cell_line_filters = _normalize_text_filter(cell_line)

    for m in metas:
        if miRNA_filters is not None:
            if m.miRNA is None or m.miRNA.lower() not in miRNA_filters:
                continue
        if cell_line_filters is not None:
            if m.cell_line is None or m.cell_line.lower() not in cell_line_filters:
                continue
        if not _matches(m.perturbation, perturbation):
            continue
        if not _matches(m.tissue, tissue):
            continue
        if not _matches(m.geo_accession, geo_accession):
            continue
        results.append(m)

    return results


def get_dataset(
    dataset_id: object = _DATASET_ID_MISSING,
    *,
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
    **kwargs: Any,
) -> Optional[DatasetMeta]:
    """Return the DatasetMeta with the given ID, or None if not found."""
    resolved_dataset_id = _resolve_dataset_id_arg(
        "get_dataset", dataset_id, kwargs
    )
    for m in load_metadata(root=root, datasets_json=datasets_json):
        if m.id == resolved_dataset_id:
            return m
    return None


def load_dataset(
    dataset_id: object = _DATASET_ID_MISSING,
    *,
    sep: str = "\t",
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
    **kwargs: Any,
):
    """
    Load a single dataset's TSV as a pandas DataFrame.

    Parameters
    ----------
    dataset_id : str
        The dataset ID (e.g. "001", "017", ...).
    sep : str
        Column separator (default: tab).

    Returns
    -------
    pandas.DataFrame
    """
    resolved_dataset_id = _resolve_dataset_id_arg(
        "load_dataset", dataset_id, kwargs
    )
    meta = get_dataset(
        resolved_dataset_id,
        root=root,
        datasets_json=datasets_json,
    )
    if meta is None:
        raise ValueError(f"No dataset with id {resolved_dataset_id!r}")

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
    miRNA: StringListFilter = None,
    cell_line: StringListFilter = None,
    perturbation: Optional[str] = None,
    tissue: Optional[str] = None,
    geo_accession: Optional[str] = None,
    sep: str = "\t",
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
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
        root=root,
        datasets_json=datasets_json
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

def _list_unique_field(
    extractor: FieldExtractor,
    *,
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
) -> List[str]:
    metas = load_metadata(root=root, datasets_json=datasets_json)
    values = {extractor(m) for m in metas}
    return sorted(v for v in values if v is not None)


def _summarize_field(
    extractor: FieldExtractor,
    *,
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
) -> Dict[str, int]:
    metas = load_metadata(root=root, datasets_json=datasets_json)
    counts = Counter(extractor(m) for m in metas)
    counts.pop(None, None)
    return dict(sorted(counts.items()))


def list_cell_lines(
    *,
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
) -> List[str]:
    """Return a sorted list of all unique cell lines."""
    return _list_unique_field(
        lambda m: m.cell_line,
        root=root,
        datasets_json=datasets_json,
    )


def list_mirnas(
    *,
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
) -> List[str]:
    """Return a sorted list of all unique miRNAs."""
    return _list_unique_field(
        lambda m: m.miRNA,
        root=root,
        datasets_json=datasets_json,
    )


def list_tissues(
    *,
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
) -> List[str]:
    """Return a sorted list of all unique tissues."""
    return _list_unique_field(
        lambda m: m.tissue,
        root=root,
        datasets_json=datasets_json,
    )


def list_geo_accessions(
    *,
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
) -> List[str]:
    """Return a sorted list of all unique GEO accessions."""
    return _list_unique_field(
        lambda m: m.geo_accession,
        root=root,
        datasets_json=datasets_json,
    )


def list_perturbations(
    *,
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
) -> List[str]:
    """Return a sorted list of all unique perturbation types."""
    return _list_unique_field(
        lambda m: m.perturbation,
        root=root,
        datasets_json=datasets_json,
    )


def summarize_cell_lines(
    *,
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
) -> Dict[str, int]:
    """Return a dict: cell_line -> number of datasets."""
    return _summarize_field(
        lambda m: m.cell_line,
        root=root,
        datasets_json=datasets_json,
    )


def summarize_mirnas(
    *,
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
) -> Dict[str, int]:
    """Return a dict: miRNA -> number of datasets."""
    return _summarize_field(
        lambda m: m.miRNA,
        root=root,
        datasets_json=datasets_json,
    )


def summarize_tissues(
    *,
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
) -> Dict[str, int]:
    """Return a dict: tissue -> number of datasets."""
    return _summarize_field(
        lambda m: m.tissue,
        root=root,
        datasets_json=datasets_json,
    )


def summarize_perturbations(
    *,
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
) -> Dict[str, int]:
    """Return a dict: perturbation -> number of datasets."""
    return _summarize_field(
        lambda m: m.perturbation,
        root=root,
        datasets_json=datasets_json,
    )


def group_by_mirna(*, root: Optional[pathlib.Path] = None, datasets_json: Optional[pathlib.Path] = None) -> Dict[str, List[DatasetMeta]]:
    """Return a dict: miRNA -> list of DatasetMeta."""
    metas = load_metadata(root=root, datasets_json=datasets_json)
    grouped: Dict[str, List[DatasetMeta]] = {}
    for m in metas:
        grouped.setdefault(m.miRNA, []).append(m)
    return grouped


def group_by_geo(*, root: Optional[pathlib.Path] = None, datasets_json: Optional[pathlib.Path] = None) -> Dict[str, List[DatasetMeta]]:
    """Return a dict: GEO accession -> list of DatasetMeta."""
    metas = load_metadata(root=root, datasets_json=datasets_json)
    grouped: Dict[str, List[DatasetMeta]] = {}
    for m in metas:
        if m.geo_accession is None:
            continue
        grouped.setdefault(m.geo_accession, []).append(m)
    return grouped


def list_ids(
    *,
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
) -> List[str]:
    # return a short list of dataset ids.
    metas = load_metadata(root=root, datasets_json=datasets_json)
    return sorted(m.id for m in metas)


def list_datasets_by_cell_line(
    cell_line: str,
    *,
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
) -> List[DatasetMeta]:
    """Deprecated wrapper for list_datasets(cell_line=...)."""
    warnings.warn(
        "list_datasets_by_cell_line is deprecated; "
        "use list_datasets(cell_line=...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return list_datasets(cell_line=cell_line, root=root, datasets_json=datasets_json)


def summarize_datasets(
    *,
    root: Optional[pathlib.Path] = None,
    datasets_json: Optional[pathlib.Path] = None,
) -> List[Dict[str, Any]]:
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
    metas = load_metadata(root=root, datasets_json=datasets_json)
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
