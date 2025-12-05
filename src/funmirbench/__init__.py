"""
FuNmiRBench: Benchmarking package for functional miRNA target prediction.
"""

from .datasets import (
    DatasetMeta,
    load_metadata,
    list_datasets,
    load_dataset,
    load_all_datasets,
)

__all__ = [
    "DatasetMeta",
    "load_metadata",
    "list_datasets",
    "load_dataset",
    "load_all_datasets",
]
