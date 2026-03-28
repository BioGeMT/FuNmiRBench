from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetMeta:
    id: str
    miRNA: str
    cell_line: str
    tissue: str
    perturbation: str
    organism: str
    geo_accession: str
    data_path: str
    root: Path

    @property
    def full_path(self):
        return self.root / self.data_path
