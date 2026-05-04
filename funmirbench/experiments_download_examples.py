"""Download real GEO example inputs for the ingestion pipeline."""

from __future__ import annotations

import argparse
import logging
import pathlib
import requests
import gzip
import os
import shutil
import subprocess

from funmirbench.logger import parse_log_level, setup_logging


EXAMPLES = {
    "mirbase-hsa-mature": {
        "description": "miRBase mature human miRNA names for config validation.",
        "targets": [
            {
                "url": "https://mirbase.org/download_version_files/22.1/mature.fa",
                "dest": pathlib.Path("data/experiments/raw/refs/mirbase/mature.fa"),
            },
        ],
    },
    "ensembl-v115-refs": {
        "description": "Shared Homo sapiens Ensembl v115 genome FASTA and GTF for reads examples.",
        "targets": [
            {
                "url": "https://ftp.ensembl.org/pub/release-115/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa.gz",
                "dest": pathlib.Path(
                    "data/experiments/raw/refs/ensembl_v115/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
                ),
            },
            {
                "url": "https://ftp.ensembl.org/pub/release-115/gtf/homo_sapiens/Homo_sapiens.GRCh38.115.gtf.gz",
                "dest": pathlib.Path(
                    "data/experiments/raw/refs/ensembl_v115/Homo_sapiens.GRCh38.115.gtf.gz"
                ),
            },
        ],
    },
    "gse253003-counts": {
        "description": "Real count matrix for GSE253003 miR-323a-3p vs miRCTRL.",
        "targets": [
            {
                "url": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE253003&file=GSE253003_Count.csv.gz&format=file",
                "dest": pathlib.Path("data/experiments/raw/GSE253003/GSE253003_Count.csv.gz"),
            }
        ],
    },
    "gse129146-reads": {
        "description": "Real FASTQ files for GSE129146 miR-455-3p overexpression.",
        "targets": [
            {
                "url": "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR883/000/SRR8832550/SRR8832550.fastq.gz",
                "dest": pathlib.Path("data/experiments/raw/GSE129146/SRR8832550.fastq.gz"),
            },
            {
                "url": "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR883/001/SRR8832551/SRR8832551.fastq.gz",
                "dest": pathlib.Path("data/experiments/raw/GSE129146/SRR8832551.fastq.gz"),
            },
            {
                "url": "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR883/002/SRR8832552/SRR8832552.fastq.gz",
                "dest": pathlib.Path("data/experiments/raw/GSE129146/SRR8832552.fastq.gz"),
            },
            {
                "url": "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR883/003/SRR8832553/SRR8832553.fastq.gz",
                "dest": pathlib.Path("data/experiments/raw/GSE129146/SRR8832553.fastq.gz"),
            },
            {
                "url": "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR883/004/SRR8832554/SRR8832554.fastq.gz",
                "dest": pathlib.Path("data/experiments/raw/GSE129146/SRR8832554.fastq.gz"),
            },
            {
                "url": "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR883/005/SRR8832555/SRR8832555.fastq.gz",
                "dest": pathlib.Path("data/experiments/raw/GSE129146/SRR8832555.fastq.gz"),
            },
        ],
    },
}

DEFAULT_SELECTION = ["mirbase-hsa-mature", "ensembl-v115-refs", "gse253003-counts", "gse129146-reads"]

logger = logging.getLogger(__name__)


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def download_file(url: str, dest: pathlib.Path, *, force: bool) -> None:
    if dest.exists() and not force:
        logger.info("skip %s", dest)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("download %s", url)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with dest.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    logger.info("saved %s", dest)


def build_hsa_mature_mirna_list(repo: pathlib.Path) -> None:
    mature_fa = repo / "data/experiments/raw/refs/mirbase/mature.fa"
    out_path = repo / "data/experiments/raw/refs/mirbase/hsa_mature_mirnas.txt"

    if not mature_fa.exists():
        raise ValueError(f"miRBase mature FASTA does not exist: {mature_fa}")

    names = []
    with mature_fa.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(">"):
                continue

            name = line[1:].split()[0].strip()

            if name.startswith("hsa-"):
                names.append(name)

    if not names:
        raise ValueError(f"No hsa mature miRNAs found in {mature_fa}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sorted(set(names))) + "\n", encoding="utf-8")
    logger.info("saved %s (%s hsa mature miRNAs)", out_path, len(set(names)))

def resolve_selection(selection: list[str]) -> list[str]:
    if not selection or "all" in selection:
        return DEFAULT_SELECTION
    invalid = [name for name in selection if name not in EXAMPLES]
    if invalid:
        raise ValueError(f"Unknown example selections: {invalid}")
    return selection


def download_examples(selection: list[str], *, repo: pathlib.Path | None = None, force: bool = False) -> None:
    repo = (repo or repo_root()).resolve()
    selected = resolve_selection(selection)

    for name in selected:
        logger.info("== %s ==", name)
        for target in EXAMPLES[name]["targets"]:
            download_file(target["url"], repo / target["dest"], force=force)

    if "mirbase-hsa-mature" in selected:
        build_hsa_mature_mirna_list(repo)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download example experiment inputs and shared reference resources.")
    parser.add_argument(
        "--example",
        action="append",
        dest="examples",
        choices=["all", *EXAMPLES.keys()],
        help="Example input to download. Repeat for multiple selections. Defaults to all.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(parse_log_level(args.log_level))
    download_examples(args.examples or ["all"], force=args.force)


if __name__ == "__main__":
    main()
