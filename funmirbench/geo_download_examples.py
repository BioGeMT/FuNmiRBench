"""Download real GEO example inputs for the ingestion pipeline."""

from __future__ import annotations

import argparse
import pathlib

import requests


EXAMPLES = {
    "gse253003-counts": {
        "description": "Real count matrix for GSE253003 miR-323a-3p vs miRCTRL.",
        "targets": [
            {
                "url": "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE253003&file=GSE253003_Count.csv.gz&format=file",
                "dest": pathlib.Path("data/experiments/raw/GSE253003/GSE253003_Count.csv.gz"),
            }
        ],
    },
    "gse93717-reads": {
        "description": "Real FASTQ files for GSE93717 miR-941 overexpression.",
        "targets": [
            {
                "url": "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR518/005/SRR5181705/SRR5181705.fastq.gz",
                "dest": pathlib.Path("data/experiments/raw/GSE93717/SRR5181705.fastq.gz"),
            },
            {
                "url": "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR518/006/SRR5181706/SRR5181706.fastq.gz",
                "dest": pathlib.Path("data/experiments/raw/GSE93717/SRR5181706.fastq.gz"),
            },
            {
                "url": "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR518/007/SRR5181707/SRR5181707.fastq.gz",
                "dest": pathlib.Path("data/experiments/raw/GSE93717/SRR5181707.fastq.gz"),
            },
            {
                "url": "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR518/008/SRR5181708/SRR5181708.fastq.gz",
                "dest": pathlib.Path("data/experiments/raw/GSE93717/SRR5181708.fastq.gz"),
            },
            {
                "url": "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR518/009/SRR5181709/SRR5181709.fastq.gz",
                "dest": pathlib.Path("data/experiments/raw/GSE93717/SRR5181709.fastq.gz"),
            },
            {
                "url": "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR518/000/SRR5181710/SRR5181710.fastq.gz",
                "dest": pathlib.Path("data/experiments/raw/GSE93717/SRR5181710.fastq.gz"),
            },
        ],
    },
}

DEFAULT_SELECTION = ["gse253003-counts", "gse93717-reads"]


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def download_file(url: str, dest: pathlib.Path, *, force: bool) -> None:
    if dest.exists() and not force:
        print(f"skip {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {url}")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with dest.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    print(f"saved {dest}")


def resolve_selection(selection: list[str]) -> list[str]:
    if not selection or "all" in selection:
        return DEFAULT_SELECTION
    invalid = [name for name in selection if name not in EXAMPLES]
    if invalid:
        raise ValueError(f"Unknown example selections: {invalid}")
    return selection


def download_examples(selection: list[str], *, repo: pathlib.Path | None = None, force: bool = False) -> None:
    repo = (repo or repo_root()).resolve()
    for name in resolve_selection(selection):
        print(f"== {name} ==")
        for target in EXAMPLES[name]["targets"]:
            download_file(target["url"], repo / target["dest"], force=force)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download real example GEO inputs for the ingestion pipeline.")
    parser.add_argument(
        "--example",
        action="append",
        dest="examples",
        choices=["all", *EXAMPLES.keys()],
        help="Example input to download. Repeat for multiple selections. Defaults to all.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_examples(args.examples or ["all"], force=args.force)


if __name__ == "__main__":
    main()
