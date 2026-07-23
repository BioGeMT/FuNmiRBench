#!/usr/bin/env python3
"""Download UCSC hg38 phastCons100way BigWig for conservation plots."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import requests

from funmirbench.logger import setup_logging


DEFAULT_URL = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/"
    "phastCons100way/hg38.phastCons100way.bw"
)
DEFAULT_MD5_URL = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/"
    "phastCons100way/md5sum.txt"
)
DEFAULT_OUT = Path("data/resources/conservation/hg38.phastCons100way.bw")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download UCSC hg38.phastCons100way.bw."
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--md5-url", default=DEFAULT_MD5_URL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-md5", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def expected_md5(url: str, filename: str) -> str | None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    for line in response.text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].lstrip("./") == filename:
            return parts[0]
    return None


def file_md5(path: Path, chunk_size: int) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, out: Path, chunk_size: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    headers = {}
    existing = tmp.stat().st_size if tmp.exists() else 0
    if existing:
        headers["Range"] = f"bytes={existing}-"

    with requests.get(url, stream=True, headers=headers, timeout=120) as response:
        response.raise_for_status()
        if existing and response.status_code != 206:
            logger.info(
                "Server did not honor Range request; restarting download from byte 0."
            )
            existing = 0
        mode = "ab" if existing else "wb"
        total = response.headers.get("Content-Length")
        total_size = int(total) + existing if total is not None else None
        written = existing
        with tmp.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
    if total_size:
        logger.info("Downloaded %.2f/%.2f GB", written / 1e9, total_size / 1e9)
    else:
        logger.info("Downloaded %.2f GB", written / 1e9)
    tmp.replace(out)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    if args.out.exists() and not args.force:
        logger.info("Already exists: %s", args.out)
    else:
        download(args.url, args.out, args.chunk_size)

    if not args.skip_md5:
        expected = expected_md5(args.md5_url, Path(args.url).name)
        if expected is None:
            raise ValueError(f"Could not find md5 for {Path(args.url).name}")
        observed = file_md5(args.out, args.chunk_size)
        if observed != expected:
            raise ValueError(
                f"MD5 mismatch for {args.out}: expected {expected}, observed {observed}"
            )
        logger.info("MD5 ok: %s", observed)

    logger.info("Conservation BigWig: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
