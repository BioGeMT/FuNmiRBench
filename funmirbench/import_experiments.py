"""Download FuNmiRBench experiment DE tables from Zenodo."""

import argparse
import io
import os
import pathlib
import re
import shutil
import zipfile
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests

DEFAULT_RECORD_URL = "https://zenodo.org/records/17585186"


def _record_id_from_url(record_url):
    match = re.search(r"/records?/(\d+)", record_url)
    return match.group(1)


def _with_token(url, token):
    if not token:
        return url
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    query["token"] = [token]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def _download_bytes(url):
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    return response.content


def _extract_zip_into(zip_bytes, out_dir, *, overwrite):
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = pathlib.Path(info.filename).name
            if not name.endswith(".tsv"):
                continue
            out_path = out_dir / name
            if out_path.exists() and not overwrite:
                continue
            with archive.open(info) as src, out_path.open("wb") as dst:
                dst.write(src.read())
            written.append(out_path)
    return written


def _import_from_dir(from_dir, out_dir, *, overwrite):
    paths = sorted(p for p in from_dir.iterdir() if p.is_file() and p.suffix.lower() == ".tsv")
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for src in paths:
        dst = out_dir / src.name
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    return copied, skipped


def main():
    parser = argparse.ArgumentParser(description="Download experiment DE tables from Zenodo.")
    parser.add_argument("--from-dir", type=pathlib.Path, default=None)
    parser.add_argument("--record-url", default=DEFAULT_RECORD_URL)
    parser.add_argument("--token", default=os.getenv("ZENODO_TOKEN"))
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("data/experiments/processed"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()

    if args.from_dir is not None:
        copied, skipped = _import_from_dir(
            args.from_dir.resolve(),
            out_dir,
            overwrite=args.overwrite,
        )
        print(f"Imported {copied} TSV files (skipped {skipped} existing)")
        return

    record_id = _record_id_from_url(args.record_url)
    archive_url = _with_token(
        f"https://zenodo.org/api/records/{record_id}/files-archive",
        args.token,
    )
    archive = _download_bytes(archive_url)
    written = _extract_zip_into(archive, out_dir, overwrite=args.overwrite)
    print(f"Downloaded {len(written)} TSV files to {out_dir}")


if __name__ == "__main__":
    main()
