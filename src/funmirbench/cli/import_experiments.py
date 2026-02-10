"""
Download the FuNmiRBench experiment corpus from Zenodo into the local data/ directory.

Supports restricted Zenodo records via access token (link-request token).
"""

from __future__ import annotations

import argparse
import io
import os
import pathlib
import re
import sys
import zipfile
from typing import Optional, List, Tuple
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

try:
    import requests  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This command requires 'requests'. Install it with `pip install requests`."
    ) from exc


DEFAULT_RECORD_URL = "https://zenodo.org/records/17585186"
DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = pathlib.Path("data/experiments/processed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download experiment DE tables from a Zenodo record into data/experiments/processed/."
    )
    p.add_argument(
        "--record-url",
        default=DEFAULT_RECORD_URL,
        help="Zenodo record URL (default: FuNmiRBench corpus record).",
    )
    p.add_argument(
        "--token",
        default=os.getenv("ZENODO_TOKEN"),
        help="Zenodo access token (or set env var ZENODO_TOKEN).",
    )
    p.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for TSV files (repo-relative by default).",
    )
    p.add_argument(
        "--root",
        type=pathlib.Path,
        default=DEFAULT_ROOT,
        help="Repo root (used to resolve relative out-dir).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    return p.parse_args()


def _record_id_from_url(record_url: str) -> str:
    # supports /records/<id> or /record/<id>
    m = re.search(r"/records?/(\d+)", record_url)
    if not m:
        raise ValueError(f"Could not parse record id from URL: {record_url}")
    return m.group(1)


def _with_token(url: str, token: Optional[str]) -> str:
    if not token:
        return url
    u = urlparse(url)
    qs = parse_qs(u.query)
    qs["token"] = [token]
    return urlunparse(u._replace(query=urlencode(qs, doseq=True)))


def _download_bytes(url: str) -> bytes:
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    return r.content


def _try_download_archive(record_id: str, token: Optional[str]) -> Optional[bytes]:
    # Zenodo "Download all" is typically the files-archive endpoint.
    # For restricted records, token is required.
    url = f"https://zenodo.org/api/records/{record_id}/files-archive"
    url = _with_token(url, token)
    try:
        return _download_bytes(url)
    except Exception:
        return None


def _extract_zip_into(zip_bytes: bytes, out_dir: pathlib.Path, *, overwrite: bool) -> List[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[pathlib.Path] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = pathlib.Path(info.filename).name  # flatten
            if not name.endswith(".tsv"):
                continue
            out_path = out_dir / name
            if out_path.exists() and not overwrite:
                continue
            with zf.open(info) as src, out_path.open("wb") as dst:
                dst.write(src.read())
            written.append(out_path)
    return written


def _parse_file_links_from_html(html: str) -> List[str]:
    """
    Fallback: parse per-file download links from the record HTML.
    This is intentionally simple and avoids heavy dependencies.
    """
    # Typical pattern: href="/records/<id>/files/<filename>?download=1"
    links = re.findall(r'href="(/records/\d+/files/[^"]+?download=1[^"]*)"', html)
    # De-dup while preserving order
    seen = set()
    out = []
    for l in links:
        if l in seen:
            continue
        seen.add(l)
        out.append("https://zenodo.org" + l)
    return out


def _download_files_from_page(record_url: str, token: Optional[str]) -> List[Tuple[str, bytes]]:
    page_url = _with_token(record_url, token)
    html = _download_bytes(page_url).decode("utf-8", errors="replace")
    links = _parse_file_links_from_html(html)
    if not links:
        raise RuntimeError("Could not find any file download links in the record page.")
    files: List[Tuple[str, bytes]] = []
    for url in links:
        url = _with_token(url, token)
        data = _download_bytes(url)
        name = pathlib.Path(urlparse(url).path).name
        files.append((name, data))
    return files


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    record_id = _record_id_from_url(args.record_url)

    if not args.token:
        print("NOTE: no token provided. If the record is restricted, downloads will fail.", file=sys.stderr)

    # Try archive first (fast + simple)
    archive = _try_download_archive(record_id, args.token)
    if archive:
        written = _extract_zip_into(archive, out_dir, overwrite=args.overwrite)
        print(f"Downloaded archive and wrote {len(written)} TSV files to {out_dir}")
        return

    # Fallback: parse individual file links
    files = _download_files_from_page(args.record_url, args.token)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, data in files:
        if not name.endswith(".tsv"):
            continue
        path = out_dir / name
        if path.exists() and not args.overwrite:
            continue
        path.write_bytes(data)
        written += 1

    logger.info("Downloaded %d TSV files to %s", written, out_dir)



if __name__ == "__main__":
    main()
