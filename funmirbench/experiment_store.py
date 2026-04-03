"""Helpers for caching benchmark experiment DE tables from Zenodo."""

from __future__ import annotations

import hashlib
import pathlib
from urllib.parse import quote

import requests


ZENODO_RECORD = "18745741"
ZENODO_API_RECORD_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}"


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def experiments_processed_dir(*, repo: pathlib.Path | None = None) -> pathlib.Path:
    repo = (repo or repo_root()).resolve()
    return repo / "data" / "experiments" / "processed"


def compute_md5(path: str | pathlib.Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    path = pathlib.Path(path)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksum(value: str) -> tuple[str, str]:
    algorithm, _, digest = str(value).partition(":")
    if not algorithm or not digest:
        raise ValueError(f"Unsupported checksum value: {value!r}")
    return algorithm.lower(), digest.lower()


def zenodo_download_url(filename: str) -> str:
    return (
        f"{ZENODO_API_RECORD_URL}/files/{quote(str(filename), safe='')}/content"
    )


def fetch_zenodo_file_registry(
    *,
    token: str | None = None,
    timeout: int = 120,
) -> dict[str, dict]:
    url = ZENODO_API_RECORD_URL
    if token:
        url = f"{url}?token={token}"

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    registry = {}
    for item in payload.get("files", []):
        key = str(item["key"])
        registry[key] = {
            "filename": key,
            "size": int(item.get("size", 0) or 0),
            "checksum": str(item.get("checksum", "")),
            "url": item.get("links", {}).get("self") or zenodo_download_url(key),
        }
    return registry


def resolve_cached_experiment_path(
    de_table_path: str | pathlib.Path,
    *,
    repo: pathlib.Path | None = None,
) -> pathlib.Path:
    path = pathlib.Path(de_table_path)
    if path.is_absolute():
        return path
    return (repo or repo_root()).resolve() / path


def ensure_zenodo_experiment_cached(
    de_table_path: str | pathlib.Path,
    *,
    repo: pathlib.Path | None = None,
    registry: dict[str, dict] | None = None,
    token: str | None = None,
    timeout: int = 120,
    force: bool = False,
) -> pathlib.Path:
    dest = resolve_cached_experiment_path(de_table_path, repo=repo)
    filename = dest.name
    registry = registry or fetch_zenodo_file_registry(token=token, timeout=timeout)

    if filename not in registry:
        raise KeyError(
            f"{filename!r} is not present in Zenodo record {ZENODO_RECORD}."
        )

    meta = registry[filename]
    checksum_value = str(meta.get("checksum", "") or "")

    if dest.exists() and not force:
        if checksum_value:
            algorithm, expected_digest = parse_checksum(checksum_value)
            if algorithm != "md5":
                raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
            if compute_md5(dest) != expected_digest:
                raise ValueError(
                    f"Existing file {dest} failed checksum verification for {filename}."
                )
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(str(meta["url"]), stream=True, timeout=timeout)
    response.raise_for_status()
    with dest.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)

    if checksum_value:
        algorithm, expected_digest = parse_checksum(checksum_value)
        if algorithm != "md5":
            raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
        actual_digest = compute_md5(dest)
        if actual_digest != expected_digest:
            raise ValueError(
                f"Checksum mismatch for {filename}: expected {expected_digest}, got {actual_digest}."
            )

    return dest
