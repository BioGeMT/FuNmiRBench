"""Project root and path resolution utilities."""

from __future__ import annotations

import os
import pathlib
from typing import Optional

_cached_root: Optional[pathlib.Path] = None


def _find_root_by_marker() -> pathlib.Path:
    """Walk up from this file looking for pyproject.toml with funmirbench."""
    current = pathlib.Path(__file__).resolve().parent
    for parent in (current, *current.parents):
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            try:
                text = candidate.read_text(encoding="utf-8")
                if 'name = "funmirbench"' in text:
                    return parent
            except OSError:
                continue
    raise RuntimeError(
        "Could not locate FuNmiRBench project root (no pyproject.toml with "
        'name = "funmirbench" found in parent directories).'
    )


def project_root(override: Optional[pathlib.Path] = None) -> pathlib.Path:
    """
    Resolve the FuNmiRBench project root directory.

    Priority: explicit override > FUNMIRBENCH_ROOT env var > marker-file search.
    """
    if override is not None:
        return override.expanduser().resolve()

    env_root = os.getenv("FUNMIRBENCH_ROOT")
    if env_root:
        return pathlib.Path(env_root).expanduser().resolve()

    global _cached_root
    if _cached_root is None:
        _cached_root = _find_root_by_marker()
    return _cached_root


def resolve_path(root: pathlib.Path, p: pathlib.Path) -> pathlib.Path:
    """Resolve *p* to an absolute path, treating relative paths as root-relative."""
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def root_relative_path(
    root: pathlib.Path,
    p: pathlib.Path,
    *,
    label: str = "path",
) -> pathlib.Path:
    """
    Ensure *p* is repo-relative.

    If *p* is already relative, return it unchanged.
    If absolute, convert to a path relative to *root* (raising ValueError if
    *p* is not under *root*).
    """
    if not p.is_absolute():
        return p
    try:
        return p.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{label} must be repo-relative, or an absolute path under root. "
            f"Got: {p} (root: {root})"
        ) from exc
