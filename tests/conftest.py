"""Shared test fixtures."""

import pathlib
import textwrap

import pytest


@pytest.fixture()
def tmp_tsv_factory(tmp_path):
    """Factory that writes a TSV string to a temp file and returns its path."""

    def _make(content: str, name: str = "de_table.tsv") -> pathlib.Path:
        p = tmp_path / name
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p

    return _make


@pytest.fixture()
def fake_project_root(tmp_path):
    """Create a minimal fake project root with a pyproject.toml."""
    toml = tmp_path / "pyproject.toml"
    toml.write_text(
        '[project]\nname = "funmirbench"\nversion = "0.0.0"\n',
        encoding="utf-8",
    )
    return tmp_path
