"""Shared test fixtures."""

import pathlib
import sys
import textwrap

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def tmp_tsv_factory(tmp_path):
    """Factory that writes a TSV string to a temp file and returns its path."""

    def _make(content: str, name: str = "de_table.tsv") -> pathlib.Path:
        p = tmp_path / name
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p

    return _make
