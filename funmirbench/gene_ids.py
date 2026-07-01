"""Gene identifier normalization helpers."""

from __future__ import annotations


def strip_ensembl_version(value: object) -> str:
    return str(value).strip().split(".", 1)[0]
