import argparse
import csv
import json
import pathlib
import re
from typing import Dict, List


DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_INFO_TSV = pathlib.Path("metadata/predictions_info.tsv")
DEFAULT_OUT_JSON = pathlib.Path("metadata/predictions.json")

REQUIRED_COLUMNS = [
    "tool_id",
    "official_name",
    "organism",
    "score_type",
    "score_direction",
    "score_range",
    "input_id_gene_type",
    "canonical_id_gene_type",
    "input_id_mirna_type",
    "canonical_id_mirna_type",
    "canonical_tsv_path",
]

# Fixed-ish ID-type format: <database>_<version>
IDTYPE_RE = re.compile(r"^[a-z0-9]+_[a-z0-9._-]+$", re.IGNORECASE)

SCORE_DIRECTIONS = {"higher_is_stronger", "lower_is_stronger", "unknown"}
SCORE_TYPES = {"probability", "regression", "rank", "pvalue", "energy", "binary", "other"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build metadata/predictions.json from metadata/predictions_info.tsv"
    )
    p.add_argument("--info-tsv", type=pathlib.Path, default=DEFAULT_INFO_TSV)
    p.add_argument("--out-json", type=pathlib.Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    return p.parse_args()


def _validate_row(row: Dict[str, str], row_num: int) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in row or row[c].strip() == ""]
    if missing:
        raise ValueError(f"Row {row_num} is missing required columns: {missing}")

    sd = row["score_direction"].strip()
    if sd not in SCORE_DIRECTIONS:
        raise ValueError(
            f"Row {row_num} has invalid score_direction {sd!r}. "
            f"Allowed: {sorted(SCORE_DIRECTIONS)}"
        )

    st = row["score_type"].strip()
    if st not in SCORE_TYPES:
        raise ValueError(
            f"Row {row_num} has invalid score_type {st!r}. "
            f"Allowed: {sorted(SCORE_TYPES)}"
        )

    for k in (
        "input_id_gene_type",
        "canonical_id_gene_type",
        "input_id_mirna_type",
        "canonical_id_mirna_type",
    ):
        v = row[k].strip()
        if not IDTYPE_RE.match(v):
            raise ValueError(
                f"Row {row_num} has invalid {k}={v!r}. "
                "Expected format like 'ensembl_109' or 'mirbase_v22'."
            )

    # canonical_tsv_path should be a relative path (like datasets.json data_path)
    p = row["canonical_tsv_path"].strip()
    if p.startswith("/") or p.startswith("~"):
        raise ValueError(
            f"Row {row_num} canonical_tsv_path must be a repo-relative path (got {p!r})."
        )
    if not p.endswith(".tsv"):
        raise ValueError(
            f"Row {row_num} canonical_tsv_path should point to a .tsv file (got {p!r})."
        )


def main() -> None:
    args = parse_args()

    root = args.root.expanduser().resolve()
    info_tsv = args.info_tsv
    out_json = args.out_json

    if not info_tsv.is_absolute():
        info_tsv = root / info_tsv
    if not out_json.is_absolute():
        out_json = root / out_json

    if not info_tsv.exists():
        raise FileNotFoundError(f"Input TSV not found: {info_tsv}")

    entries: List[Dict[str, str]] = []
    seen_tool_ids = set()

    with info_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header row: {info_tsv}")

        for i, row in enumerate(reader, start=2):
            _validate_row(row, i)

            tool_id = row["tool_id"].strip()
            if tool_id in seen_tool_ids:
                raise ValueError(f"Duplicate tool_id {tool_id!r} at row {i}")
            seen_tool_ids.add(tool_id)

            entries.append(
                {
                    "tool_id": tool_id,
                    "official_name": row["official_name"].strip(),
                    "organism": row["organism"].strip(),
                    "score_type": row["score_type"].strip(),
                    "score_direction": row["score_direction"].strip(),
                    "score_range": row["score_range"].strip(),
                    "input_id_gene_type": row["input_id_gene_type"].strip(),
                    "canonical_id_gene_type": row["canonical_id_gene_type"].strip(),
                    "input_id_mirna_type": row["input_id_mirna_type"].strip(),
                    "canonical_id_mirna_type": row["canonical_id_mirna_type"].strip(),
                    "canonical_tsv_path": row["canonical_tsv_path"].strip(),
                }
            )

    entries = sorted(entries, key=lambda d: d["tool_id"])

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    print(f"Wrote {len(entries)} entries to {out_json}")


if __name__ == "__main__":
    main()
