import argparse
import csv
import json
import pathlib
from typing import Dict, List


DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INFO_TSV = DEFAULT_ROOT / "metadata" / "predictions_info.tsv"
DEFAULT_OUT_JSON = DEFAULT_ROOT / "metadata" / "predictions.json"


REQUIRED_COLUMNS = [
    "tool_id",
    "display_name",
    "organism",
    "score_direction",
    "score_range",
    "input_id_type",
    "canonical_id_type",
    "canonical_tsv_path",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build metadata/predictions.json from metadata/predictions_info.tsv"
    )
    p.add_argument(
        "--info-tsv",
        type=pathlib.Path,
        default=DEFAULT_INFO_TSV,
        help="Input TSV with prediction resources (default: metadata/predictions_info.tsv).",
    )
    p.add_argument(
        "--out-json",
        type=pathlib.Path,
        default=DEFAULT_OUT_JSON,
        help="Output JSON path (default: metadata/predictions.json).",
    )
    p.add_argument(
        "--root",
        type=pathlib.Path,
        default=DEFAULT_ROOT,
        help="Project root directory used to resolve relative paths (default: repo root).",
    )
    return p.parse_args()


def _validate_row(row: Dict[str, str], row_num: int) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in row or row[c].strip() == ""]
    if missing:
        raise ValueError(f"Row {row_num} is missing required columns: {missing}")

    # Minimal validation for score_range like "0-1" or "0-100"
    score_range = row["score_range"].strip()
    if "-" not in score_range:
        raise ValueError(
            f"Row {row_num} has invalid score_range {score_range!r}; expected like '0-1' or '0-100'."
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

        for i, row in enumerate(reader, start=2):  # header is line 1
            _validate_row(row, i)

            tool_id = row["tool_id"].strip()
            if tool_id in seen_tool_ids:
                raise ValueError(f"Duplicate tool_id {tool_id!r} at row {i}")
            seen_tool_ids.add(tool_id)

            entry = {
                "tool_id": tool_id,
                "display_name": row["display_name"].strip(),
                "organism": row["organism"].strip(),
                "score_direction": row["score_direction"].strip(),
                "score_range": row["score_range"].strip(),
                "input_id_type": row["input_id_type"].strip(),
                "canonical_id_type": row["canonical_id_type"].strip(),
                "canonical_tsv_path": row["canonical_tsv_path"].strip(),
            }
            entries.append(entry)

    # Stable output order
    entries = sorted(entries, key=lambda d: d["tool_id"])

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    print(f"Wrote {len(entries)} entries to {out_json}")


if __name__ == "__main__":
    main()
