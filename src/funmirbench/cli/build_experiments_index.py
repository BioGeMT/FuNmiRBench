import argparse
import csv
import json
import pathlib
import urllib.parse
from typing import List, Dict, Optional


DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[3]

# Defaults are repo-relative (like datasets.json "data_path")
DEFAULT_INFO_TSV = pathlib.Path("metadata/mirna_experiment_info.tsv")
DEFAULT_OUT_JSON = pathlib.Path("metadata/datasets.json")
DEFAULT_PROCESSED_DIR = pathlib.Path("data/experiments/processed")

REQUIRED_COLUMNS = [
    "mirna_name",
    "mirna_sequence",
    "article_pubmed_id",
    "tested_cell_line",
    "treatment",
    "tissue",
    "experiment_type",
    "gse_url",
    "de_table_path",
]


def map_experiment_type(experiment_type_raw: str) -> Optional[str]:
    """
    Map raw experiment_type values (OE, KO, KD) to:
    - 'overexpression'
    - 'knockdown'
    """
    if not experiment_type_raw:
        return None

    exp = experiment_type_raw.strip().upper()
    if exp == "OE":
        return "overexpression"
    if exp in ("KO", "KD"):
        return "knockdown"
    return None


def extract_geo_accession(gse_url: str) -> Optional[str]:
    """
    Extract GEO accession (e.g. GSE112859) from the URL.
    """
    if not gse_url:
        return None

    parsed = urllib.parse.urlparse(gse_url)
    qs = urllib.parse.parse_qs(parsed.query)
    acc = qs.get("acc", [None])[0]
    if acc:
        return acc

    return gse_url.rstrip("/").split("/")[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build metadata/datasets.json from metadata/mirna_experiment_info.tsv"
    )
    p.add_argument(
        "--info-tsv",
        type=pathlib.Path,
        default=DEFAULT_INFO_TSV,
        help="Input TSV with experiment metadata (default: metadata/mirna_experiment_info.tsv).",
    )
    p.add_argument(
        "--out-json",
        type=pathlib.Path,
        default=DEFAULT_OUT_JSON,
        help="Output JSON path (default: metadata/datasets.json).",
    )
    p.add_argument(
        "--processed-dir",
        type=pathlib.Path,
        default=DEFAULT_PROCESSED_DIR,
        help=(
            "Repo-relative base directory for processed experiment files. "
            "Used to form data_path (default: data/experiments/processed)."
        ),
    )
    p.add_argument(
        "--root",
        type=pathlib.Path,
        default=DEFAULT_ROOT,
        help="Project root directory used to resolve relative paths (default: repo root).",
    )
    return p.parse_args()


def _validate_header(fieldnames: Optional[List[str]], tsv_path: pathlib.Path) -> None:
    if not fieldnames:
        raise ValueError(f"TSV has no header row: {tsv_path}")
    missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    if missing:
        raise ValueError(f"TSV missing required columns {missing}: {tsv_path}")


def _resolve_under_root(root: pathlib.Path, p: pathlib.Path) -> pathlib.Path:
    """Resolve p to an absolute path, treating relative paths as root-relative."""
    return (root / p).resolve() if not p.is_absolute() else p.resolve()


def _root_relative_or_error(root: pathlib.Path, p: pathlib.Path, *, arg_name: str) -> pathlib.Path:
    """
    Ensure p is repo-relative (like datasets.json data_path semantics).

    If caller passes an absolute path, require that it is under root and convert to a root-relative path.
    """
    if not p.is_absolute():
        return p
    try:
        rel = p.resolve().relative_to(root.resolve())
    except Exception as exc:
        raise ValueError(
            f"{arg_name} must be repo-relative, or an absolute path under --root. "
            f"Got: {p} (root: {root})"
        ) from exc
    return rel


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    info_tsv = _resolve_under_root(root, args.info_tsv)
    out_json = _resolve_under_root(root, args.out_json)

    # Keep processed_dir written into JSON as repo-relative by default (like data_path),
    # but allow absolute paths IF they are under root (then convert to relative).
    processed_dir_rel = _root_relative_or_error(root, _resolve_under_root(root, args.processed_dir), arg_name="--processed-dir")

    if not info_tsv.exists():
        raise FileNotFoundError(f"Input TSV not found: {info_tsv}")

    datasets: List[Dict] = []

    with info_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        _validate_header(reader.fieldnames, info_tsv)

        # start=2 because row 1 is header
        for i, row in enumerate(reader, start=2):
            mirna_name = (row.get("mirna_name") or "").strip()
            mirna_seq = (row.get("mirna_sequence") or "").strip()
            pubmed_id = (row.get("article_pubmed_id") or "").strip()
            cell_line = (row.get("tested_cell_line") or "").strip() or None
            tissue = (row.get("tissue") or "").strip() or None
            treatment = (row.get("treatment") or "").strip() or None

            perturbation = map_experiment_type((row.get("experiment_type") or "").strip())
            gse_url = (row.get("gse_url") or "").strip()
            geo_accession = extract_geo_accession(gse_url)
            de_table_path = (row.get("de_table_path") or "").strip()

            if not mirna_name or not mirna_seq or not pubmed_id or not gse_url or not de_table_path:
                raise ValueError(f"Row {i} missing required values (check TSV): {info_tsv}")

            # Build a repo-relative data_path string
            data_path = (processed_dir_rel / de_table_path).as_posix()

            entry = {
                "id": f"{i-1:03d}",  # dataset IDs are 001-based (first data row is 2)
                "geo_accession": geo_accession,
                "miRNA": mirna_name,
                "miRNA_sequence": mirna_seq,
                "cell_line": cell_line,
                "tissue": tissue,
                "perturbation": perturbation,
                "organism": "Homo sapiens",
                "method": "RNA-Seq",
                "treatment": treatment,
                "pubmed_id": pubmed_id,
                "gse_url": gse_url,
                "data_path": data_path,
            }
            datasets.append(entry)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(datasets, f, indent=2)

    print(f"Wrote {len(datasets)} entries to {out_json}")


if __name__ == "__main__":
    main()
