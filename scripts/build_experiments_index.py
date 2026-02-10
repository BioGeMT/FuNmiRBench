import argparse
import csv
import json
import pathlib
import urllib.parse
from typing import List, Dict, Optional


DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_INFO_TSV = DEFAULT_ROOT / "metadata" / "mirna_experiment_info.tsv"
DEFAULT_OUT_JSON = DEFAULT_ROOT / "metadata" / "datasets.json"
DEFAULT_PROCESSED_DIR = pathlib.Path("data") / "processed_GEO"


def map_experiment_type(experiment_type_raw: str) -> Optional[str]:
    """
    Map raw experiment_type values (OE, KO) to:
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
        help="Base directory for processed experiment files, used to form data_path "
             "(default: data/processed_GEO).",
    )
    p.add_argument(
        "--root",
        type=pathlib.Path,
        default=DEFAULT_ROOT,
        help="Project root directory used to resolve default paths (default: repo root).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve defaults relative to root if caller set --root explicitly.
    root = args.root.expanduser().resolve()

    info_tsv: pathlib.Path = args.info_tsv
    out_json: pathlib.Path = args.out_json
    processed_dir: pathlib.Path = args.processed_dir

    # If user passed relative paths, interpret them relative to root
    if not info_tsv.is_absolute():
        info_tsv = root / info_tsv
    if not out_json.is_absolute():
        out_json = root / out_json

    # processed_dir is written into JSON as a relative path by default;
    # keep it relative unless user explicitly provides an absolute path.
    if processed_dir.is_absolute():
        processed_dir_rel = processed_dir
    else:
        processed_dir_rel = processed_dir

    datasets: List[Dict] = []

    if not info_tsv.exists():
        raise FileNotFoundError(f"Input TSV not found: {info_tsv}")

    with info_tsv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for i, row in enumerate(reader, start=1):
            mirna_name = row["mirna_name"].strip()
            mirna_seq = row["mirna_sequence"].strip()
            pubmed_id = row["article_pubmed_id"].strip()
            cell_line = row["tested_cell_line"].strip() or None
            tissue = row["tissue"].strip() or None
            treatment = row["treatment"].strip() or None

            perturbation = map_experiment_type(row["experiment_type"])
            gse_url = row["gse_url"].strip()
            geo_accession = extract_geo_accession(gse_url)
            de_table_path = row["de_table_path"].strip()

            # Build the data_path in a configurable way (default keeps old layout).
            data_path = (processed_dir_rel / de_table_path).as_posix()

            entry = {
                "id": f"{i:03d}",
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
