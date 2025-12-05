import csv
import json
import pathlib
import urllib.parse
from typing import List, Dict, Optional


ROOT = pathlib.Path(__file__).resolve().parents[1]
INFO_TSV = ROOT / "metadata" / "mirna_experiment_info.tsv"
OUT_JSON = ROOT / "metadata" / "datasets.json"


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


def main() -> None:
    datasets: List[Dict] = []

    with INFO_TSV.open("r", encoding="utf-8") as f:
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
                "data_path": f"data/processed_GEO/{de_table_path}"
            }

            datasets.append(entry)

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(datasets, f, indent=2)

    print(f"Wrote {len(datasets)} entries to {OUT_JSON}")


if __name__ == "__main__":
    main()
