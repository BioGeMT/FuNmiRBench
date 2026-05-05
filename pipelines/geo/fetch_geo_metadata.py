"""
Fetch GEO series metadata and propose a row for metadata/mirna_experiment_info.tsv.

Uses the GEO SOFT text API to retrieve series and sample-level metadata without
requiring SRA credentials. Outputs JSON with proposed field values and raw sample
information for human/LLM review.

Usage:
    python pipelines/geo/fetch_geo_metadata.py --gse-url GSE93717
    python pipelines/geo/fetch_geo_metadata.py --gse-url https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE93717
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests


GEO_SOFT_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    "?acc={gse}&targ=all&form=text&view=brief"
)

MIRBASE_MATURE_FA_URL = "https://mirbase.org/download/mature.fa"
MIRBASE_CACHE_MAX_AGE_DAYS = 30  # re-download if cache is older than this many days


# Keywords used to classify samples as likely control or likely condition.
# Scoring: each match adds +1 to that side; highest score wins.
_CONTROL_KEYWORDS = [
    r"\bcontrol\b", r"\bctrl\b", r"\bmock\b", r"\bscramble[d]?\b",
    r"\bneg(?:ative)?\b", r"\bNC\b", r"\bempty.vector\b",
    r"\bmir.ctrl\b", r"\bnon.targeting\b", r"\bwild.?type\b", r"\bWT\b",
]
_CONDITION_KEYWORDS = [
    r"\bmimic\b", r"\boverexpression\b", r"\bOE\b",
    r"\binhibitor\b", r"\bantago(?:mir)?\b", r"\bknockout\b",
    r"\bKO\b", r"\bknockdown\b", r"\bKD\b", r"\bsuppression\b",
    r"\btransfect\b",
]


# ---------------------------------------------------------------------------
# GEO SOFT parser
# ---------------------------------------------------------------------------

def extract_gse_accession(gse_url: str) -> str:
    gse_url = str(gse_url).strip()
    parsed = urlparse(gse_url)
    accession = parse_qs(parsed.query).get("acc", [""])[0].strip()
    if accession:
        return accession
    tail = parsed.path.rstrip("/").split("/")[-1].strip()
    if tail.upper().startswith("GSE"):
        return tail
    if gse_url.upper().startswith("GSE"):
        return gse_url
    raise ValueError(f"Cannot extract GSE accession from: {gse_url!r}")


def fetch_soft(gse: str) -> str:
    url = GEO_SOFT_URL.format(gse=gse)
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch GEO SOFT for {gse}: {e}")
    return response.text


def parse_soft(soft_text: str) -> dict:
    """
    Parse GEO SOFT text into a dict with keys:
      series: {field: value_or_list}
      samples: {GSM_id: {field: value_or_list}}
    """
    result = {"series": {}, "samples": {}}
    current_block = None
    current_id = None

    for line in soft_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Block headers
        if line.startswith("^SERIES"):
            current_block = "series"
            current_id = None
            continue
        if line.startswith("^SAMPLE"):
            parts = line.split("=", 1)
            current_id = parts[1].strip() if len(parts) > 1 else None
            current_block = "sample"
            if current_id:
                result["samples"].setdefault(current_id, {})
            continue
        if line.startswith("^"):
            current_block = None
            current_id = None
            continue

        # Field lines
        if not line.startswith("!"):
            continue
        parts = line[1:].split("=", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        value = parts[1].strip()

        if current_block == "series":
            existing = result["series"].get(key)
            if existing is None:
                result["series"][key] = value
            elif isinstance(existing, list):
                existing.append(value)
            else:
                result["series"][key] = [existing, value]
        elif current_block == "sample" and current_id:
            existing = result["samples"][current_id].get(key)
            if existing is None:
                result["samples"][current_id][key] = value
            elif isinstance(existing, list):
                existing.append(value)
            else:
                result["samples"][current_id][key] = [existing, value]

    return result


# ---------------------------------------------------------------------------
# Metadata extraction helpers
# ---------------------------------------------------------------------------

def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _first(value, default="") -> str:
    lst = _as_list(value)
    return lst[0].strip() if lst else default


def extract_pubmed_url(series: dict) -> str:
    pmids = _as_list(series.get("Series_pubmed_id"))
    if pmids:
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmids[0].strip()}"
    return ""


def extract_sample_info(sample_data: dict) -> dict:
    """Flatten a sample's SOFT fields into a clean dict."""
    chars = {}
    for val in _as_list(sample_data.get("Sample_characteristics_ch1")):
        if ":" in val:
            k, v = val.split(":", 1)
            chars[k.strip().lower()] = v.strip()

    # Cell line: try characteristics first, then source
    cell_line = (
        chars.get("cell line")
        or chars.get("cell type")
        or _first(sample_data.get("Sample_source_name_ch1"))
    )
    tissue = chars.get("tissue") or chars.get("tissue type") or ""
    organism = _first(sample_data.get("Sample_organism_ch1"))
    title = _first(sample_data.get("Sample_title"))

    return {
        "title": title,
        "organism": organism,
        "cell_line": cell_line,
        "tissue": tissue,
        "characteristics": chars,
    }


def _score_sample(title: str, chars: dict) -> tuple[int, int]:
    """Return (control_score, condition_score) for a sample."""
    text = (title + " " + " ".join(chars.values())).lower()
    ctrl = sum(1 for pat in _CONTROL_KEYWORDS if re.search(pat, text, re.IGNORECASE))
    cond = sum(1 for pat in _CONDITION_KEYWORDS if re.search(pat, text, re.IGNORECASE))
    return ctrl, cond


def classify_samples(samples: dict) -> dict:
    """
    Classify each GSM sample as 'control', 'condition', or 'uncertain'.

    Returns {gsm_id: {title, organism, cell_line, tissue, group, ctrl_score, cond_score}}.
    """
    classified = {}
    for gsm, data in samples.items():
        info = extract_sample_info(data)
        ctrl_score, cond_score = _score_sample(info["title"], info["characteristics"])

        if ctrl_score > cond_score:
            group = "control"
        elif cond_score > ctrl_score:
            group = "condition"
        else:
            group = "uncertain"

        classified[gsm] = {**info, "group": group, "ctrl_score": ctrl_score, "cond_score": cond_score}

    return classified


# ---------------------------------------------------------------------------
# miRBase sequence lookup
# ---------------------------------------------------------------------------

def _get_cached_mature_fa(cache_dir: str = "data/mirbase", max_age_days: int = MIRBASE_CACHE_MAX_AGE_DAYS) -> Path:
    """
    Return path to a local copy of miRBase mature.fa, downloading/refreshing as needed.

    The file is re-downloaded if it does not exist or is older than max_age_days.
    If a refresh fails but a stale copy exists, the stale copy is used with a warning.
    """
    cache_path = Path(cache_dir) / "mature.fa"

    needs_download = True
    if cache_path.exists():
        age_days = (time.time() - cache_path.stat().st_mtime) / 86400
        if age_days < max_age_days:
            needs_download = False
        else:
            print(
                f"[miRBase] Cache is {age_days:.0f} days old (limit {max_age_days}). Refreshing...",
                file=sys.stderr,
            )

    if needs_download:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[miRBase] Downloading mature.fa from {MIRBASE_MATURE_FA_URL} ...", file=sys.stderr)
        try:
            response = requests.get(MIRBASE_MATURE_FA_URL, timeout=120, stream=True)
            response.raise_for_status()
            tmp_path = cache_path.with_suffix(".fa.tmp")
            with open(tmp_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=65536):
                    fh.write(chunk)
            tmp_path.replace(cache_path)  # atomic rename
            print(f"[miRBase] Saved to {cache_path}", file=sys.stderr)
        except requests.RequestException as e:
            if cache_path.exists():
                print(
                    f"[miRBase] WARNING: refresh failed ({e}); using stale cache at {cache_path}",
                    file=sys.stderr,
                )
            else:
                raise RuntimeError(f"Failed to download miRBase mature.fa: {e}")

    return cache_path


def _parse_mature_fa(path: Path) -> dict:
    """Parse a miRBase mature.fa FASTA file into {mirna_name: sequence}."""
    sequences = {}
    current_name = None
    current_seq: list[str] = []

    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if current_name:
                    sequences[current_name] = "".join(current_seq)
                # Header format: >hsa-miR-21-5p MIMAT0000076 Homo sapiens miR-21-5p
                current_name = line[1:].split()[0]
                current_seq = []
            elif current_name:
                current_seq.append(line)
        if current_name:
            sequences[current_name] = "".join(current_seq)

    return sequences


def lookup_mirna_sequence(
    mirna_name: str,
    cache_dir: str = "data/mirbase",
    max_age_days: int = MIRBASE_CACHE_MAX_AGE_DAYS,
) -> str | None:
    """
    Return the mature sequence for *mirna_name* (e.g. ``hsa-miR-21-5p``).

    Downloads/refreshes miRBase mature.fa as needed (cached in *cache_dir*,
    refreshed when older than *max_age_days* days).  Returns ``None`` if the
    name is not found.
    """
    cache_path = _get_cached_mature_fa(cache_dir=cache_dir, max_age_days=max_age_days)
    sequences = _parse_mature_fa(cache_path)
    return sequences.get(mirna_name)


# ---------------------------------------------------------------------------
# Main proposal builder
# ---------------------------------------------------------------------------

def build_proposal(gse: str, soft_parsed: dict, mirna_sequence: str = "") -> dict:
    series = soft_parsed["series"]
    classified = classify_samples(soft_parsed["samples"])

    # Series-level fields
    title = _first(series.get("Series_title"))
    summary = _first(series.get("Series_summary"))
    pubmed_url = extract_pubmed_url(series)
    gse_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}"

    # Aggregate organism / cell line / tissue from samples (majority vote)
    organisms = [s["organism"] for s in classified.values() if s["organism"]]
    cell_lines = [s["cell_line"] for s in classified.values() if s["cell_line"]]
    tissues = [s["tissue"] for s in classified.values() if s["tissue"]]

    def majority(lst):
        if not lst:
            return ""
        return max(set(lst), key=lst.count)

    organism = majority(organisms)
    cell_line = majority(cell_lines)
    tissue = majority(tissues)

    # Sample groupings
    control_gsms = [gsm for gsm, s in classified.items() if s["group"] == "control"]
    condition_gsms = [gsm for gsm, s in classified.items() if s["group"] == "condition"]
    uncertain_gsms = [gsm for gsm, s in classified.items() if s["group"] == "uncertain"]

    # Fields the LLM must fill in (not derivable from GEO alone)
    needs_review = []
    if not control_gsms:
        needs_review.append("control_samples — no samples auto-classified as control")
    if not condition_gsms:
        needs_review.append("condition_samples — no samples auto-classified as condition")
    if uncertain_gsms:
        needs_review.append(f"sample grouping — {len(uncertain_gsms)} sample(s) unclassified: {uncertain_gsms}")

    needs_review += [
        "mirna_name — extract from series title/summary below",
        "experiment_type — OE (overexpression) or KO (knockout/knockdown/inhibition)",
        "id — derived as {gse}_{experiment_type}_{mirna_name_safe} once above are confirmed",
        "de_table_path — will be set to data/experiments/processed/{id}.tsv",
    ]
    if not mirna_sequence:
        needs_review.append(
            "mirna_sequence — run with --mirna-name <name> to auto-fetch from miRBase, "
            "or look up manually at https://mirbase.org"
        )

    return {
        "proposed_row": {
            "id": "",
            "mirna_name": "",
            "mirna_sequence": mirna_sequence,
            "article_pubmed_id": pubmed_url,
            "organism": organism,
            "tested_cell_line": cell_line,
            "treatment": "",
            "tissue": tissue,
            "method": "RNA-seq",
            "experiment_type": "",
            "gse_url": gse_url,
            "de_table_path": "",
            "raw_data_dir": "",
            "control_samples": ",".join(control_gsms),
            "condition_samples": ",".join(condition_gsms),
        },
        "series_info": {
            "gse": gse,
            "title": title,
            "summary": summary,
        },
        "sample_details": {
            gsm: {
                "title": s["title"],
                "organism": s["organism"],
                "cell_line": s["cell_line"],
                "tissue": s["tissue"],
                "group": s["group"],
                "scores": {"control": s["ctrl_score"], "condition": s["cond_score"]},
            }
            for gsm, s in classified.items()
        },
        "needs_review": needs_review,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fetch GEO series metadata and propose a mirna_experiment_info.tsv row. "
            "Pass --mirna-name alone to look up a miRNA sequence from miRBase."
        )
    )
    parser.add_argument(
        "--gse-url", required=False, default=None,
        help="GEO series URL or accession (e.g. GSE93717 or full GEO URL)",
    )
    parser.add_argument(
        "--mirna-name", required=False, default=None,
        help=(
            "miRNA name (e.g. hsa-miR-21-5p). When given, the mature sequence is "
            "fetched from a local miRBase cache (refreshed every "
            f"{MIRBASE_CACHE_MAX_AGE_DAYS} days) and included in the output."
        ),
    )
    args = parser.parse_args()

    # --- standalone sequence lookup ---
    if args.mirna_name and not args.gse_url:
        try:
            seq = lookup_mirna_sequence(args.mirna_name)
        except RuntimeError as e:
            print(json.dumps({"error": str(e)}))
            return 1
        if seq is None:
            print(json.dumps({
                "error": f"'{args.mirna_name}' not found in miRBase mature.fa. "
                         "Check the name or update the cache."
            }))
            return 1
        print(json.dumps({"mirna_name": args.mirna_name, "mirna_sequence": seq}))
        return 0

    # --- GEO metadata fetch (+ optional sequence lookup) ---
    if not args.gse_url:
        parser.error("Provide --gse-url (to fetch GEO metadata) or --mirna-name (for sequence lookup).")

    try:
        gse = extract_gse_accession(args.gse_url)
    except ValueError as e:
        print(json.dumps({"error": str(e)}))
        return 1

    try:
        soft_text = fetch_soft(gse)
    except RuntimeError as e:
        print(json.dumps({"error": str(e)}))
        return 1

    soft_parsed = parse_soft(soft_text)
    if not soft_parsed["series"]:
        print(json.dumps({"error": f"No series data found for {gse}. Check the accession."}))
        return 1

    mirna_sequence = ""
    if args.mirna_name:
        try:
            seq = lookup_mirna_sequence(args.mirna_name)
            if seq is None:
                print(
                    f"[miRBase] WARNING: '{args.mirna_name}' not found in mature.fa; "
                    "leaving mirna_sequence empty.",
                    file=sys.stderr,
                )
            else:
                mirna_sequence = seq
        except RuntimeError as e:
            print(f"[miRBase] WARNING: sequence lookup failed ({e}); leaving empty.", file=sys.stderr)

    proposal = build_proposal(gse, soft_parsed, mirna_sequence=mirna_sequence)
    print(json.dumps(proposal, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
