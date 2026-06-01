"""Stream miRAW dictionary-like predictions into a compact best-per-pair TSV."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import logging
import shutil
import subprocess
from pathlib import Path
from typing import BinaryIO, Iterator

logger = logging.getLogger("miraw_preprocess")

_GENE_NAME_KEYS = (b"'GeneName'", b'"GeneName"')
_MIRNA_KEYS = (b"'miRNA'", b'"miRNA"')
_PREDICTION_KEYS = (b"'Prediction'", b'"Prediction"')
_WHITESPACE = b" \t\r\n"
_NUMBER_END = b",} \t\r\n"


def _open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", errors="replace")
    return open(path, mode, encoding="utf-8", errors="replace")


@contextmanager
def _open_input_bytes(path: Path) -> Iterator[BinaryIO]:
    if path.suffix == ".gz" and (pigz := shutil.which("pigz")):
        process = subprocess.Popen(
            [pigz, "-dc", str(path)],
            stdout=subprocess.PIPE,
        )
        if process.stdout is None:
            raise RuntimeError("Failed to open pigz stdout")
        try:
            yield process.stdout
        finally:
            process.stdout.close()
            return_code = process.wait()
            if return_code:
                raise RuntimeError(f"pigz failed with exit code {return_code}: {path}")
        return

    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            yield handle
        return

    with open(path, "rb") as handle:
        yield handle


def _extract_quoted_bytes(line: bytes, keys: tuple[bytes, ...]) -> bytes | None:
    for key in keys:
        start = line.find(key)
        if start == -1:
            continue
        colon = line.find(b":", start + len(key))
        if colon == -1:
            return None
        value_start = colon + 1
        while value_start < len(line) and line[value_start] in _WHITESPACE:
            value_start += 1
        if value_start >= len(line) or line[value_start] not in (ord("'"), ord('"')):
            return None
        quote = line[value_start]
        value_start += 1
        value_end = line.find(bytes((quote,)), value_start)
        if value_end == -1:
            return None
        return line[value_start:value_end].strip()
    return None


def _extract_prediction(line: bytes) -> float | None:
    for key in _PREDICTION_KEYS:
        start = line.find(key)
        if start == -1:
            continue
        colon = line.find(b":", start + len(key))
        if colon == -1:
            return None
        value_start = colon + 1
        while value_start < len(line) and line[value_start] in _WHITESPACE:
            value_start += 1
        value_end = value_start
        while value_end < len(line) and line[value_end] not in _NUMBER_END:
            value_end += 1
        try:
            return float(line[value_start:value_end])
        except ValueError:
            return None
    return None


def _parse_line(line: bytes) -> tuple[str, str, str, float] | None:
    raw_gene = _extract_quoted_bytes(line, _GENE_NAME_KEYS)
    mirna = _extract_quoted_bytes(line, _MIRNA_KEYS)
    score = _extract_prediction(line)
    if raw_gene is None or mirna is None or score is None:
        return None

    ensembl_raw, separator, raw_gene_name = raw_gene.partition(b"__")
    if not separator:
        return None
    ensembl_id = ensembl_raw.strip().split(b".", 1)[0]
    raw_gene_name = raw_gene_name.strip()
    if not raw_gene_name or raw_gene_name.startswith(b"biotype="):
        raw_gene_name = b""
    if not ensembl_id or not mirna:
        return None
    return (
        ensembl_id.decode("utf-8", errors="replace"),
        raw_gene_name.decode("utf-8", errors="replace"),
        mirna.decode("utf-8", errors="replace"),
        score,
    )


def preprocess_miraw_predictions(input_path: Path, output_path: Path) -> dict[str, int]:
    best: dict[tuple[str, str], tuple[float, str]] = {}
    rows_seen = 0
    rows_parsed = 0
    rows_replaced = 0
    rows_tied_or_lower = 0

    with _open_input_bytes(input_path) as handle:
        for rows_seen, line in enumerate(handle, start=1):
            parsed = _parse_line(line)
            if parsed is None:
                continue
            ensembl_id, raw_gene_name, mirna_name, score = parsed
            rows_parsed += 1
            key = (ensembl_id, mirna_name)
            previous = best.get(key)
            if previous is None or score > previous[0]:
                if previous is not None:
                    rows_replaced += 1
                best[key] = (score, raw_gene_name)
            else:
                rows_tied_or_lower += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text(output_path, "wt") as handle:
        handle.write("Ensembl_ID\tRaw_Gene_Name\tmiRNA_Name\tScore\n")
        for ensembl_id, mirna_name in sorted(best):
            score, raw_gene_name = best[(ensembl_id, mirna_name)]
            handle.write(f"{ensembl_id}\t{raw_gene_name}\t{mirna_name}\t{score:.17g}\n")

    stats = {
        "rows_seen": rows_seen,
        "rows_parsed": rows_parsed,
        "rows_skipped": rows_seen - rows_parsed,
        "unique_pairs": len(best),
        "rows_replaced_by_higher_score": rows_replaced,
        "rows_tied_or_lower_score": rows_tied_or_lower,
    }
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/all_ensgs.txt.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/best_per_pair.tsv.gz"))
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stats = preprocess_miraw_predictions(args.input, args.output)
    logger.info("Wrote %s", args.output)
    for key, value in stats.items():
        logger.info("%s=%d", key, value)


if __name__ == "__main__":
    main()
