#!/usr/bin/env python3
"""Generate the weak demo predictor TSV."""

from pathlib import Path

from funmirbench.build_predictions import build_mock_scores, write_tsv


def log(message):
    print(message, flush=True)


def main():
    repo_root = Path(__file__).resolve().parents[3]
    experiments_tsv = repo_root / "metadata" / "mirna_experiment_info.tsv"
    out_path = repo_root / "data" / "predictions" / "mock" / "mock_canonical.tsv"
    log("Loading experiment registry for mock predictor...")
    scores = build_mock_scores(experiments_tsv, repo_root)
    log(f"Writing mock predictor TSV to {out_path}...")
    write_tsv(scores, out_path)
    log(f"Wrote {len(scores)} rows to {out_path}")


if __name__ == "__main__":
    main()
