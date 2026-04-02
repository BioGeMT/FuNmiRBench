#!/usr/bin/env python3
"""Generate the strong demo-only predictor TSV."""

from pathlib import Path

from funmirbench.build_cheating_predictions import DEMO_DATASET_IDS, build_cheating_scores
from funmirbench.build_predictions import write_tsv


def log(message):
    print(message, flush=True)


def main():
    repo_root = Path(__file__).resolve().parents[3]
    experiments_tsv = repo_root / "metadata" / "mirna_experiment_info.tsv"
    out_path = repo_root / "data" / "predictions" / "cheating" / "cheating_standardized.tsv"
    log("Loading experiment registry for demo strong predictor...")
    scores = build_cheating_scores(
        experiments_tsv,
        repo_root,
        dataset_ids=DEMO_DATASET_IDS,
    )
    log(f"Writing cheating predictor TSV to {out_path}...")
    write_tsv(scores, out_path)
    log(f"Wrote {len(scores)} rows to {out_path}")


if __name__ == "__main__":
    main()
