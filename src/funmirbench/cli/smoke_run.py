"""
FuNmiRBench smoke-run

Runs a minimal end-to-end workflow:
1) Download experiments from Zenodo
2) Build experiment index
3) Validate experiments
4) Build mock predictor
5) Evaluate a small subset of datasets
"""

import argparse
import sys

from funmirbench import datasets
from funmirbench.cli import (
    import_experiments,
    build_experiments_index,
    validate_experiments,
    build_predictions,
    join_experiment_predictions,
    plot_correlation,
)


def parse_args():
    p = argparse.ArgumentParser(description="Run FuNmiRBench smoke test")
    p.add_argument("--experiment-type", choices=["OE", "KO", "KD"])
    p.add_argument("--cell-line")
    p.add_argument("--mirna")
    p.add_argument("--limit", type=int, default=2)
    return p.parse_args()


def _run_cli(module, argv):
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = old_argv


def _map_experiment_type_to_perturbation(experiment_type):
    if experiment_type is None:
        return None
    exp = experiment_type.strip().upper()
    if exp == "OE":
        return "overexpression"
    if exp in ("KO", "KD"):
        return "knockdown"
    raise ValueError("experiment_type must be one of: OE, KO, KD")


def main():
    args = parse_args()

    # print("STEP 1 — Download experiments from Zenodo")
    # _run_cli(import_experiments, ["import_experiments"])

    print("STEP 2 — Build experiments index")
    _run_cli(build_experiments_index, ["build_experiments_index"])

    print("STEP 3 — Validate experiments")
    _run_cli(validate_experiments, ["validate_experiments"])

    print("STEP 4 — Build mock predictions")
    _run_cli(build_predictions, ["build_predictions", "--tool", "mock"])

    print("STEP 5 — Select datasets")

    perturbation = _map_experiment_type_to_perturbation(args.experiment_type)

    metas = datasets.list_datasets(
        miRNA=args.mirna,
        cell_line=args.cell_line,
        perturbation=perturbation,
    )

    if not metas:
        raise SystemExit(
            "No datasets matched the filters. "
            "Check exact spellings for --cell-line / --mirna or change --experiment-type."
        )

    selected = metas[: max(args.limit, 1)]
    selected_ids = [m.id for m in selected]

    print("Selected dataset IDs:", ", ".join(selected_ids))

    for dataset_id in selected_ids:
        print(f"  - Joining dataset {dataset_id} with mock predictions")

        joined_out = f"data/joined/{dataset_id}_mock.tsv"

        _run_cli(
            join_experiment_predictions,
            [
                "join_experiment_predictions",
                "--dataset-id", dataset_id,
                "--tool", "mock",
                "--out", joined_out,
            ],
        )

        print(f"  - Plotting correlation for dataset {dataset_id}")

        _run_cli(
            plot_correlation,
            [
                "plot_correlation",
                "--joined-tsv", joined_out,
                "--out-dir", "data/plots",
            ],
        )


if __name__ == "__main__":
    main()