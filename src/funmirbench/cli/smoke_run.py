"""
FuNmiRBench smoke-run

Runs a minimal end-to-end workflow:
1) Build experiment index
2) Validate experiments
3) Build mock predictor
4) Interactively select a small subset of datasets
5) Join and run full evaluation plots/reports

Notes
-----
- Zenodo download is intentionally not run here for now.
- This script can be used interactively, or with CLI args to skip prompts.
- Assumes plot_correlation.py is available as funmirbench.cli.plot_correlation
"""

from __future__ import annotations

import argparse
import sys

from pyfiglet import figlet_format
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from funmirbench import datasets
from funmirbench.cli import (
    build_experiments_index,
    validate_experiments,
    build_predictions,
    join_experiment_predictions,
    plot_correlation,
)


console = Console()


def parse_args():
    p = argparse.ArgumentParser(description="Run FuNmiRBench smoke test")
    p.add_argument("--experiment-type", choices=["OE", "KO", "KD", "ALL"], default=None)
    p.add_argument("--cell-line", default=None)
    p.add_argument("--mirna", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--fdr-threshold", type=float, default=0.05)
    p.add_argument("--abs-logfc-threshold", type=float, default=1.0)
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
    if exp == "ALL":
        return None
    if exp == "OE":
        return "overexpression"
    if exp in ("KO", "KD"):
        return "knockdown"
    raise ValueError("experiment_type must be one of: OE, KO, KD, ALL")


def _prompt_if_missing(value, prompt_text, default=None):
    if value is not None:
        return value
    if default is not None:
        raw = input(f"{prompt_text} [{default}]: ").strip()
        return raw or default
    raw = input(f"{prompt_text}: ").strip()
    return raw or None


def _normalize_blank(value):
    if value is None:
        return None
    value = value.strip()
    return value or None


def _make_options_table(title, values, column_name):
    table = Table(title=title, expand=True)
    table.add_column(column_name, style="bright_cyan")
    if not values:
        table.add_row("(none)")
    else:
        for v in values:
            table.add_row(v)
    return table


def _print_banner():
    banner = figlet_format("FuNmiRBench", font="slant")
    console.print(
        Panel.fit(
            f"[bold bright_magenta]{banner}[/bold bright_magenta]",
            border_style="bright_magenta",
            padding=(0, 1),
        )
    )
    console.print(
        "[bold cyan]Fu[/bold cyan]"
        "[bold magenta]N[/bold magenta]"
        "[bold bright_green]miR[/bold bright_green]"
        "[bold yellow]Bench[/bold yellow] "
        "[dim]- Functional miRNA Benchmark[/dim]\n"
    )


def _make_run_outline_panel():
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="white")
    table.add_row("STEP 1", "Download experiments")
    table.add_row("STEP 2", "Build index")
    table.add_row("STEP 3", "Validate")
    table.add_row("STEP 4", "Build predictions")
    table.add_row("STEP 5", "Run benchmark")
    return Panel.fit(table, title="FuNmiRBench banner", border_style="cyan")


def _make_summary_panel():
    metas = datasets.load_metadata()
    summary = Table(show_header=False, box=None, padding=(0, 1))
    summary.add_column(style="bold green", no_wrap=True)
    summary.add_column(style="white")
    summary.add_row("Datasets available", str(len(metas)))
    summary.add_row("Unique miRNAs", str(len(datasets.list_mirnas())))
    summary.add_row("Unique cell lines", str(len(datasets.list_cell_lines())))
    summary.add_row("Predictors", "mock")
    return Panel.fit(summary, title="Current benchmark summary", border_style="green")


def _print_selected_datasets_table(selected):
    table = Table(title="Selected Datasets")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("miRNA", style="bright_green")
    table.add_column("Cell Line", style="yellow")
    table.add_column("Perturbation", style="magenta")
    table.add_column("GEO", style="blue")
    for m in selected:
        table.add_row(
            m.id,
            m.miRNA or "",
            m.cell_line or "",
            m.perturbation or "",
            m.geo_accession or "",
        )
    console.print(table)


def _print_final_outputs_table(selected_ids):
    table = Table(title="Generated Outputs")
    table.add_column("Dataset ID", style="cyan", no_wrap=True)
    table.add_column("Joined TSV", style="bright_green")
    table.add_column("Evaluation outputs", style="yellow")
    for dataset_id in selected_ids:
        table.add_row(
            dataset_id,
            f"data/joined/{dataset_id}_mock.tsv",
            "data/plots/",
        )
    console.print(table)


def main():
    _print_banner()
    args = parse_args()

    console.print(Columns([_make_run_outline_panel(), _make_summary_panel()], expand=True, equal=True))

    console.rule("[bold blue]STEP 2 — Build experiments index[/bold blue]")
    _run_cli(build_experiments_index, ["build_experiments_index"])

    console.rule("[bold blue]STEP 3 — Validate experiments[/bold blue]")
    _run_cli(validate_experiments, ["validate_experiments"])

    console.rule("[bold blue]STEP 4 — Build mock predictions[/bold blue]")
    _run_cli(build_predictions, ["build_predictions", "--tool", "mock"])

    console.rule("[bold blue]STEP 5 — Interactive dataset selection[/bold blue]")

    cell_lines = datasets.list_cell_lines()
    mirnas = datasets.list_mirnas()

    console.print("[bold]Experiment type options[/bold]")
    console.print("  [cyan]OE[/cyan]  [dim](overexpression)[/dim]")
    console.print("  [magenta]KO[/magenta]  [dim](treated as knockdown in the benchmark index)[/dim]")
    console.print("  [magenta]KD[/magenta]  [dim](treated as knockdown in the benchmark index)[/dim]")
    console.print("  [yellow]ALL[/yellow]\n")

    cell_table = _make_options_table("Available Cell Lines", cell_lines, "Cell Line")
    mirna_table = _make_options_table("Available miRNAs", mirnas, "miRNA")
    console.print(Columns([cell_table, mirna_table], expand=True, equal=True))

    args.experiment_type = _prompt_if_missing(
        args.experiment_type,
        "Choose experiment type",
        default="ALL",
    )
    args.cell_line = _prompt_if_missing(
        args.cell_line,
        "Choose cell line exactly as shown above, or leave blank for all",
        default="",
    )
    args.mirna = _prompt_if_missing(
        args.mirna,
        "Choose miRNA exactly as shown above, or leave blank for all",
        default="",
    )
    limit_raw = _prompt_if_missing(
        str(args.limit) if args.limit is not None else None,
        "How many datasets should be evaluated",
        default="2",
    )

    args.experiment_type = args.experiment_type.strip().upper()
    args.cell_line = _normalize_blank(args.cell_line)
    args.mirna = _normalize_blank(args.mirna)
    args.limit = int(limit_raw)

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

    _print_selected_datasets_table(selected)

    for m in selected:
        dataset_id = m.id
        console.rule(f"[bold green]Running dataset {dataset_id}[/bold green]")

        joined_out = f"data/joined/{dataset_id}_mock.tsv"

        console.print(f"[cyan]Joining[/cyan] dataset [bold]{dataset_id}[/bold] with mock predictions")
        _run_cli(
            join_experiment_predictions,
            [
                "join_experiment_predictions",
                "--dataset-id", dataset_id,
                "--tool", "mock",
                "--out", joined_out,
            ],
        )

        console.print(f"[magenta]Evaluating[/magenta] plots and metrics for dataset [bold]{dataset_id}[/bold]")
        _run_cli(
            plot_correlation,
            [
                "plot_correlation",
                "--joined-tsv", joined_out,
                "--out-dir", "data/plots",
                "--dataset-id", dataset_id,
                "--mirna", m.miRNA or "",
                "--cell-line", m.cell_line or "",
                "--perturbation", m.perturbation or "",
                "--geo-accession", m.geo_accession or "",
                "--top-n", str(args.top_n),
                "--fdr-threshold", str(args.fdr_threshold),
                "--abs-logfc-threshold", str(args.abs_logfc_threshold),
            ],
        )

    console.rule("[bold bright_green]Smoke-run complete[/bold bright_green]")
    _print_final_outputs_table(selected_ids)


if __name__ == "__main__":
    main()
