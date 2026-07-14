#!/usr/bin/env python3
"""Generate the coverage panels for manuscript Figure 2.

The script reads the ``joined.tsv`` files from one completed FuNmiRBench run and
writes each panel as a separate publication-ready plot plus the exact summary
values used to draw it.

Panel definitions
-----------------
A. Number of benchmark experiments retained for each predictor.
B. Predictor gene-set coverage across the full benchmark gene universe, plus
   the Intersection Gene Set (IGS) shared by all predictors.
C. Coverage of perturbation-consistent positive miRNA-gene pairs.
D. Coverage of background/non-positive miRNA-gene pairs.

Examples
--------
Generate all panels from a specific run::

    uv run python scripts/figure2_coverage.py \
        --run-dir results/20260709_122904 \
        --panel all

Outputs are written by default to::

    results/manuscript_figure2/<run-name>/panel_<letter>/
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from funmirbench.evaluate_common import CURVE_COLORS


DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_METADATA_PATH = Path("metadata/predictions_info.tsv")
DEFAULT_OUTPUT_ROOT = Path("results/manuscript_figure2")
DEFAULT_FORMATS = ("png", "svg")
RUN_EFFECT_THRESHOLD = 1.0
RUN_FDR_THRESHOLD = 0.05
PANEL_CHOICES = ("A", "B", "C", "D", "all")
IGS_COLOR = "#9E9E9E"
FGS_COLOR = "#424242"

OE_ALIASES = {
    "OE",
    "OVEREXPRESSION",
    "OVER_EXPRESSION",
}
LOSS_ALIASES = {
    "KO",
    "KD",
    "KNOCKOUT",
    "KNOCK_OUT",
    "KNOCKDOWN",
    "KNOCK_DOWN",
}


@dataclass(frozen=True)
class BenchmarkDataset:
    """One benchmark dataset loaded from a completed run."""

    dataset_id: str
    frame: pd.DataFrame


@dataclass(frozen=True)
class Figure2Inputs:
    """Validated inputs shared by all Figure 2 panels."""

    run_dir: Path
    datasets: tuple[BenchmarkDataset, ...]
    tool_ids: tuple[str, ...]
    tool_labels: dict[str, str]
    tool_colors: dict[str, str]
    fdr_threshold: float | None
    effect_threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Figure 2 coverage panels from a FuNmiRBench run."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Completed benchmark run directory. If omitted, the newest directory "
            "under results/ containing datasets/*/joined.tsv is used."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root used when auto-selecting the newest completed run.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Predictor metadata TSV used for display names and predictor order.",
    )
    parser.add_argument(
        "--panel",
        choices=PANEL_CHOICES,
        default="all",
        help="Generate one panel or all panels.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: results/manuscript_figure2/<run-name>/.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster output resolution.",
    )
    return parser.parse_args()

def find_latest_completed_run(results_dir: Path) -> Path:
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    candidates = []
    for child in results_dir.iterdir():
        if child.is_dir() and list(child.glob("datasets/*/joined.tsv")):
            candidates.append(child)

    if not candidates:
        raise FileNotFoundError(
            f"No completed benchmark run with datasets/*/joined.tsv found under {results_dir}."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_tool_metadata(path: Path) -> tuple[tuple[str, ...], dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Predictor metadata does not exist: {path}")

    metadata = pd.read_csv(path, sep="\t", dtype=str)
    required = {"tool_id", "official_name"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    metadata = metadata.dropna(subset=["tool_id"]).copy()
    metadata["tool_id"] = metadata["tool_id"].astype(str).str.strip()
    metadata["official_name"] = metadata["official_name"].fillna(metadata["tool_id"])
    metadata["official_name"] = metadata["official_name"].astype(str).str.strip()

    tool_ids = tuple(metadata["tool_id"].tolist())
    labels = dict(zip(metadata["tool_id"], metadata["official_name"]))
    return tool_ids, labels


def load_joined_datasets(run_dir: Path) -> tuple[BenchmarkDataset, ...]:
    joined_paths = sorted(run_dir.glob("datasets/*/joined.tsv"))
    if not joined_paths:
        raise FileNotFoundError(f"No datasets/*/joined.tsv files found in {run_dir}")

    datasets = []
    for path in joined_paths:
        frame = pd.read_csv(path, sep="\t", low_memory=False)
        dataset_id = path.parent.name
        if "dataset_id" in frame.columns and not frame.empty:
            dataset_id = str(frame["dataset_id"].iloc[0])
        datasets.append(BenchmarkDataset(dataset_id=dataset_id, frame=frame))
    return tuple(datasets)


def score_column(tool_id: str) -> str:
    return f"score_{tool_id}"


def available_tool_ids(
    datasets: Iterable[BenchmarkDataset], metadata_order: Iterable[str]
) -> tuple[str, ...]:
    datasets = tuple(datasets)
    common_score_columns: set[str] | None = None
    for dataset in datasets:
        current = {column for column in dataset.frame.columns if column.startswith("score_")}
        common_score_columns = (
            current if common_score_columns is None else common_score_columns & current
        )

    common_score_columns = common_score_columns or set()
    common_tool_ids = {column.removeprefix("score_") for column in common_score_columns}
    ordered = tuple(tool_id for tool_id in metadata_order if tool_id in common_tool_ids)
    extras = sorted(common_tool_ids.difference(ordered))
    resolved = ordered + tuple(extras)
    if not resolved:
        raise ValueError("No score_<tool_id> columns are shared by all joined datasets.")
    return resolved


def evaluation_tool_colors(tool_ids: Iterable[str]) -> dict[str, str]:
    """Reproduce the predictor colors used by the evaluation pipeline.

    ``evaluate_joined_dataframe`` discovers ``score_*`` columns in sorted order
    and assigns ``CURVE_COLORS`` sequentially. The mapping here deliberately
    follows that same ordering, independently of the display order in metadata.
    """

    evaluation_order = sorted(str(tool_id) for tool_id in tool_ids)
    return {
        tool_id: CURVE_COLORS[index % len(CURVE_COLORS)]
        for index, tool_id in enumerate(evaluation_order)
    }


def perturbation_series(frame: pd.DataFrame) -> pd.Series:
    if "perturbation" not in frame.columns:
        raise ValueError("Joined table is missing the perturbation column.")
    return (
        frame["perturbation"]
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"[\s-]+", "_", regex=True)
    )


def annotate_ground_truth(
    frame: pd.DataFrame,
    *,
    fdr_threshold: float | None,
    effect_threshold: float,
) -> pd.DataFrame:
    required = {"gene_id", "logFC", "perturbation"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Joined table is missing required columns: {sorted(missing)}")

    work = frame.copy()
    work["gene_id"] = work["gene_id"].astype(str)
    work["logFC"] = pd.to_numeric(work["logFC"], errors="coerce")

    perturbation = perturbation_series(work)
    oe_mask = perturbation.isin(OE_ALIASES)
    loss_mask = perturbation.isin(LOSS_ALIASES)
    unsupported = ~(oe_mask | loss_mask)
    if unsupported.any():
        values = sorted(perturbation.loc[unsupported].dropna().unique().tolist())
        raise ValueError(
            "Unsupported perturbation values after normalization: "
            f"{values}. Supported OE aliases: {sorted(OE_ALIASES)}; "
            f"loss-of-function aliases: {sorted(LOSS_ALIASES)}"
        )

    work["expected_effect"] = np.where(oe_mask, -work["logFC"], work["logFC"])
    usable = work["logFC"].notna()

    if fdr_threshold is not None:
        fdr_column = "benchmark_FDR" if "benchmark_FDR" in work.columns else "FDR"
        if fdr_column not in work.columns:
            raise ValueError(
                "An FDR threshold was requested, but neither benchmark_FDR nor FDR is available."
            )
        work[fdr_column] = pd.to_numeric(work[fdr_column], errors="coerce")
        usable &= work[fdr_column].between(0.0, 1.0, inclusive="both")
        significant = work[fdr_column] < float(fdr_threshold)
    else:
        significant = pd.Series(True, index=work.index)

    work = work.loc[usable].copy()
    significant = significant.loc[work.index]
    work["is_positive"] = (
        (work["expected_effect"] > float(effect_threshold)) & significant
    ).astype(bool)
    work["is_background"] = ~work["is_positive"]
    return work


def prepare_inputs(
    *,
    run_dir: Path,
    metadata_path: Path,
    fdr_threshold: float | None,
    effect_threshold: float,
) -> Figure2Inputs:
    datasets = load_joined_datasets(run_dir)
    metadata_order, labels = load_tool_metadata(metadata_path)
    tool_ids = available_tool_ids(datasets, metadata_order)

    annotated = tuple(
        BenchmarkDataset(
            dataset_id=dataset.dataset_id,
            frame=annotate_ground_truth(
                dataset.frame,
                fdr_threshold=fdr_threshold,
                effect_threshold=effect_threshold,
            ),
        )
        for dataset in datasets
    )
    resolved_labels = {
        tool_id: labels.get(tool_id, tool_id.replace("_", " "))
        for tool_id in tool_ids
    }
    return Figure2Inputs(
        run_dir=run_dir,
        datasets=annotated,
        tool_ids=tool_ids,
        tool_labels=resolved_labels,
        tool_colors=evaluation_tool_colors(tool_ids),
        fdr_threshold=fdr_threshold,
        effect_threshold=effect_threshold,
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def save_panel(
    fig: plt.Figure,
    *,
    out_dir: Path,
    stem: str,
    dpi: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in DEFAULT_FORMATS:
        fig.savefig(
            out_dir / f"{stem}.{extension}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def load_experiment_metric_table(run_dir: Path) -> pd.DataFrame | None:
    """Load the per-experiment coverage table used by the evaluation report.

    Missing values identify experiments that were not evaluable for a predictor,
    which is the definition used for panel A. This preserves the 31/29/27 counts
    reported in the manuscript draft instead of counting any raw score presence.
    """

    path = run_dir / "tables" / "per_experiment" / "coverage_per_experiment.tsv"
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t", low_memory=False)


def panel_a_experiment_coverage(inputs: Figure2Inputs) -> tuple[pd.DataFrame, plt.Figure]:
    metric_table = load_experiment_metric_table(inputs.run_dir)
    rows = []
    total_experiments = len(inputs.datasets)

    for tool_id in inputs.tool_ids:
        if metric_table is not None and tool_id in metric_table.columns:
            retained_mask = metric_table[tool_id].notna()
            retained = int(retained_mask.sum())
            excluded_datasets = (
                metric_table.loc[~retained_mask, "dataset_id"].astype(str).tolist()
                if "dataset_id" in metric_table.columns
                else []
            )
        else:
            retained = sum(
                bool(dataset.frame[score_column(tool_id)].notna().any())
                for dataset in inputs.datasets
            )
            excluded_datasets = []

        rows.append(
            {
                "tool_id": tool_id,
                "predictor": inputs.tool_labels[tool_id],
                "experiments_retained": retained,
                "experiments_total": total_experiments,
                "experiments_excluded": total_experiments - retained,
                "experiment_coverage": (
                    retained / total_experiments if total_experiments else np.nan
                ),
                "excluded_dataset_ids": ";".join(excluded_datasets),
            }
        )

    summary = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(summary))
    colors = [inputs.tool_colors[tool_id] for tool_id in summary["tool_id"]]
    bars = ax.bar(
        x,
        summary["experiments_retained"].to_numpy(),
        color=colors,
    )
    ax.axhline(total_experiments, linestyle="--", linewidth=1.2, color="black")
    ax.set_xticks(x, summary["predictor"], rotation=25, ha="right")
    ax.set_ylabel("Experiments retained")
    ax.set_ylim(0, max(total_experiments * 1.16, 1))
    ax.set_title("A  Experiment coverage")
    for bar, value, percentage in zip(
        bars,
        summary["experiments_retained"],
        summary["experiment_coverage"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(value)}/{total_experiments}\n({percentage:.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    style_axis(ax)
    fig.tight_layout()
    return summary, fig


def panel_b_gene_set_coverage(inputs: Figure2Inputs) -> tuple[pd.DataFrame, plt.Figure]:
    all_pairs = pd.concat(
        [
            dataset.frame.assign(dataset_id=dataset.dataset_id)
            for dataset in inputs.datasets
        ],
        ignore_index=True,
    )
    fgs_genes = set(all_pairs["gene_id"].dropna().astype(str))
    scored_gene_sets = {
        tool_id: set(
            all_pairs.loc[all_pairs[score_column(tool_id)].notna(), "gene_id"]
            .dropna()
            .astype(str)
        )
        for tool_id in inputs.tool_ids
    }
    igs_genes = (
        set.intersection(*scored_gene_sets.values()) if scored_gene_sets else set()
    )

    rows = []
    denominator = len(fgs_genes)
    for tool_id in inputs.tool_ids:
        count = len(scored_gene_sets[tool_id])
        rows.append(
            {
                "set_id": tool_id,
                "label": inputs.tool_labels[tool_id],
                "gene_count": count,
                "fgs_gene_count": denominator,
                "fgs_coverage": count / denominator if denominator else np.nan,
                "set_type": "predictor",
            }
        )
    rows.extend(
        [
            {
                "set_id": "IGS",
                "label": "IGS",
                "gene_count": len(igs_genes),
                "fgs_gene_count": denominator,
                "fgs_coverage": len(igs_genes) / denominator if denominator else np.nan,
                "set_type": "intersection",
            },
            {
                "set_id": "FGS",
                "label": "FGS",
                "gene_count": denominator,
                "fgs_gene_count": denominator,
                "fgs_coverage": 1.0 if denominator else np.nan,
                "set_type": "full",
            },
        ]
    )
    summary = pd.DataFrame(rows)

    colors = [
        inputs.tool_colors[set_id]
        if set_id in inputs.tool_colors
        else IGS_COLOR if set_id == "IGS" else FGS_COLOR
        for set_id in summary["set_id"]
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(summary))
    bars = ax.bar(
        x,
        100.0 * summary["fgs_coverage"].to_numpy(),
        color=colors,
    )
    ax.set_xticks(x, summary["label"], rotation=25, ha="right")
    ax.set_ylabel("Unique gene coverage of FGS (%)")
    ax.set_ylim(0, 110)
    ax.set_title("B  Gene-set coverage and common intersection")
    for bar, count, percentage in zip(
        bars, summary["gene_count"], summary["fgs_coverage"]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count):,}\n({percentage:.1%})",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )
    style_axis(ax)
    fig.tight_layout()
    return summary, fig


def pooled_pair_coverage(
    inputs: Figure2Inputs,
    *,
    subset_column: str,
    subset_label: str,
) -> pd.DataFrame:
    rows = []
    for tool_id in inputs.tool_ids:
        denominator = 0
        numerator = 0
        for dataset in inputs.datasets:
            subset = dataset.frame[subset_column].astype(bool)
            denominator += int(subset.sum())
            numerator += int(
                dataset.frame.loc[subset, score_column(tool_id)].notna().sum()
            )
        rows.append(
            {
                "tool_id": tool_id,
                "predictor": inputs.tool_labels[tool_id],
                f"{subset_label}_pairs_scored": numerator,
                f"{subset_label}_pairs_total": denominator,
                f"{subset_label}_coverage": (
                    numerator / denominator if denominator else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def pair_coverage_figure(
    summary: pd.DataFrame,
    *,
    inputs: Figure2Inputs,
    coverage_column: str,
    numerator_column: str,
    denominator_column: str,
    title: str,
    ylabel: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(summary))
    percentages = 100.0 * summary[coverage_column].to_numpy()
    colors = [inputs.tool_colors[tool_id] for tool_id in summary["tool_id"]]
    bars = ax.bar(x, percentages, color=colors)
    ax.set_xticks(x, summary["predictor"], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 110)
    ax.set_title(title)
    for bar, numerator, denominator, percentage in zip(
        bars,
        summary[numerator_column],
        summary[denominator_column],
        summary[coverage_column],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{percentage:.1%}\n{int(numerator):,}/{int(denominator):,}",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )
    style_axis(ax)
    fig.tight_layout()
    return fig


def panel_c_positive_coverage(inputs: Figure2Inputs) -> tuple[pd.DataFrame, plt.Figure]:
    summary = pooled_pair_coverage(
        inputs,
        subset_column="is_positive",
        subset_label="positive",
    )
    fig = pair_coverage_figure(
        summary,
        inputs=inputs,
        coverage_column="positive_coverage",
        numerator_column="positive_pairs_scored",
        denominator_column="positive_pairs_total",
        title="C  Perturbation-consistent positive coverage",
        ylabel="Positive pair coverage (%)",
    )
    return summary, fig


def panel_d_background_coverage(inputs: Figure2Inputs) -> tuple[pd.DataFrame, plt.Figure]:
    summary = pooled_pair_coverage(
        inputs,
        subset_column="is_background",
        subset_label="background",
    )
    fig = pair_coverage_figure(
        summary,
        inputs=inputs,
        coverage_column="background_coverage",
        numerator_column="background_pairs_scored",
        denominator_column="background_pairs_total",
        title="D  Background/non-positive coverage",
        ylabel="Background pair coverage (%)",
    )
    return summary, fig


def write_panel(
    panel: str,
    *,
    inputs: Figure2Inputs,
    out_dir: Path,
    dpi: int,
) -> None:
    builders = {
        "A": panel_a_experiment_coverage,
        "B": panel_b_gene_set_coverage,
        "C": panel_c_positive_coverage,
        "D": panel_d_background_coverage,
    }
    summary, fig = builders[panel](inputs)
    stem = {
        "A": "figure2A_experiment_coverage",
        "B": "figure2B_gene_set_coverage",
        "C": "figure2C_positive_coverage",
        "D": "figure2D_background_coverage",
    }[panel]
    panel_dir = out_dir / f"panel_{panel}"
    panel_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(panel_dir / f"{stem}.tsv", sep="\t", index=False)
    save_panel(fig, out_dir=panel_dir, stem=stem, dpi=dpi)
    print(f"Wrote panel {panel}: {panel_dir / stem}")


def main() -> None:
    args = parse_args()
    run_dir = (args.run_dir or find_latest_completed_run(args.results_dir)).resolve()
    metadata_path = args.metadata.resolve()
    out_dir = (args.out_dir or DEFAULT_OUTPUT_ROOT / run_dir.name).resolve()

    inputs = prepare_inputs(
        run_dir=run_dir,
        metadata_path=metadata_path,
        fdr_threshold=RUN_FDR_THRESHOLD,
        effect_threshold=RUN_EFFECT_THRESHOLD,
    )

    panels = ("A", "B", "C", "D") if args.panel == "all" else (args.panel,)
    print(f"Run directory: {run_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Datasets: {len(inputs.datasets)}")
    print(f"Predictors: {', '.join(inputs.tool_ids)}")
    print("Ground-truth filters: inherited from the completed benchmark run")
    for panel in panels:
        write_panel(
            panel,
            inputs=inputs,
            out_dir=out_dir,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
