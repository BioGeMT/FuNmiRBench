#!/usr/bin/env python3
"""Generate the coverage panels for manuscript Figure 2.

The script reads the ``joined.tsv`` files from one completed FuNmiRBench run and
writes each panel as a separate publication-ready plot plus the exact summary
values used to draw it. Stable manuscript assets are written to
``manuscript_assets/figure2``.

Panel definitions
-----------------
A. Number of benchmark experiments retained for each predictor.
B. Predictor gene-set coverage across the full benchmark gene universe, plus
   the all-algorithm Intersection Gene Set (IGS).
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
    manuscript_assets/figure2/
"""

from __future__ import annotations

import argparse
import shutil
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
DEFAULT_MANUSCRIPT_OUTPUT_DIR = Path("manuscript_assets/figure2")
DEFAULT_FORMATS = ("png", "svg")
RUN_EFFECT_THRESHOLD = 1.0
RUN_FDR_THRESHOLD = 0.05
PANEL_CHOICES = ("A", "B", "C", "D", "all")
INTERSECTION_COLORS = {
    "all_algorithm_intersection": "#7A8798",
}
PREDICTOR_BAR_ALPHA = 0.35
IGS_BAR_ALPHA = 0.55
BAR_LABEL_FONTSIZE = 12

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
        "--manuscript-out-dir",
        type=Path,
        default=DEFAULT_MANUSCRIPT_OUTPUT_DIR,
        help="Stable manuscript output directory. Default: manuscript_assets/figure2/.",
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


def set_panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(
        -0.12,
        1.07,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_title(title, pad=18, fontsize=14, fontweight="bold")


def save_panel(
    fig: plt.Figure,
    *,
    out_dir: Path,
    stem: str,
    dpi: int,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in DEFAULT_FORMATS:
        path = out_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


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
        edgecolor=colors,
        alpha=PREDICTOR_BAR_ALPHA,
        linewidth=1.0,
    )
    ax.axhline(total_experiments, linestyle="--", linewidth=1.2, color="black")
    ax.set_xticks(x, summary["predictor"], rotation=25, ha="right")
    ax.set_ylabel("Experiments retained")
    ax.set_ylim(0, max(total_experiments * 1.16, 1))
    set_panel_title(ax, "A", "Experiment coverage")
    for bar, retained in zip(bars, summary["experiments_retained"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(retained)}/{total_experiments}",
            ha="center",
            va="bottom",
            fontsize=BAR_LABEL_FONTSIZE,
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

    available_sets = [scored_gene_sets[tool_id] for tool_id in inputs.tool_ids]
    igs_genes = set.intersection(*available_sets) if available_sets else set()
    rows.append(
        {
            "set_id": "all_algorithm_intersection",
            "label": "IGS\n(all algorithms)",
            "gene_count": len(igs_genes),
            "fgs_gene_count": denominator,
            "fgs_coverage": len(igs_genes) / denominator if denominator else np.nan,
            "set_type": "intersection",
        }
    )
    summary = pd.DataFrame(rows)

    colors = [
        inputs.tool_colors[set_id]
        if set_id in inputs.tool_colors
        else INTERSECTION_COLORS[set_id]
        for set_id in summary["set_id"]
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(summary))
    alphas = [
        PREDICTOR_BAR_ALPHA if set_type == "predictor" else IGS_BAR_ALPHA
        for set_type in summary["set_type"]
    ]
    bars = ax.bar(
        x,
        100.0 * summary["fgs_coverage"].to_numpy(),
        color=colors,
        edgecolor=colors,
        linewidth=1.0,
    )
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
    ax.set_xticks(x, summary["label"], rotation=25, ha="right")
    ax.set_ylabel("Genes covered (%)")
    ax.set_ylim(0, 110)
    set_panel_title(ax, "B", "Gene-set coverage and IGS")
    ax.text(
        0.01,
        0.96,
        f"FGS: {denominator:,} genes",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )
    for bar, percentage in zip(bars, summary["fgs_coverage"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{percentage:.1%}",
            ha="center",
            va="bottom",
            fontsize=BAR_LABEL_FONTSIZE,
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
    panel: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(summary))
    percentages = 100.0 * summary[coverage_column].to_numpy()
    colors = [inputs.tool_colors[tool_id] for tool_id in summary["tool_id"]]
    bars = ax.bar(
        x,
        percentages,
        color=colors,
        edgecolor=colors,
        alpha=PREDICTOR_BAR_ALPHA,
        linewidth=1.0,
    )
    ax.set_xticks(x, summary["predictor"], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 110)
    set_panel_title(ax, panel, title)
    for bar, percentage in zip(bars, summary[coverage_column]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{percentage:.1%}",
            ha="center",
            va="bottom",
            fontsize=BAR_LABEL_FONTSIZE,
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
        title="Positive miRNA-gene pair coverage",
        ylabel="Positive miRNA-gene pairs covered (%)",
        panel="C",
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
        title="Background miRNA-gene pair coverage",
        ylabel="Background miRNA-gene pairs covered (%)",
        panel="D",
    )
    return summary, fig


def predictor_mean_mirnas_per_scored_gene(inputs: Figure2Inputs) -> pd.DataFrame:
    all_pairs = pd.concat(
        [
            dataset.frame.assign(dataset_id=dataset.dataset_id)
            for dataset in inputs.datasets
        ],
        ignore_index=True,
    )
    rows = []
    for tool_id in inputs.tool_ids:
        scored = (
            all_pairs.loc[
                all_pairs[score_column(tool_id)].notna(),
                ["gene_id", "mirna"],
            ]
            .dropna()
            .copy()
        )
        total_pairs = int(scored.shape[0])
        per_gene = (
            scored.drop_duplicates(["gene_id", "mirna"])
            .groupby("gene_id")["mirna"]
            .nunique()
        )
        rows.append(
            {
                "tool_id": tool_id,
                "total_scored_pairs": total_pairs,
                "mean_mirnas_per_scored_gene": (
                    float(per_gene.mean()) if len(per_gene) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def write_supplementary_coverage_table(
    *,
    inputs: Figure2Inputs,
    manuscript_out_dir: Path,
) -> Path:
    experiment, experiment_fig = panel_a_experiment_coverage(inputs)
    plt.close(experiment_fig)
    experiment = experiment.rename(
        columns={
            "experiments_retained": "panel_a_experiments_retained",
            "experiments_total": "panel_a_experiments_total",
            "experiments_excluded": "panel_a_experiments_excluded",
            "experiment_coverage": "panel_a_experiment_coverage",
        }
    )
    gene, gene_fig = panel_b_gene_set_coverage(inputs)
    plt.close(gene_fig)
    gene = gene.rename(
        columns={
            "set_id": "tool_id",
            "label": "predictor",
            "gene_count": "panel_b_gene_count",
            "fgs_gene_count": "panel_b_fgs_gene_count",
            "fgs_coverage": "panel_b_fgs_coverage",
        }
    )
    positive, positive_fig = panel_c_positive_coverage(inputs)
    plt.close(positive_fig)
    positive = positive.rename(
        columns={
            "positive_pairs_scored": "panel_c_positive_pairs_scored",
            "positive_pairs_total": "panel_c_positive_pairs_total",
            "positive_coverage": "panel_c_positive_pair_coverage",
        }
    )
    background, background_fig = panel_d_background_coverage(inputs)
    plt.close(background_fig)
    background = background.rename(
        columns={
            "background_pairs_scored": "panel_d_background_pairs_scored",
            "background_pairs_total": "panel_d_background_pairs_total",
            "background_coverage": "panel_d_background_pair_coverage",
        }
    )
    mirnas_per_gene = predictor_mean_mirnas_per_scored_gene(inputs)

    for frame in (experiment, positive, background):
        if "predictor" in frame.columns:
            frame.drop(columns=["predictor"], inplace=True)

    table = gene.merge(experiment, on="tool_id", how="left")
    table = table.merge(positive, on="tool_id", how="left")
    table = table.merge(background, on="tool_id", how="left")
    table = table.merge(mirnas_per_gene, on="tool_id", how="left")
    table.insert(
        1,
        "name",
        table["tool_id"].map(inputs.tool_labels).fillna(table["predictor"]),
    )
    table.loc[
        table["tool_id"] == "all_algorithm_intersection",
        "name",
    ] = "IGS (all algorithms)"

    columns = [
        "name",
        "set_type",
        "panel_a_experiments_retained",
        "panel_a_experiments_total",
        "panel_a_experiments_excluded",
        "panel_a_experiment_coverage",
        "panel_b_gene_count",
        "panel_b_fgs_gene_count",
        "panel_b_fgs_coverage",
        "panel_c_positive_pairs_scored",
        "panel_c_positive_pairs_total",
        "panel_c_positive_pair_coverage",
        "panel_d_background_pairs_scored",
        "panel_d_background_pairs_total",
        "panel_d_background_pair_coverage",
        "total_scored_pairs",
        "mean_mirnas_per_scored_gene",
    ]
    table = table[columns]
    path = manuscript_out_dir / "figure2_supplementary_coverage_table.tsv"
    manuscript_out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)
    print(f"Wrote supplementary Figure 2 table: {path}")
    return path


def write_panel(
    panel: str,
    *,
    inputs: Figure2Inputs,
    out_dir: Path,
    manuscript_out_dir: Path,
    dpi: int,
) -> dict[str, Path]:
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
    table_path = panel_dir / f"{stem}.tsv"
    summary.to_csv(table_path, sep="\t", index=False)
    figure_paths = save_panel(fig, out_dir=panel_dir, stem=stem, dpi=dpi)
    manuscript_out_dir.mkdir(parents=True, exist_ok=True)
    manuscript_table = manuscript_out_dir / f"{stem}.tsv"
    shutil.copy2(table_path, manuscript_table)
    manuscript_figures = {}
    for figure_path in figure_paths:
        manuscript_path = manuscript_out_dir / figure_path.name
        shutil.copy2(figure_path, manuscript_path)
        manuscript_figures[figure_path.suffix.lstrip(".")] = manuscript_path
    print(f"Wrote panel {panel}: {panel_dir / stem}")
    return {
        "table": manuscript_table,
        **manuscript_figures,
    }


def trim_white_border(image: np.ndarray, *, threshold: float = 0.985, pad: int = 16) -> np.ndarray:
    rgb = image[..., :3]
    nonwhite = np.any(rgb < threshold, axis=2)
    if not np.any(nonwhite):
        return image
    rows = np.where(nonwhite.any(axis=1))[0]
    cols = np.where(nonwhite.any(axis=0))[0]
    top = max(int(rows[0]) - pad, 0)
    bottom = min(int(rows[-1]) + pad + 1, image.shape[0])
    left = max(int(cols[0]) - pad, 0)
    right = min(int(cols[-1]) + pad + 1, image.shape[1])
    return image[top:bottom, left:right]


def pad_to_shape(
    image: np.ndarray,
    shape: tuple[int, int],
    *,
    vertical: str = "center",
    horizontal: str = "center",
) -> np.ndarray:
    target_height, target_width = shape
    height, width = image.shape[:2]
    if height > target_height or width > target_width:
        raise ValueError("Target shape must be at least as large as the image.")

    channels = image.shape[2] if image.ndim == 3 else 1
    canvas_shape = (target_height, target_width, channels)
    canvas = np.ones(canvas_shape, dtype=image.dtype)
    if vertical == "top":
        top = 0
    elif vertical == "bottom":
        top = target_height - height
    else:
        top = (target_height - height) // 2
    if horizontal == "left":
        left = 0
    elif horizontal == "right":
        left = target_width - width
    else:
        left = (target_width - width) // 2
    canvas[top : top + height, left : left + width, ...] = image
    return canvas


def write_combined_figure(panel_outputs: dict[str, dict[str, Path]], *, out_dir: Path, dpi: int) -> None:
    required = ("A", "B", "C", "D")
    missing = [panel for panel in required if panel not in panel_outputs or "png" not in panel_outputs[panel]]
    if missing:
        return

    panel_images = {
        panel: trim_white_border(plt.imread(panel_outputs[panel]["png"]))
        for panel in required
    }
    row_heights = [
        max(panel_images[panel].shape[0] for panel in ("A", "B")),
        max(panel_images[panel].shape[0] for panel in ("C", "D")),
    ]
    column_widths = [
        max(panel_images[panel].shape[1] for panel in ("A", "C")),
        max(panel_images[panel].shape[1] for panel in ("B", "D")),
    ]
    cells = {
        "A": pad_to_shape(panel_images["A"], (row_heights[0], column_widths[0]), vertical="top"),
        "B": pad_to_shape(panel_images["B"], (row_heights[0], column_widths[1]), vertical="top"),
        "C": pad_to_shape(panel_images["C"], (row_heights[1], column_widths[0]), vertical="top"),
        "D": pad_to_shape(panel_images["D"], (row_heights[1], column_widths[1]), vertical="top"),
    }

    gap_x = 90
    gap_y = 20
    channels = next(iter(cells.values())).shape[2]
    dtype = next(iter(cells.values())).dtype
    top_row = np.hstack(
        [
            cells["A"],
            np.ones((row_heights[0], gap_x, channels), dtype=dtype),
            cells["B"],
        ]
    )
    bottom_row = np.hstack(
        [
            cells["C"],
            np.ones((row_heights[1], gap_x, channels), dtype=dtype),
            cells["D"],
        ]
    )
    combined = np.vstack(
        [
            top_row,
            np.ones((gap_y, top_row.shape[1], channels), dtype=dtype),
            bottom_row,
        ]
    )

    fig_width = 16.2
    fig_height = fig_width * combined.shape[0] / combined.shape[1]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(combined)
    ax.axis("off")
    fig.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in DEFAULT_FORMATS:
        fig.savefig(out_dir / f"figure2_combined.{extension}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote combined Figure 2: {out_dir / 'figure2_combined'}")


def main() -> None:
    args = parse_args()
    run_dir = (args.run_dir or find_latest_completed_run(args.results_dir)).resolve()
    metadata_path = args.metadata.resolve()
    out_dir = (args.out_dir or DEFAULT_OUTPUT_ROOT / run_dir.name).resolve()
    manuscript_out_dir = args.manuscript_out_dir.resolve()

    inputs = prepare_inputs(
        run_dir=run_dir,
        metadata_path=metadata_path,
        fdr_threshold=RUN_FDR_THRESHOLD,
        effect_threshold=RUN_EFFECT_THRESHOLD,
    )

    panels = ("A", "B", "C", "D") if args.panel == "all" else (args.panel,)
    print(f"Run directory: {run_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Manuscript output directory: {manuscript_out_dir}")
    print(f"Datasets: {len(inputs.datasets)}")
    print(f"Predictors: {', '.join(inputs.tool_ids)}")
    print("Ground-truth filters: inherited from the completed benchmark run")
    panel_outputs = {}
    for panel in panels:
        panel_outputs[panel] = write_panel(
            panel,
            inputs=inputs,
            out_dir=out_dir,
            manuscript_out_dir=manuscript_out_dir,
            dpi=args.dpi,
        )
    if tuple(panels) == ("A", "B", "C", "D"):
        write_combined_figure(panel_outputs, out_dir=manuscript_out_dir, dpi=args.dpi)
        write_supplementary_coverage_table(
            inputs=inputs,
            manuscript_out_dir=manuscript_out_dir,
        )


if __name__ == "__main__":
    main()
