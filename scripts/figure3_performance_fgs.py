#!/usr/bin/env python3
"""Generate Figure 3 Full Gene Set performance inputs and draft panels.

Figure 3 evaluates predictors on the Full Gene Set (FGS). For each dataset and
predictor, scored miRNA-gene pairs are converted to dataset-local normalized
ranks where 0 is the weakest scored pair and 1 is the strongest scored pair.
Tied scores receive average ranks. Unscored pairs remain in the FGS analysis and
are assigned rank 0, which penalizes predictors for gene-selection coverage.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score

from funmirbench.evaluate_common import CURVE_COLORS, _rank_scale_scores
from funmirbench.logger import setup_logging


logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_METADATA_PATH = Path("metadata/predictions_info.tsv")
DEFAULT_MANUSCRIPT_OUTPUT_DIR = Path("manuscript_assets/figure3")
DEFAULT_MANUSCRIPT_TABLES_DIR = Path("manuscript_assets/tables")
DEFAULT_FORMATS = ("png", "svg")
METRICS = ("aps", "pr_auc", "auroc", "spearman")
PLOT_METRICS = ("coverage", "positive_coverage", *METRICS)
METRIC_LABELS = {
    "coverage": "Coverage",
    "positive_coverage": "Positive coverage",
    "aps": "Average precision",
    "pr_auc": "PR-AUC",
    "auroc": "AUROC",
    "spearman": "Spearman rho",
}
OE_ALIASES = {"OE", "OVEREXPRESSION", "OVER_EXPRESSION"}
LOSS_ALIASES = {"KO", "KD", "KNOCKOUT", "KNOCK_OUT", "KNOCKDOWN", "KNOCK_DOWN"}


@dataclass(frozen=True)
class BenchmarkDataset:
    dataset_id: str
    frame: pd.DataFrame


@dataclass(frozen=True)
class Figure3Inputs:
    run_dir: Path
    datasets: tuple[BenchmarkDataset, ...]
    tool_ids: tuple[str, ...]
    tool_labels: dict[str, str]
    tool_colors: dict[str, str]
    fdr_threshold: float | None
    effect_threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Figure 3 FGS performance tables and draft panels."
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
        "--manuscript-out-dir",
        type=Path,
        default=DEFAULT_MANUSCRIPT_OUTPUT_DIR,
        help="Stable manuscript figure output directory. Default: manuscript_assets/figure3/.",
    )
    parser.add_argument(
        "--manuscript-tables-dir",
        type=Path,
        default=DEFAULT_MANUSCRIPT_TABLES_DIR,
        help="Stable manuscript table output directory. Default: manuscript_assets/tables/.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster output resolution.")
    parser.add_argument("--log-level", default="INFO", help="Logging level. Default: INFO.")
    return parser.parse_args()


def find_latest_completed_run(results_dir: Path) -> Path:
    candidates = [
        child
        for child in results_dir.iterdir()
        if child.is_dir() and list(child.glob("datasets/*/joined.tsv"))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No completed benchmark run with datasets/*/joined.tsv found under {results_dir}."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_run_thresholds(run_dir: Path) -> tuple[float | None, float]:
    config_path = run_dir / "benchmark_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Completed run is missing config snapshot: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    evaluation = config.get("evaluation") or {}
    missing = [
        key for key in ("fdr_threshold", "effect_threshold") if key not in evaluation
    ]
    if missing:
        raise KeyError(f"{config_path} is missing evaluation keys: {missing}")
    fdr = evaluation["fdr_threshold"]
    return (None if fdr is None else float(fdr), float(evaluation["effect_threshold"]))


def load_tool_metadata(path: Path) -> tuple[tuple[str, ...], dict[str, str]]:
    metadata = pd.read_csv(path, sep="\t", dtype=str)
    required = {"tool_id", "official_name"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    metadata = metadata.dropna(subset=["tool_id"]).copy()
    metadata["tool_id"] = metadata["tool_id"].astype(str).str.strip()
    metadata["official_name"] = metadata["official_name"].fillna(metadata["tool_id"])
    metadata["official_name"] = metadata["official_name"].astype(str).str.strip()
    return tuple(metadata["tool_id"].tolist()), dict(
        zip(metadata["tool_id"], metadata["official_name"])
    )


def load_joined_datasets(run_dir: Path) -> tuple[BenchmarkDataset, ...]:
    paths = sorted(run_dir.glob("datasets/*/joined.tsv"))
    if not paths:
        raise FileNotFoundError(f"No datasets/*/joined.tsv files found in {run_dir}")
    datasets = []
    for path in paths:
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
    common_score_columns: set[str] | None = None
    for dataset in datasets:
        current = {column for column in dataset.frame.columns if column.startswith("score_")}
        common_score_columns = (
            current if common_score_columns is None else common_score_columns & current
        )
    common_tool_ids = {
        column.removeprefix("score_") for column in (common_score_columns or set())
    }
    ordered = tuple(tool_id for tool_id in metadata_order if tool_id in common_tool_ids)
    resolved = ordered + tuple(sorted(common_tool_ids.difference(ordered)))
    if not resolved:
        raise ValueError("No score_<tool_id> columns are shared by all joined datasets.")
    return resolved


def evaluation_tool_colors(tool_ids: Iterable[str]) -> dict[str, str]:
    evaluation_order = sorted(str(tool_id) for tool_id in tool_ids)
    return {
        tool_id: CURVE_COLORS[index % len(CURVE_COLORS)]
        for index, tool_id in enumerate(evaluation_order)
    }


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
    perturbation = (
        work["perturbation"]
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"[\s-]+", "_", regex=True)
    )
    oe_mask = perturbation.isin(OE_ALIASES)
    loss_mask = perturbation.isin(LOSS_ALIASES)
    unsupported = ~(oe_mask | loss_mask)
    if unsupported.any():
        values = sorted(perturbation.loc[unsupported].dropna().unique().tolist())
        raise ValueError(f"Unsupported perturbation values: {values}")

    work["expected_effect"] = np.where(oe_mask, -work["logFC"], work["logFC"])
    usable = work["logFC"].notna()
    if fdr_threshold is not None:
        fdr_column = "benchmark_FDR" if "benchmark_FDR" in work.columns else "FDR"
        if fdr_column not in work.columns:
            raise ValueError("FDR threshold requires benchmark_FDR or FDR.")
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
    return work


def prepare_inputs(
    *,
    run_dir: Path,
    metadata_path: Path,
    fdr_threshold: float | None,
    effect_threshold: float,
) -> Figure3Inputs:
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
    return Figure3Inputs(
        run_dir=run_dir,
        datasets=annotated,
        tool_ids=tool_ids,
        tool_labels={tool_id: labels.get(tool_id, tool_id) for tool_id in tool_ids},
        tool_colors=evaluation_tool_colors(tool_ids),
        fdr_threshold=fdr_threshold,
        effect_threshold=effect_threshold,
    )


def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def compute_metrics(y_true: pd.Series, y_score: pd.Series, expected_effect: pd.Series) -> dict[str, float]:
    y_true_array = y_true.astype(int).to_numpy()
    y_score_array = y_score.astype(float).to_numpy()
    positives = int(y_true_array.sum())
    negatives = int(len(y_true_array) - positives)
    if positives == 0 or negatives == 0:
        return {metric: float("nan") for metric in METRICS}
    spearman = spearmanr(y_score_array, expected_effect.astype(float).to_numpy()).correlation
    return {
        "aps": float(average_precision_score(y_true_array, y_score_array)),
        "pr_auc": pr_auc_score(y_true_array, y_score_array),
        "auroc": float(roc_auc_score(y_true_array, y_score_array)),
        "spearman": float(spearman) if pd.notna(spearman) else float("nan"),
    }


def fgs_rank_series(frame: pd.DataFrame, tool_id: str) -> pd.Series:
    ranks = _rank_scale_scores(frame[score_column(tool_id)])
    return ranks.fillna(0.0).astype(float)


def compute_fgs_per_experiment_metrics(inputs: Figure3Inputs) -> pd.DataFrame:
    rows = []
    for dataset in inputs.datasets:
        frame = dataset.frame
        for tool_id in inputs.tool_ids:
            score_col = score_column(tool_id)
            if score_col not in frame.columns:
                continue
            fgs_rank = fgs_rank_series(frame, tool_id)
            metrics = compute_metrics(
                frame["is_positive"],
                fgs_rank,
                frame["expected_effect"],
            )
            scored = int(frame[score_col].notna().sum())
            total = int(len(frame))
            positives_total = int(frame["is_positive"].sum())
            positives_scored = int(frame.loc[frame[score_col].notna(), "is_positive"].sum())
            rows.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "tool_id": tool_id,
                    "predictor": inputs.tool_labels[tool_id],
                    "rows_total": total,
                    "rows_scored": scored,
                    "rows_missing_score": total - scored,
                    "coverage": scored / total if total else np.nan,
                    "positives_total": positives_total,
                    "positives_scored": positives_scored,
                    "positive_coverage": (
                        positives_scored / positives_total if positives_total else np.nan
                    ),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def compute_leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby(["tool_id", "predictor"], as_index=False)
    summary = grouped.agg(
        datasets=("dataset_id", "nunique"),
        mean_aps=("aps", "mean"),
        median_aps=("aps", "median"),
        mean_pr_auc=("pr_auc", "mean"),
        mean_auroc=("auroc", "mean"),
        mean_spearman=("spearman", "mean"),
        mean_coverage=("coverage", "mean"),
        mean_positive_coverage=("positive_coverage", "mean"),
    )
    summary = summary.sort_values(
        ["mean_aps", "mean_pr_auc", "mean_auroc"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    summary.insert(0, "fgs_rank", np.arange(1, len(summary) + 1))
    return summary


def write_rank_table(inputs: Figure3Inputs, out_path: Path) -> None:
    chunks = []
    for dataset in inputs.datasets:
        frame = dataset.frame[["dataset_id", "mirna", "perturbation", "gene_id", "is_positive"]].copy()
        for tool_id in inputs.tool_ids:
            frame[f"fgs_rank_{tool_id}"] = fgs_rank_series(dataset.frame, tool_id)
        chunks.append(frame)
    out = pd.concat(chunks, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, *, dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in DEFAULT_FORMATS:
        path = out_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_metric_distributions(inputs: Figure3Inputs, metrics: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
    positions = np.arange(1, len(inputs.tool_ids) + 1)
    for ax, metric, letter in zip(axes.ravel(), PLOT_METRICS, ("A", "B", "C", "D", "E", "F")):
        data = [
            metrics.loc[metrics["tool_id"] == tool_id, metric].dropna().astype(float).to_numpy()
            for tool_id in inputs.tool_ids
        ]
        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.56,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#2D3748", "linewidth": 1.3},
            whiskerprops={"color": "#6B7280", "linewidth": 1.0},
            capprops={"color": "#6B7280", "linewidth": 1.0},
            boxprops={"edgecolor": "#6B7280", "linewidth": 1.0},
        )
        for patch, tool_id in zip(box["boxes"], inputs.tool_ids):
            color = inputs.tool_colors[tool_id]
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.28)

        for index, (tool_id, values) in enumerate(zip(inputs.tool_ids, data), start=1):
            if values.size == 0:
                continue
            jitter = np.linspace(-0.09, 0.09, values.size) if values.size > 1 else np.array([0.0])
            ax.scatter(
                np.full(values.size, index, dtype=float) + jitter,
                values,
                s=30,
                color=inputs.tool_colors[tool_id],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.75,
                zorder=3,
            )
        ax.set_title(f"{letter}. {METRIC_LABELS[metric]}", loc="left", fontweight="bold")
        ax.set_xticks(
            positions,
            [inputs.tool_labels[tool_id] for tool_id in inputs.tool_ids],
            rotation=25,
            ha="right",
        )
        ax.set_ylabel(METRIC_LABELS[metric])
        if metric != "spearman":
            ax.set_ylim(0.0, 1.02)
        else:
            ax.set_ylim(-1.02, 1.02)
        style_axis(ax)
    fig.suptitle("Cross-dataset FGS predictor distributions", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout()
    return fig


def plot_leaderboard(inputs: Figure3Inputs, leaderboard: pd.DataFrame) -> plt.Figure:
    ordered = leaderboard.sort_values("mean_aps", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = [inputs.tool_colors[tool_id] for tool_id in ordered["tool_id"]]
    ax.barh(ordered["predictor"], ordered["mean_aps"], color=colors, alpha=0.45)
    ax.set_xlabel("Mean FGS average precision")
    ax.set_xlim(0.0, max(ordered["mean_aps"].max() * 1.18, 0.05))
    ax.set_title("FGS leaderboard", loc="left", fontweight="bold")
    for y, value in enumerate(ordered["mean_aps"]):
        ax.text(value, y, f" {value:.3f}", va="center", fontsize=10)
    style_axis(ax)
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    run_dir = args.run_dir or find_latest_completed_run(args.results_dir)
    fdr_threshold, effect_threshold = load_run_thresholds(run_dir)
    logger.info("Using run directory: %s", run_dir)
    logger.info("Using GT thresholds: fdr=%s effect=%s", fdr_threshold, effect_threshold)

    inputs = prepare_inputs(
        run_dir=run_dir,
        metadata_path=args.metadata,
        fdr_threshold=fdr_threshold,
        effect_threshold=effect_threshold,
    )
    metrics = compute_fgs_per_experiment_metrics(inputs)
    leaderboard = compute_leaderboard(metrics)

    tables_dir = args.manuscript_tables_dir
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tables_dir / "figure3_fgs_per_experiment_metrics.tsv"
    leaderboard_path = tables_dir / "figure3_fgs_leaderboard.tsv"
    rank_path = tables_dir / "figure3_fgs_local_ranks.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    leaderboard.to_csv(leaderboard_path, sep="\t", index=False)
    write_rank_table(inputs, rank_path)
    logger.info("Wrote %s", metrics_path)
    logger.info("Wrote %s", leaderboard_path)
    logger.info("Wrote %s", rank_path)

    out_dir = args.manuscript_out_dir
    save_figure(
        plot_metric_distributions(inputs, metrics),
        out_dir,
        "figure3_fgs_metric_distributions",
        dpi=args.dpi,
    )
    save_figure(
        plot_leaderboard(inputs, leaderboard),
        out_dir,
        "figure3_fgs_leaderboard",
        dpi=args.dpi,
    )
    logger.info("Wrote Figure 3 draft assets under %s", out_dir)


if __name__ == "__main__":
    main()
