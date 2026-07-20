"""Shared manuscript performance figure helpers.

All universes are miRNA-gene pair universes:
- algorithm-specific: each predictor is evaluated only on its scored pairs.
- intersection pair set: all predictors are evaluated on pairs scored by every predictor.
- full pair set: all ground-truth pairs are evaluated; missing scores receive rank 0.
"""

from __future__ import annotations

import argparse
import hashlib
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

from funmirbench.evaluate_common import CURVE_COLORS, TOP_PREDICTION_CDF_N, _rank_scale_scores
from funmirbench.logger import setup_logging


logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_METADATA_PATH = Path("metadata/predictions_info.tsv")
DEFAULT_MANUSCRIPT_TABLES_DIR = Path("manuscript_assets/tables")
DEFAULT_FORMATS = ("png", "svg")
RANDOM_TOOL_ID = "random_baseline"
UNIVERSE_ALGORITHM_SPECIFIC = "algorithm_specific"
UNIVERSE_INTERSECTION = "intersection_pair_set"
UNIVERSE_FULL = "full_pair_set"
METRICS = ("aps", "pr_auc", "top_n_median_effect", "auroc", "spearman", "spearman_r2")
PLOT_METRICS = ("aps", "top_n_median_effect", "auroc", "spearman")
METRIC_LABELS = {
    "aps": "Average precision",
    "pr_auc": "PR-AUC",
    "top_n_median_effect": "Top-N median effect",
    "auroc": "AUROC",
    "spearman": "Spearman rho",
    "spearman_r2": "Spearman R2",
}
FIGURE_TITLE_SIZE = 13
PANEL_TITLE_SIZE = 11
AXIS_LABEL_SIZE = 10
TICK_LABEL_SIZE = 9
OE_ALIASES = {"OE", "OVEREXPRESSION", "OVER_EXPRESSION"}
LOSS_ALIASES = {"KO", "KD", "KNOCKOUT", "KNOCK_OUT", "KNOCKDOWN", "KNOCK_DOWN"}


@dataclass(frozen=True)
class BenchmarkDataset:
    dataset_id: str
    frame: pd.DataFrame


@dataclass(frozen=True)
class PerformanceInputs:
    run_dir: Path
    datasets: tuple[BenchmarkDataset, ...]
    tool_ids: tuple[str, ...]
    tool_labels: dict[str, str]
    tool_colors: dict[str, str]
    fdr_threshold: float | None
    effect_threshold: float


@dataclass(frozen=True)
class FigurePerformanceConfig:
    figure_id: str
    universe: str
    title: str
    output_prefix: str
    include_random: bool
    default_out_dir: Path
    random_label: str = "Random baseline"


def parse_args(config: FigurePerformanceConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Generate manuscript {config.figure_id} performance tables and draft panels."
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
        default=config.default_out_dir,
        help=f"Stable manuscript figure output directory. Default: {config.default_out_dir}/.",
    )
    parser.add_argument(
        "--manuscript-tables-dir",
        type=Path,
        default=DEFAULT_MANUSCRIPT_TABLES_DIR,
        help="Stable manuscript table output directory. Default: manuscript_assets/tables/.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster output resolution.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=TOP_PREDICTION_CDF_N,
        help=f"Number of top predictions used for the CDF-derived median effect metric. Default: {TOP_PREDICTION_CDF_N}.",
    )
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
    fdr = evaluation.get("fdr_threshold", 0.05)
    return (None if fdr is None else float(fdr), float(evaluation.get("effect_threshold", 1.0)))


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
    common_tool_ids.discard(RANDOM_TOOL_ID)
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
) -> PerformanceInputs:
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
    tool_colors = evaluation_tool_colors(tool_ids)
    tool_labels = {tool_id: labels.get(tool_id, tool_id) for tool_id in tool_ids}
    return PerformanceInputs(
        run_dir=run_dir,
        datasets=annotated,
        tool_ids=tool_ids,
        tool_labels=tool_labels,
        tool_colors=tool_colors,
        fdr_threshold=fdr_threshold,
        effect_threshold=effect_threshold,
    )


def pr_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def top_n_median_effect(
    frame: pd.DataFrame,
    scores: pd.Series,
    *,
    top_n: int,
) -> float:
    if top_n <= 0:
        raise ValueError("--top-n must be greater than 0.")
    work = frame[["expected_effect"]].copy()
    work["score"] = scores.astype(float)
    if "gene_id" in frame.columns:
        work["gene_id"] = frame["gene_id"]
    work = work.dropna(subset=["score", "expected_effect"])
    if work.empty:
        return float("nan")
    sort_cols = ["score"]
    ascending = [False]
    if "gene_id" in work.columns:
        sort_cols.append("gene_id")
        ascending.append(True)
    work = work.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    top_values = work["expected_effect"].head(min(int(top_n), len(work)))
    return float(np.nanmedian(top_values))


def compute_metrics(
    y_true: pd.Series,
    y_score: pd.Series,
    expected_effect: pd.Series,
    *,
    top_n_median_effect_value: float,
) -> dict[str, float]:
    y_true_array = y_true.astype(int).to_numpy()
    y_score_array = y_score.astype(float).to_numpy()
    expected_effect_array = expected_effect.astype(float).to_numpy()
    positives = int(y_true_array.sum())
    negatives = int(len(y_true_array) - positives)
    spearman = spearmanr(y_score_array, expected_effect_array).correlation
    spearman_value = float(spearman) if pd.notna(spearman) else float("nan")
    spearman_r2 = spearman_value**2 if pd.notna(spearman_value) else float("nan")
    if positives == 0 or negatives == 0:
        metrics = {metric: float("nan") for metric in METRICS}
        metrics["top_n_median_effect"] = top_n_median_effect_value
        metrics["spearman"] = spearman_value
        metrics["spearman_r2"] = spearman_r2
        return metrics
    return {
        "aps": float(average_precision_score(y_true_array, y_score_array)),
        "pr_auc": pr_auc_score(y_true_array, y_score_array),
        "top_n_median_effect": top_n_median_effect_value,
        "auroc": float(roc_auc_score(y_true_array, y_score_array)),
        "spearman": spearman_value,
        "spearman_r2": spearman_r2,
    }


def random_rank_series(frame: pd.DataFrame, dataset_id: str, universe: str) -> pd.Series:
    seed = int.from_bytes(
        hashlib.sha256(f"{dataset_id}:{universe}".encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )
    random_scores = pd.Series(np.random.default_rng(seed).random(len(frame)), index=frame.index)
    return _rank_scale_scores(random_scores).fillna(0.0).astype(float)


def rank_series(frame: pd.DataFrame, tool_id: str, *, dataset_id: str, universe: str) -> pd.Series:
    if tool_id == RANDOM_TOOL_ID:
        return random_rank_series(frame, dataset_id, universe)
    ranks = _rank_scale_scores(frame[score_column(tool_id)])
    if universe == UNIVERSE_FULL:
        return ranks.fillna(0.0).astype(float)
    return ranks.astype(float)


def evaluation_frame(frame: pd.DataFrame, tool_id: str, tool_ids: tuple[str, ...], universe: str) -> pd.DataFrame:
    if tool_id == RANDOM_TOOL_ID:
        if universe == UNIVERSE_INTERSECTION:
            score_cols = [score_column(real_tool_id) for real_tool_id in tool_ids if real_tool_id != RANDOM_TOOL_ID]
            return frame.loc[frame[score_cols].notna().all(axis=1)].copy()
        return frame.copy()

    score_col = score_column(tool_id)
    if universe == UNIVERSE_ALGORITHM_SPECIFIC:
        return frame.loc[frame[score_col].notna()].copy()
    if universe == UNIVERSE_INTERSECTION:
        score_cols = [score_column(real_tool_id) for real_tool_id in tool_ids if real_tool_id != RANDOM_TOOL_ID]
        return frame.loc[frame[score_cols].notna().all(axis=1)].copy()
    if universe == UNIVERSE_FULL:
        return frame.copy()
    raise ValueError(f"Unsupported performance universe: {universe}")


def compute_per_experiment_metrics(
    inputs: PerformanceInputs,
    config: FigurePerformanceConfig,
    *,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    tool_ids = (*inputs.tool_ids, RANDOM_TOOL_ID) if config.include_random else inputs.tool_ids
    for dataset in inputs.datasets:
        frame = dataset.frame
        for tool_id in tool_ids:
            score_col = score_column(tool_id)
            if tool_id != RANDOM_TOOL_ID and score_col not in frame.columns:
                continue
            work = evaluation_frame(frame, tool_id, tool_ids, config.universe)
            if work.empty:
                continue
            scores = rank_series(
                work,
                tool_id,
                dataset_id=dataset.dataset_id,
                universe=config.universe,
            )
            metrics = compute_metrics(
                work["is_positive"],
                scores,
                work["expected_effect"],
                top_n_median_effect_value=top_n_median_effect(work, scores, top_n=top_n),
            )
            total = int(len(frame))
            positives_total = int(frame["is_positive"].sum())
            if config.universe in {UNIVERSE_ALGORITHM_SPECIFIC, UNIVERSE_INTERSECTION}:
                scored = int(len(work))
                positives_scored = int(work["is_positive"].sum())
                predictor = config.random_label if tool_id == RANDOM_TOOL_ID else inputs.tool_labels[tool_id]
            elif tool_id == RANDOM_TOOL_ID:
                scored = int(len(work))
                positives_scored = int(work["is_positive"].sum())
                predictor = config.random_label
            else:
                scored = int(frame[score_col].notna().sum())
                positives_scored = int(frame.loc[frame[score_col].notna(), "is_positive"].sum())
                predictor = inputs.tool_labels[tool_id]
            rows.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "universe": config.universe,
                    "tool_id": tool_id,
                    "predictor": predictor,
                    "pairs_evaluated": int(len(work)),
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
        mean_top_n_median_effect=("top_n_median_effect", "mean"),
        mean_auroc=("auroc", "mean"),
        mean_spearman=("spearman", "mean"),
        mean_spearman_r2=("spearman_r2", "mean"),
        mean_coverage=("coverage", "mean"),
        mean_positive_coverage=("positive_coverage", "mean"),
    )
    summary = summary.sort_values(
        ["mean_aps", "mean_pr_auc", "mean_auroc"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


def compute_metric_leaderboard(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    summary = (
        metrics.groupby(["tool_id", "predictor"], as_index=False)
        .agg(
            datasets=("dataset_id", "nunique"),
            mean_metric=(metric, "mean"),
            median_metric=(metric, "median"),
        )
        .sort_values("mean_metric", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary.insert(4, "metric", metric)
    return summary


def write_rank_table(inputs: PerformanceInputs, out_path: Path, *, include_random: bool) -> None:
    chunks = []
    tool_ids = (*inputs.tool_ids, RANDOM_TOOL_ID) if include_random else inputs.tool_ids
    for dataset in inputs.datasets:
        frame = dataset.frame[["dataset_id", "mirna", "perturbation", "gene_id", "is_positive"]].copy()
        for tool_id in tool_ids:
            frame[f"fps_rank_{tool_id}"] = rank_series(
                dataset.frame,
                tool_id,
                dataset_id=dataset.dataset_id,
                universe=UNIVERSE_FULL,
            )
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


def tool_label(inputs: PerformanceInputs, tool_id: str) -> str:
    if tool_id == RANDOM_TOOL_ID:
        return "Random baseline"
    return inputs.tool_labels[tool_id]


def tool_color(inputs: PerformanceInputs, tool_id: str) -> str:
    if tool_id == RANDOM_TOOL_ID:
        return "#7A8798"
    return inputs.tool_colors[tool_id]


def metric_label(metric: str, *, top_n: int) -> str:
    if metric == "top_n_median_effect":
        return f"Top-{top_n} median effect"
    return METRIC_LABELS[metric]


def draw_metric_boxplot(
    ax: plt.Axes,
    inputs: PerformanceInputs,
    metrics: pd.DataFrame,
    metric: str,
    *,
    letter: str,
    top_n: int,
) -> None:
    plot_tool_ids = tuple(metrics["tool_id"].drop_duplicates().tolist())
    positions = np.arange(1, len(plot_tool_ids) + 1)
    data = [
        metrics.loc[metrics["tool_id"] == tool_id, metric].dropna().astype(float).to_numpy()
        for tool_id in plot_tool_ids
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
    for patch, tool_id in zip(box["boxes"], plot_tool_ids):
        color = tool_color(inputs, tool_id)
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.28)

    for index, (tool_id, values) in enumerate(zip(plot_tool_ids, data), start=1):
        if values.size == 0:
            continue
        jitter = np.linspace(-0.09, 0.09, values.size) if values.size > 1 else np.array([0.0])
        ax.scatter(
            np.full(values.size, index, dtype=float) + jitter,
            values,
            s=30,
            color=tool_color(inputs, tool_id),
            edgecolor="white",
            linewidth=0.45,
            alpha=0.75,
            zorder=3,
        )
    ax.set_title(
        f"{letter}. {metric_label(metric, top_n=top_n)}",
        loc="left",
        fontweight="bold",
        fontsize=PANEL_TITLE_SIZE,
    )
    ax.set_xticks(
        positions,
        [tool_label(inputs, tool_id) for tool_id in plot_tool_ids],
        rotation=25,
        ha="right",
        fontsize=TICK_LABEL_SIZE,
    )
    ax.set_ylabel(metric_label(metric, top_n=top_n), fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    if metric in {"aps", "pr_auc", "auroc"}:
        ax.set_ylim(0.0, 1.02)
    elif metric == "spearman":
        ax.set_ylim(-1.02, 1.02)
    style_axis(ax)


def draw_leaderboard(
    ax: plt.Axes,
    inputs: PerformanceInputs,
    leaderboard: pd.DataFrame,
    *,
    metric: str,
    value_col: str,
    title: str,
    xlabel: str,
    letter: str,
) -> None:
    ordered = leaderboard.sort_values(value_col, ascending=True)
    colors = [tool_color(inputs, tool_id) for tool_id in ordered["tool_id"]]
    ax.barh(ordered["predictor"], ordered[value_col], color=colors, alpha=0.45)
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
    values = ordered[value_col].dropna()
    if values.empty:
        ax.set_xlim(0.0, 1.0)
    elif metric == "spearman_r2":
        ax.set_xlim(0.0, max(float(values.max()) * 1.18, 0.05))
    else:
        ax.set_xlim(0.0, max(float(values.max()) * 1.18, 0.05))
    ax.set_title(f"{letter}. {title}", loc="left", fontweight="bold", fontsize=PANEL_TITLE_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    for y, value in enumerate(ordered[value_col]):
        if pd.isna(value):
            continue
        ax.text(value, y, f" {value:.3f}", va="center", fontsize=TICK_LABEL_SIZE)
    style_axis(ax)


def plot_performance_figure(
    inputs: PerformanceInputs,
    metrics: pd.DataFrame,
    leaderboard: pd.DataFrame,
    spearman_r2_leaderboard: pd.DataFrame,
    config: FigurePerformanceConfig,
    *,
    top_n: int,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=(13.2, 12.0))
    for ax, metric, letter in zip(axes[:2].ravel(), PLOT_METRICS, ("A", "B", "C", "D")):
        draw_metric_boxplot(ax, inputs, metrics, metric, letter=letter, top_n=top_n)
    draw_leaderboard(
        axes[2, 0],
        inputs,
        leaderboard,
        metric="aps",
        value_col="mean_aps",
        title="APS leaderboard",
        xlabel="Mean average precision",
        letter="E",
    )
    draw_leaderboard(
        axes[2, 1],
        inputs,
        spearman_r2_leaderboard,
        metric="spearman_r2",
        value_col="mean_metric",
        title="Spearman R2 leaderboard",
        xlabel="Mean Spearman R2",
        letter="F",
    )
    fig.suptitle(config.title, fontsize=FIGURE_TITLE_SIZE, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig


def run_performance_figure(config: FigurePerformanceConfig) -> None:
    args = parse_args(config)
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
    metrics = compute_per_experiment_metrics(inputs, config, top_n=args.top_n)
    leaderboard = compute_leaderboard(metrics)
    spearman_r2_leaderboard = compute_metric_leaderboard(metrics, "spearman_r2")

    tables_dir = args.manuscript_tables_dir
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tables_dir / f"{config.output_prefix}_per_experiment_metrics.tsv"
    leaderboard_path = tables_dir / f"{config.output_prefix}_leaderboard.tsv"
    spearman_r2_leaderboard_path = tables_dir / f"{config.output_prefix}_spearman_r2_leaderboard.tsv"
    rank_path = tables_dir / f"{config.output_prefix}_local_ranks.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    leaderboard.to_csv(leaderboard_path, sep="\t", index=False)
    spearman_r2_leaderboard.to_csv(spearman_r2_leaderboard_path, sep="\t", index=False)
    if config.universe == UNIVERSE_FULL:
        write_rank_table(inputs, rank_path, include_random=config.include_random)
        logger.info("Wrote %s", rank_path)
    logger.info("Wrote %s", metrics_path)
    logger.info("Wrote %s", leaderboard_path)
    logger.info("Wrote %s", spearman_r2_leaderboard_path)

    out_dir = args.manuscript_out_dir
    save_figure(
        plot_performance_figure(
            inputs,
            metrics,
            leaderboard,
            spearman_r2_leaderboard,
            config,
            top_n=args.top_n,
        ),
        out_dir,
        f"{config.output_prefix}_combined",
        dpi=args.dpi,
    )
    logger.info("Wrote %s draft assets under %s", config.figure_id, out_dir)
