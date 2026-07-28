"""Shared manuscript performance figure helpers.

All universes are miRNA-gene pair universes:
- algorithm-specific: each predictor is evaluated only on its scored pairs.
- intersection pair set: all predictors are evaluated on pairs scored by every predictor.
- full pair set: all ground-truth pairs are evaluated; missing scores receive rank 0.

Optional gene-set filters restrict those pair universes by gene membership. The
IGS filter keeps ground-truth pairs whose genes are in the Figure 2
all-predictor Intersection Gene Set, then applies the configured pair universe.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score

from funmirbench.evaluate_common import CURVE_COLORS, TOP_PREDICTION_CDF_N, _rank_scale_scores
from funmirbench.logger import setup_logging


logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_METADATA_PATH = Path("metadata/predictions_info.tsv")
DEFAULT_MANUSCRIPT_TABLES_DIR = Path("manuscript_assets/tables")
DEFAULT_FORMATS = ("png", "svg")
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
RANDOM_TOOL_ID = "random_baseline"
UNIVERSE_ALGORITHM_SPECIFIC = "algorithm_specific"
UNIVERSE_INTERSECTION = "intersection_pair_set"
UNIVERSE_FULL = "full_pair_set"
GENE_SET_FILTER_IGS = "igs"
METRICS = ("aps", "pr_auc", "top_n_median_effect", "auroc", "spearman", "spearman_r2")
PANEL_SPECS = (
    ("A", "aps", "boxplot"),
    ("B", "top_n_median_effect", "boxplot"),
    ("C", "auroc", "boxplot"),
    ("D", "spearman", "boxplot"),
    ("E", "aps", "leaderboard"),
    ("F", "spearman_r2", "leaderboard"),
)
METRIC_LABELS = {
    "aps": "Average precision",
    "pr_auc": "PR-AUC",
    "top_n_median_effect": "Top-N median effect",
    "auroc": "AUROC",
    "spearman": "Spearman rho",
    "spearman_r2": "Spearman R2",
}
PANEL_LETTER_FONTSIZE = 18
PANEL_TITLE_FONTSIZE = 17
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
BAR_LABEL_FONTSIZE = 16
ANNOTATION_FONTSIZE = 14
LEGEND_FONTSIZE = 13
GRID_ALPHA = 0.25
PANEL_FIGSIZE = (7.2, 4.8)
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
    leaderboard_mode: str = "mean"
    panel_specs: tuple[tuple[str, str, str], ...] = PANEL_SPECS
    metric_ylim: dict[str, tuple[float, float]] | None = None
    include_random_in_leaderboard: bool = True
    winner_summary_fallback_universe: str | None = None
    gene_set_filter: str | None = None


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


def intersection_gene_set(inputs: PerformanceInputs) -> set[str]:
    """Return genes with at least one scored prediction from every evaluated predictor."""
    all_pairs = pd.concat((dataset.frame for dataset in inputs.datasets), ignore_index=True)
    scored_gene_sets = []
    for tool_id in inputs.tool_ids:
        scored_gene_sets.append(
            set(
                all_pairs.loc[all_pairs[score_column(tool_id)].notna(), "gene_id"]
                .dropna()
                .astype(str)
            )
        )
    return set.intersection(*scored_gene_sets) if scored_gene_sets else set()


def apply_gene_set_filter(
    frame: pd.DataFrame,
    *,
    gene_set_filter: str | None,
    igs_genes: set[str] | None,
) -> pd.DataFrame:
    if gene_set_filter is None:
        return frame
    if gene_set_filter != GENE_SET_FILTER_IGS:
        raise ValueError(f"Unsupported gene-set filter: {gene_set_filter}")
    if igs_genes is None:
        raise ValueError("IGS gene-set filter requires an IGS gene set.")
    return frame.loc[frame["gene_id"].astype(str).isin(igs_genes)].copy()


def metrics_universe_label(config: FigurePerformanceConfig) -> str:
    if config.gene_set_filter == GENE_SET_FILTER_IGS and config.universe == UNIVERSE_FULL:
        return "igs_restricted_full_pair_set"
    if config.gene_set_filter == GENE_SET_FILTER_IGS:
        return f"igs_restricted_{config.universe}"
    return config.universe


def compute_per_experiment_metrics(
    inputs: PerformanceInputs,
    config: FigurePerformanceConfig,
    *,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    tool_ids = (*inputs.tool_ids, RANDOM_TOOL_ID) if config.include_random else inputs.tool_ids
    igs_genes = intersection_gene_set(inputs) if config.gene_set_filter == GENE_SET_FILTER_IGS else None
    universe_label = metrics_universe_label(config)
    for dataset in inputs.datasets:
        frame = apply_gene_set_filter(
            dataset.frame,
            gene_set_filter=config.gene_set_filter,
            igs_genes=igs_genes,
        )
        if frame.empty:
            continue
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
                    "universe": universe_label,
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


def compute_metric_winner_counts(metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    winners = []
    work = metrics.dropna(subset=[metric]).copy()
    for dataset_id, group in work.groupby("dataset_id", sort=False):
        best_value = group[metric].max()
        row = group.loc[group[metric] == best_value, ["tool_id", "predictor"]].iloc[0]
        winners.append(
            {
                "dataset_id": dataset_id,
                "tool_id": row["tool_id"],
                "predictor": row["predictor"],
                metric: best_value,
            }
        )

    total_experiments = int(work["dataset_id"].nunique())
    if winners:
        winner_frame = pd.DataFrame(winners)
        counts = (
            winner_frame.groupby(["tool_id", "predictor"], as_index=False)
            .agg(best_experiments=("dataset_id", "nunique"))
        )
    else:
        counts = pd.DataFrame(columns=["tool_id", "predictor", "best_experiments"])

    predictors = metrics[["tool_id", "predictor"]].drop_duplicates()
    summary = predictors.merge(counts, on=["tool_id", "predictor"], how="left")
    summary["best_experiments"] = summary["best_experiments"].fillna(0).astype(int)
    summary["total_experiments"] = total_experiments
    summary["best_experiment_fraction"] = (
        summary["best_experiments"] / total_experiments if total_experiments else np.nan
    )
    return (
        summary.sort_values(
            ["best_experiments", "predictor"],
            ascending=[True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


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
    ax.grid(axis="y", alpha=GRID_ALPHA, linewidth=0.8)
    ax.set_axisbelow(True)


def set_panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.figure.text(
        0.025,
        0.965,
        letter,
        ha="left",
        va="top",
        fontsize=PANEL_LETTER_FONTSIZE,
        fontweight="bold",
    )
    ax.set_title(title, pad=18, fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, *, dpi: int) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for extension in DEFAULT_FORMATS:
        path = out_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths[extension] = path
    plt.close(fig)
    return paths


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
    canvas = np.ones((target_height, target_width, channels), dtype=image.dtype)
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


def parse_svg_viewbox(path: Path) -> tuple[float, float, ET.Element]:
    root = ET.parse(path).getroot()
    viewbox = root.attrib.get("viewBox")
    if viewbox is None:
        raise ValueError(f"SVG is missing a viewBox: {path}")
    parts = [float(part) for part in viewbox.split()]
    if len(parts) != 4:
        raise ValueError(f"Unexpected SVG viewBox for {path}: {viewbox}")
    return parts[2], parts[3], root


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
    metric_ylim: dict[str, tuple[float, float]] | None = None,
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
    set_panel_title(ax, letter, metric_label(metric, top_n=top_n))
    ax.set_xticks(
        positions,
        [tool_label(inputs, tool_id) for tool_id in plot_tool_ids],
        rotation=25,
        ha="right",
        fontsize=TICK_LABEL_FONTSIZE,
    )
    ax.set_ylabel(metric_label(metric, top_n=top_n), fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    if metric_ylim and metric in metric_ylim:
        ax.set_ylim(*metric_ylim[metric])
    elif metric in {"aps", "pr_auc", "auroc"}:
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
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    values = ordered[value_col].dropna()
    if values.empty:
        ax.set_xlim(0.0, 1.0)
    elif metric == "spearman_r2":
        ax.set_xlim(0.0, max(float(values.max()) * 1.18, 0.05))
    else:
        ax.set_xlim(0.0, max(float(values.max()) * 1.18, 0.05))
    set_panel_title(ax, letter, title)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    for y, value in enumerate(ordered[value_col]):
        if pd.isna(value):
            continue
        ax.text(value, y, f" {value:.3f}", va="center", fontsize=TICK_LABEL_FONTSIZE)
    style_axis(ax)


def draw_winner_count_panel(
    ax: plt.Axes,
    inputs: PerformanceInputs,
    winner_counts: pd.DataFrame,
    *,
    metric: str,
    title: str,
    letter: str,
) -> None:
    ordered = winner_counts.sort_values(
        ["best_experiments", "predictor"],
        ascending=[True, True],
        kind="mergesort",
    )
    colors = [tool_color(inputs, tool_id) for tool_id in ordered["tool_id"]]
    ax.barh(ordered["predictor"], ordered["best_experiments"], color=colors, alpha=0.55)
    total_experiments = (
        int(ordered["total_experiments"].max()) if not ordered.empty else 0
    )
    ax.set_xlabel(
        f"Experiments with best {metric_label(metric, top_n=TOP_PREDICTION_CDF_N).lower()}",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax.set_xlim(0, max(total_experiments, 1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    set_panel_title(ax, letter, title)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    for y, count in enumerate(ordered["best_experiments"]):
        ax.text(
            count,
            y,
            f" {int(count)}/{total_experiments}",
            va="center",
            fontsize=ANNOTATION_FONTSIZE,
        )
    style_axis(ax)


def draw_winner_summary_panel(
    ax: plt.Axes,
    inputs: PerformanceInputs,
    metrics: pd.DataFrame,
    *,
    letter: str,
    top_n: int,
    fallback_metrics: pd.DataFrame | None = None,
) -> None:
    summary_metrics = ("aps", "top_n_median_effect", "auroc", "spearman")
    y_positions = np.arange(len(summary_metrics))
    labels = [metric_label(metric, top_n=top_n) for metric in summary_metrics]
    metric_sources = {}
    eligible_totals = {}

    for metric in summary_metrics:
        metric_source = metrics
        if fallback_metrics is not None:
            primary_defined = set(metrics.dropna(subset=[metric])["dataset_id"].unique())
            missing_datasets = set(metrics["dataset_id"].unique()).difference(primary_defined)
            if missing_datasets:
                fallback_rows = fallback_metrics.loc[
                    fallback_metrics["dataset_id"].isin(missing_datasets)
                ]
                metric_source = pd.concat(
                    [
                        metrics.loc[~metrics["dataset_id"].isin(missing_datasets)],
                        fallback_rows,
                    ],
                    ignore_index=True,
                )
        metric_source = metric_source.dropna(subset=[metric]).copy()
        metric_sources[metric] = metric_source
        eligible_totals[metric] = int(metric_source["dataset_id"].nunique())
    show_eligible_counts = len(set(eligible_totals.values())) > 1

    for y, metric in zip(y_positions, summary_metrics):
        metric_source = metric_sources[metric]
        winner_counts = compute_metric_winner_counts(metric_source, metric).set_index("tool_id")
        total_experiments = eligible_totals[metric]
        left = 0
        for tool_id in inputs.tool_ids:
            if tool_id not in winner_counts.index:
                count = 0
            else:
                count = int(winner_counts.loc[tool_id, "best_experiments"])
            if count:
                ax.barh(
                    y,
                    count,
                    left=left,
                    color=tool_color(inputs, tool_id),
                    alpha=0.58,
                    height=0.56,
                )
                if count >= 2 or left + count >= total_experiments:
                    ax.text(
                        left + count / 2,
                        y,
                        str(count),
                        ha="center",
                        va="center",
                        fontsize=ANNOTATION_FONTSIZE,
                        color="#111111",
                        clip_on=True,
                    )
                else:
                    ax.text(
                        left + count + 0.15,
                        y,
                        str(count),
                        ha="left",
                        va="center",
                        fontsize=ANNOTATION_FONTSIZE,
                        color="#111111",
                        clip_on=True,
                    )
            left += count
        if show_eligible_counts:
            ax.text(
                total_experiments,
                y,
                f" n={total_experiments}",
                ha="left",
                va="center",
                fontsize=ANNOTATION_FONTSIZE,
                color="#4B5563",
                clip_on=False,
            )

    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    max_total = max(eligible_totals.values(), default=0)
    ax.set_xlim(0, max(max_total, 1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.set_xlabel("Experiments won", fontsize=AXIS_LABEL_FONTSIZE)
    set_panel_title(ax, letter, "Best predictor by metric")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    style_axis(ax)


def draw_legend_panel(ax: plt.Axes, inputs: PerformanceInputs, *, letter: str) -> None:
    ax.axis("off")
    y_step = 0.13
    total_height = y_step * (len(inputs.tool_ids) - 1)
    y_start = 0.5 + total_height / 2
    swatch_width = 0.08
    swatch_height = 0.07
    label_gap = 0.06
    group_width = 0.46
    group_left = 0.5 - group_width / 2
    for index, tool_id in enumerate(inputs.tool_ids):
        y = y_start - index * y_step
        ax.add_patch(
            plt.Rectangle(
                (group_left, y - swatch_height / 2),
                swatch_width,
                swatch_height,
                transform=ax.transAxes,
                color=tool_color(inputs, tool_id),
                alpha=0.58,
                clip_on=False,
            )
        )
        ax.text(
            group_left + swatch_width + label_gap,
            y,
            tool_label(inputs, tool_id),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=LEGEND_FONTSIZE,
            clip_on=False,
        )


def panel_stem(output_prefix: str, letter: str, metric: str, panel_type: str) -> str:
    safe_metric = metric.replace("_", "-")
    return f"{output_prefix}_panel_{letter.lower()}_{panel_type}_{safe_metric}"


def plot_metric_panel(
    inputs: PerformanceInputs,
    metrics: pd.DataFrame,
    metric: str,
    *,
    letter: str,
    top_n: int,
    metric_ylim: dict[str, tuple[float, float]] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    draw_metric_boxplot(
        ax,
        inputs,
        metrics,
        metric,
        letter=letter,
        top_n=top_n,
        metric_ylim=metric_ylim,
    )
    fig.tight_layout()
    return fig


def plot_leaderboard_panel(
    inputs: PerformanceInputs,
    leaderboard: pd.DataFrame,
    *,
    metric: str,
    value_col: str,
    title: str,
    xlabel: str,
    letter: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    draw_leaderboard(
        ax,
        inputs,
        leaderboard,
        metric=metric,
        value_col=value_col,
        title=title,
        xlabel=xlabel,
        letter=letter,
    )
    fig.tight_layout()
    return fig


def plot_winner_count_panel(
    inputs: PerformanceInputs,
    winner_counts: pd.DataFrame,
    *,
    metric: str,
    title: str,
    letter: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    draw_winner_count_panel(
        ax,
        inputs,
        winner_counts,
        metric=metric,
        title=title,
        letter=letter,
    )
    fig.tight_layout()
    return fig


def plot_winner_summary_panel(
    inputs: PerformanceInputs,
    metrics: pd.DataFrame,
    *,
    letter: str,
    top_n: int,
    fallback_metrics: pd.DataFrame | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    draw_winner_summary_panel(
        ax,
        inputs,
        metrics,
        letter=letter,
        top_n=top_n,
        fallback_metrics=fallback_metrics,
    )
    fig.tight_layout()
    return fig


def plot_legend_panel(inputs: PerformanceInputs, *, letter: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    draw_legend_panel(ax, inputs, letter=letter)
    fig.tight_layout()
    return fig


def leaderboard_inputs(inputs: PerformanceInputs, config: FigurePerformanceConfig) -> PerformanceInputs:
    if config.include_random_in_leaderboard:
        return inputs
    return PerformanceInputs(
        run_dir=inputs.run_dir,
        datasets=inputs.datasets,
        tool_ids=tuple(tool_id for tool_id in inputs.tool_ids if tool_id != RANDOM_TOOL_ID),
        tool_labels=inputs.tool_labels,
        tool_colors=inputs.tool_colors,
        fdr_threshold=inputs.fdr_threshold,
        effect_threshold=inputs.effect_threshold,
    )


def write_panel_figures(
    inputs: PerformanceInputs,
    metrics: pd.DataFrame,
    leaderboard: pd.DataFrame,
    spearman_r2_leaderboard: pd.DataFrame,
    aps_winner_counts: pd.DataFrame,
    spearman_r2_winner_counts: pd.DataFrame,
    config: FigurePerformanceConfig,
    *,
    out_dir: Path,
    dpi: int,
    top_n: int,
    winner_summary_fallback_metrics: pd.DataFrame | None = None,
) -> dict[str, dict[str, Path]]:
    panel_outputs: dict[str, dict[str, Path]] = {}
    lead_inputs = leaderboard_inputs(inputs, config)
    lead_metrics = metrics.loc[metrics["tool_id"] != RANDOM_TOOL_ID] if not config.include_random_in_leaderboard else metrics
    for letter, metric, panel_type in config.panel_specs:
        if panel_type == "boxplot":
            fig = plot_metric_panel(
                inputs,
                metrics,
                metric,
                letter=letter,
                top_n=top_n,
                metric_ylim=config.metric_ylim,
            )
        elif panel_type == "legend":
            fig = plot_legend_panel(lead_inputs, letter=letter)
        elif metric == "winner_summary" and config.leaderboard_mode == "winner_counts":
            fig = plot_winner_summary_panel(
                lead_inputs,
                lead_metrics,
                letter=letter,
                top_n=top_n,
                fallback_metrics=(
                    winner_summary_fallback_metrics.loc[
                        winner_summary_fallback_metrics["tool_id"] != RANDOM_TOOL_ID
                    ]
                    if winner_summary_fallback_metrics is not None
                    and not config.include_random_in_leaderboard
                    else winner_summary_fallback_metrics
                ),
            )
        elif metric == "aps":
            fig = plot_leaderboard_panel(
                inputs,
                leaderboard,
                metric="aps",
                value_col="mean_aps",
                title="APS leaderboard",
                xlabel="Mean average precision",
                letter=letter,
            )
        elif config.leaderboard_mode == "winner_counts":
            fig = plot_winner_count_panel(
                lead_inputs,
                spearman_r2_winner_counts.loc[
                    spearman_r2_winner_counts["tool_id"] != RANDOM_TOOL_ID
                ] if not config.include_random_in_leaderboard else spearman_r2_winner_counts,
                metric="spearman_r2",
                title="Best Spearman R2 by experiment",
                letter=letter,
            )
        else:
            fig = plot_leaderboard_panel(
                inputs,
                spearman_r2_leaderboard,
                metric="spearman_r2",
                value_col="mean_metric",
                title="Spearman R2 leaderboard",
                xlabel="Mean Spearman R2",
                letter=letter,
            )
        panel_outputs[letter] = save_figure(
            fig,
            out_dir,
            panel_stem(config.output_prefix, letter, metric, panel_type),
            dpi=dpi,
        )
    return panel_outputs


def write_combined_png(
    panel_outputs: dict[str, dict[str, Path]],
    *,
    out_dir: Path,
    output_prefix: str,
    dpi: int,
) -> None:
    required = ("A", "B", "C", "D", "E", "F")
    missing = [panel for panel in required if panel not in panel_outputs or "png" not in panel_outputs[panel]]
    if missing:
        raise ValueError(f"Missing PNG panel outputs: {missing}")

    panel_images = {
        panel: plt.imread(panel_outputs[panel]["png"])
        for panel in required
    }
    row_pairs = (("A", "B"), ("C", "D"), ("E", "F"))
    column_pairs = (("A", "C", "E"), ("B", "D", "F"))
    row_heights = [
        max(panel_images[panel].shape[0] for panel in row_pair)
        for row_pair in row_pairs
    ]
    column_widths = [
        max(panel_images[panel].shape[1] for panel in column_pair)
        for column_pair in column_pairs
    ]
    cells = {
        panel: pad_to_shape(
            panel_images[panel],
            (row_heights[row_index], column_widths[column_index]),
            vertical="top",
        )
        for row_index, row_pair in enumerate(row_pairs)
        for column_index, panel in enumerate(row_pair)
    }
    gap_x = round(18.0 / 72.0 * dpi)
    gap_y = round(10.0 / 72.0 * dpi)
    channels = next(iter(cells.values())).shape[2]
    dtype = next(iter(cells.values())).dtype
    rows = []
    for left_panel, right_panel in row_pairs:
        rows.append(
            np.hstack(
                [
                    cells[left_panel],
                    np.ones((cells[left_panel].shape[0], gap_x, channels), dtype=dtype),
                    cells[right_panel],
                ]
            )
        )
    combined_parts = []
    for index, row in enumerate(rows):
        if index:
            combined_parts.append(np.ones((gap_y, rows[0].shape[1], channels), dtype=dtype))
        combined_parts.append(row)
    combined = np.vstack(combined_parts)

    fig_width = 16.2
    fig_height = fig_width * combined.shape[0] / combined.shape[1]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(combined)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{output_prefix}_combined.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_combined_svg(
    panel_outputs: dict[str, dict[str, Path]],
    *,
    out_dir: Path,
    output_prefix: str,
) -> None:
    required = ("A", "B", "C", "D", "E", "F")
    missing = [panel for panel in required if panel not in panel_outputs or "svg" not in panel_outputs[panel]]
    if missing:
        raise ValueError(f"Missing SVG panel outputs: {missing}")

    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)
    row_pairs = (("A", "B"), ("C", "D"), ("E", "F"))
    panel_svgs = {panel: parse_svg_viewbox(panel_outputs[panel]["svg"]) for panel in required}
    column_widths = [
        max(panel_svgs[panel][0] for panel in column_pair)
        for column_pair in (("A", "C", "E"), ("B", "D", "F"))
    ]
    row_heights = [
        max(panel_svgs[panel][1] for panel in row_pair)
        for row_pair in row_pairs
    ]
    gap_x = 18.0
    gap_y = 10.0
    combined_width = sum(column_widths) + gap_x
    combined_height = sum(row_heights) + gap_y * (len(row_heights) - 1)
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": f"{combined_width:g}pt",
            "height": f"{combined_height:g}pt",
            "viewBox": f"0 0 {combined_width:g} {combined_height:g}",
            "version": "1.1",
        },
    )
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}rect",
        {"x": "0", "y": "0", "width": f"{combined_width:g}", "height": f"{combined_height:g}", "fill": "#ffffff"},
    )
    for row_index, (left_panel, right_panel) in enumerate(row_pairs):
        y = sum(row_heights[:row_index]) + gap_y * row_index
        for column_index, panel in enumerate((left_panel, right_panel)):
            panel_width, panel_height, panel_root = panel_svgs[panel]
            x = sum(column_widths[:column_index]) + gap_x * column_index
            nested = ET.SubElement(
                root,
                f"{{{SVG_NS}}}svg",
                {
                    "x": f"{x:g}",
                    "y": f"{y:g}",
                    "width": f"{panel_width:g}",
                    "height": f"{panel_height:g}",
                    "viewBox": panel_root.attrib["viewBox"],
                },
            )
            for child in list(panel_root):
                nested.append(child)

    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{output_prefix}_combined.svg"
    ET.ElementTree(root).write(svg_path, encoding="utf-8", xml_declaration=True)
    svg_text = "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
    svg_path.write_text(f"{svg_text}\n", encoding="utf-8")


def write_combined_figure(
    panel_outputs: dict[str, dict[str, Path]],
    *,
    out_dir: Path,
    output_prefix: str,
    dpi: int,
) -> None:
    write_combined_png(panel_outputs, out_dir=out_dir, output_prefix=output_prefix, dpi=dpi)
    write_combined_svg(panel_outputs, out_dir=out_dir, output_prefix=output_prefix)


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
    winner_summary_fallback_metrics = None
    if config.winner_summary_fallback_universe is not None:
        fallback_config = FigurePerformanceConfig(
            figure_id=config.figure_id,
            universe=config.winner_summary_fallback_universe,
            title=config.title,
            output_prefix=config.output_prefix,
            include_random=config.include_random,
            default_out_dir=config.default_out_dir,
            random_label=config.random_label,
            leaderboard_mode=config.leaderboard_mode,
            panel_specs=config.panel_specs,
            metric_ylim=config.metric_ylim,
            include_random_in_leaderboard=config.include_random_in_leaderboard,
            gene_set_filter=config.gene_set_filter,
        )
        winner_summary_fallback_metrics = compute_per_experiment_metrics(
            inputs,
            fallback_config,
            top_n=args.top_n,
        )
    leaderboard = compute_leaderboard(metrics)
    spearman_r2_leaderboard = compute_metric_leaderboard(metrics, "spearman_r2")
    aps_winner_counts = compute_metric_winner_counts(metrics, "aps")
    spearman_r2_winner_counts = compute_metric_winner_counts(metrics, "spearman_r2")

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
    panel_outputs = write_panel_figures(
        inputs,
        metrics,
        leaderboard,
        spearman_r2_leaderboard,
        aps_winner_counts,
        spearman_r2_winner_counts,
        config,
        out_dir=out_dir,
        dpi=args.dpi,
        top_n=args.top_n,
        winner_summary_fallback_metrics=winner_summary_fallback_metrics,
    )
    write_combined_figure(panel_outputs, out_dir=out_dir, output_prefix=config.output_prefix, dpi=args.dpi)
    logger.info("Wrote %s draft assets under %s", config.figure_id, out_dir)
