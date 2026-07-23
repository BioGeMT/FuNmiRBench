#!/usr/bin/env python3
"""Generate Supplementary Figure 2 local-rank manuscript assets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from figure_performance_common import (
    DEFAULT_METADATA_PATH,
    DEFAULT_RESULTS_DIR,
    find_latest_completed_run,
    load_run_thresholds,
    load_tool_metadata,
    save_figure,
)
from funmirbench import evaluate_common as ev
from funmirbench.logger import setup_logging


logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path("manuscript_assets/supplement")
OUTPUT_PREFIX = "supplement_figure2_rank_distributions"
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10.5
NEGATIVE_COLOR = "#B8C4D6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Supplementary Figure 2 local-rank distributions."
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
        help="Predictor metadata TSV used for display names, order, and score direction.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for Supplementary Figure 2 assets.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster output resolution.")
    parser.add_argument("--log-level", default="INFO", help="Logging level. Default: INFO.")
    return parser.parse_args()


def score_column(tool_id: str) -> str:
    return f"score_{tool_id}"


def load_metadata(path: Path) -> tuple[tuple[str, ...], dict[str, str], dict[str, str]]:
    tool_ids, labels = load_tool_metadata(path)
    metadata = pd.read_csv(path, sep="	", dtype=str)
    if "score_direction" not in metadata.columns:
        raise ValueError(f"{path} is missing column: score_direction")
    directions = dict(zip(metadata["tool_id"].astype(str), metadata["score_direction"].astype(str)))
    return tool_ids, labels, directions


def local_rank_scale(scores: pd.Series, *, direction: str) -> pd.Series:
    values = scores.astype(float)
    ascending = direction == "lower_is_stronger"
    ranks = values.rank(method="average", ascending=ascending)
    min_rank = ranks.min(skipna=True)
    max_rank = ranks.max(skipna=True)
    if pd.isna(min_rank) or pd.isna(max_rank):
        return pd.Series(float("nan"), index=scores.index)
    if float(max_rank) <= float(min_rank):
        return pd.Series(1.0, index=scores.index, dtype=float)
    # Scale so 1.0 is the strongest rank and 0.0 is the weakest rank within the dataset.
    return 1.0 - ((ranks - float(min_rank)) / (float(max_rank) - float(min_rank)))


def load_rank_data(
    *,
    run_dir: Path,
    metadata_path: Path,
    fdr_threshold: float | None,
    effect_threshold: float,
) -> tuple[tuple[str, ...], dict[str, str], dict[str, str], dict[str, list[float]], dict[str, list[float]]]:
    metadata_order, labels, directions = load_metadata(metadata_path)
    joined_paths = sorted(run_dir.glob("datasets/*/joined.tsv"))
    if not joined_paths:
        raise FileNotFoundError(f"No datasets/*/joined.tsv files found in {run_dir}")

    available = set()
    for path in joined_paths:
        columns = pd.read_csv(path, sep="	", nrows=0).columns
        available.update(column.removeprefix("score_") for column in columns if column.startswith("score_"))
    tool_ids = tuple(tool_id for tool_id in metadata_order if tool_id in available)
    if not tool_ids:
        raise ValueError("No score_<tool_id> columns were found in joined datasets.")

    positive_data = {tool_id: [] for tool_id in tool_ids}
    background_data = {tool_id: [] for tool_id in tool_ids}
    required = {"logFC", "FDR", "perturbation"}
    score_cols = [score_column(tool_id) for tool_id in tool_ids]
    for path in joined_paths:
        frame = pd.read_csv(path, sep="	", low_memory=False)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        work = ev._filter_usable_gt_rows(frame[["logFC", "FDR", "perturbation", *score_cols]], fdr_threshold=fdr_threshold)
        if work.empty:
            continue
        work = ev._annotate_ground_truth(work)
        is_positive = ev._positive_mask(
            work,
            fdr_threshold=fdr_threshold,
            effect_threshold=effect_threshold,
        )
        for tool_id in tool_ids:
            col = score_column(tool_id)
            if col not in work.columns:
                continue
            ranks = local_rank_scale(work[col], direction=directions.get(tool_id, "higher_is_stronger"))
            positive_data[tool_id].extend(ranks.loc[is_positive].dropna().astype(float).tolist())
            background_data[tool_id].extend(ranks.loc[~is_positive].dropna().astype(float).tolist())

    if not any(positive_data.values()) or not any(background_data.values()):
        raise ValueError("No local-rank data were available for positives and background pairs.")
    colors = {tool_id: ev.CURVE_COLORS[index % len(ev.CURVE_COLORS)] for index, tool_id in enumerate(tool_ids)}
    return tool_ids, labels, colors, positive_data, background_data


def wrap_label(label: str) -> str:
    if label == "TargetScan v8":
        return "TargetScan\nv8"
    return label


def plot_local_rank_distributions(
    *,
    tool_ids: tuple[str, ...],
    labels: dict[str, str],
    colors: dict[str, str],
    positive_data: dict[str, list[float]],
    background_data: dict[str, list[float]],
    out_dir: Path,
    dpi: int,
) -> dict[str, Path]:
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    positions = [index * 1.32 for index in range(len(tool_ids))]
    bg_positions = [position - 0.18 for position in positions]
    pos_positions = [position + 0.18 for position in positions]

    bg_values = [background_data[tool_id] for tool_id in tool_ids]
    pos_values = [positive_data[tool_id] for tool_id in tool_ids]
    bg_valid_positions = [pos for pos, values in zip(bg_positions, bg_values, strict=False) if values]
    bg_valid_values = [values for values in bg_values if values]
    pos_valid_positions = [pos for pos, values in zip(pos_positions, pos_values, strict=False) if values]
    pos_valid_values = [values for values in pos_values if values]
    pos_valid_colors = [colors[tool_id] for tool_id in tool_ids if positive_data[tool_id]]

    if bg_valid_values:
        bg_violin = ax.violinplot(
            bg_valid_values,
            positions=bg_valid_positions,
            widths=0.30,
            showmeans=False,
            showextrema=False,
            showmedians=True,
        )
        for body in bg_violin["bodies"]:
            body.set_facecolor(NEGATIVE_COLOR)
            body.set_edgecolor(NEGATIVE_COLOR)
            body.set_alpha(0.45)
        bg_violin["cmedians"].set_color("#22303C")
        bg_violin["cmedians"].set_linewidth(1.5)

    if pos_valid_values:
        pos_violin = ax.violinplot(
            pos_valid_values,
            positions=pos_valid_positions,
            widths=0.30,
            showmeans=False,
            showextrema=False,
            showmedians=True,
        )
        for body, color in zip(pos_violin["bodies"], pos_valid_colors, strict=False):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.40)
        pos_violin["cmedians"].set_color("#22303C")
        pos_violin["cmedians"].set_linewidth(1.5)

    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Local rank within dataset", fontsize=LABEL_SIZE)
    ax.set_xticks(positions)
    ax.set_xticklabels([wrap_label(labels[tool_id]) for tool_id in tool_ids], fontsize=TICK_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Positive vs background local rank distributions", fontsize=TITLE_SIZE, fontweight="bold", pad=16)
    ax.legend(
        handles=[
            Patch(facecolor=NEGATIVE_COLOR, edgecolor=NEGATIVE_COLOR, alpha=0.45, label="Background genes"),
            Patch(facecolor="#C8D6EA", edgecolor="#6E89A8", alpha=0.50, label="GT positives (predictor color)"),
        ],
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(top=0.86, bottom=0.28, left=0.10, right=0.98)
    return save_figure(fig, out_dir, OUTPUT_PREFIX, dpi=dpi)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    run_dir = args.run_dir.expanduser() if args.run_dir is not None else find_latest_completed_run(args.results_dir)
    fdr_threshold, effect_threshold = load_run_thresholds(run_dir)
    logger.info("Using run directory: %s", run_dir)
    logger.info("Using GT thresholds: fdr=%s effect=%s", fdr_threshold, effect_threshold)
    tool_ids, labels, colors, positive_data, background_data = load_rank_data(
        run_dir=run_dir,
        metadata_path=args.metadata,
        fdr_threshold=fdr_threshold,
        effect_threshold=effect_threshold,
    )
    outputs = plot_local_rank_distributions(
        tool_ids=tool_ids,
        labels=labels,
        colors=colors,
        positive_data=positive_data,
        background_data=background_data,
        out_dir=args.out_dir,
        dpi=args.dpi,
    )
    for path in outputs.values():
        logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        logger.error("%s", error)
        raise SystemExit(1) from None
