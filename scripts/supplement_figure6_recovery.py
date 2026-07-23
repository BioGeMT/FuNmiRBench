#!/usr/bin/env python3
"""Generate supplementary FPS recovery figures from Figure 6 tables."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from textwrap import shorten

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator

from figure_performance_common import (
    DEFAULT_MANUSCRIPT_TABLES_DIR,
    DEFAULT_METADATA_PATH,
    DEFAULT_FORMATS,
    evaluation_tool_colors,
    load_tool_metadata,
)
from funmirbench.logger import setup_logging


logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path("manuscript_assets/supplement")
DEFAULT_STEM = "supplement_figure6_fps_recovery"
TITLE_SIZE = 11
LABEL_SIZE = 9
TICK_SIZE = 8
PANEL_LABEL_SIZE = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate supplementary figures supporting manuscript Figure 6."
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_MANUSCRIPT_TABLES_DIR,
        help="Directory containing Figure 6 TSV outputs.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Predictor metadata TSV used for display names and predictor order.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for supplementary figures.",
    )
    parser.add_argument("--stem", default=DEFAULT_STEM, help="Output filename stem.")
    parser.add_argument("--columns", type=int, default=3, help="Panel columns. Default: 3.")
    parser.add_argument(
        "--plots-per-part",
        type=int,
        default=15,
        help="Number of dataset plots per recovery part. Default: 15.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster output resolution.")
    parser.add_argument("--log-level", default="INFO", help="Logging level. Default: INFO.")
    return parser.parse_args()


def read_figure6_tables(tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "recovery": tables_dir / "figure6_fps_recovery_by_budget.tsv",
        "precision": tables_dir / "figure6_fps_precision_at_recall.tsv",
        "best": tables_dir / "figure6_fps_best_precision_at_recall.tsv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Figure 6 tables. Run scripts/figure6_fps_recovery.py first. "
            f"Missing: {missing}"
        )
    return (
        pd.read_csv(paths["recovery"], sep="\t"),
        pd.read_csv(paths["precision"], sep="\t"),
        pd.read_csv(paths["best"], sep="\t"),
    )


def ordered_tools(metadata_path: Path, frame: pd.DataFrame) -> tuple[tuple[str, ...], dict[str, str], dict[str, str]]:
    metadata_order, metadata_labels = load_tool_metadata(metadata_path)
    present = set(frame["tool_id"].astype(str))
    tool_ids = tuple(tool_id for tool_id in metadata_order if tool_id in present)
    tool_ids = tool_ids + tuple(sorted(present.difference(tool_ids)))
    labels = {
        tool_id: metadata_labels.get(
            tool_id,
            str(frame.loc[frame["tool_id"] == tool_id, "predictor"].iloc[0]),
        )
        for tool_id in tool_ids
    }
    return tool_ids, labels, evaluation_tool_colors(tool_ids)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, *, dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in DEFAULT_FORMATS:
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.12,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="both", alpha=0.22, linewidth=0.7)
    ax.tick_params(labelsize=TICK_SIZE)


def short_dataset_label(dataset_id: str) -> str:
    return shorten(dataset_id.replace("_", " "), width=40, placeholder="...")


def write_recovery_parts(
    *,
    recovery: pd.DataFrame,
    tool_ids: tuple[str, ...],
    labels: dict[str, str],
    colors: dict[str, str],
    best: pd.DataFrame,
    out_dir: Path,
    tables_dir: Path,
    stem: str,
    columns: int,
    plots_per_part: int,
    dpi: int,
) -> tuple[list[Path], Path]:
    ordered_datasets = best.sort_values(
        ["precision", "false_positives_per_true_positive", "dataset_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )["dataset_id"].astype(str).tolist()
    remaining = sorted(set(recovery["dataset_id"].astype(str)).difference(ordered_datasets))
    dataset_ids = ordered_datasets + remaining

    columns = max(1, int(columns))
    plots_per_part = max(1, int(plots_per_part))
    part_count = int(np.ceil(len(dataset_ids) / plots_per_part))
    outputs: list[Path] = []
    manifest_rows = []
    budgets = sorted(int(value) for value in recovery["budget"].dropna().unique())
    panel_letters = [chr(ord("A") + index) for index in range(26)]

    for part_index in range(part_count):
        part_datasets = dataset_ids[
            part_index * plots_per_part : min((part_index + 1) * plots_per_part, len(dataset_ids))
        ]
        rows = int(np.ceil(len(part_datasets) / columns))
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(columns * 4.6, rows * 3.25),
            squeeze=False,
        )
        for panel_index, dataset_id in enumerate(part_datasets):
            ax = axes[panel_index // columns][panel_index % columns]
            data = recovery[recovery["dataset_id"].astype(str) == dataset_id]
            for tool_id in tool_ids:
                tool_data = data[data["tool_id"].astype(str) == tool_id].sort_values("budget")
                if tool_data.empty:
                    continue
                ax.plot(
                    tool_data["budget"],
                    tool_data["recall"],
                    marker="o",
                    linewidth=1.4,
                    markersize=3.6,
                    color=colors[tool_id],
                    label=labels[tool_id],
                )
            ax.set_xscale("log")
            ax.set_xticks(budgets)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(value):,}"))
            ax.set_ylim(0, 1.02)
            ax.set_title(short_dataset_label(dataset_id), fontsize=TITLE_SIZE, fontweight="bold")
            if panel_index // columns == rows - 1:
                ax.set_xlabel("Prediction budget", fontsize=LABEL_SIZE)
            if panel_index % columns == 0:
                ax.set_ylabel("Positive-pair recall", fontsize=LABEL_SIZE)
            style_axis(ax)
            if panel_index < len(panel_letters):
                add_panel_label(ax, panel_letters[panel_index])
            manifest_rows.append(
                {
                    "part": part_index + 1,
                    "panel_index": panel_index + 1,
                    "global_panel_index": part_index * plots_per_part + panel_index + 1,
                    "dataset_id": dataset_id,
                }
            )

        for empty_index in range(len(part_datasets), rows * columns):
            axes[empty_index // columns][empty_index % columns].axis("off")

        handles, legend_labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=len(tool_ids),
            frameon=False,
            fontsize=LABEL_SIZE,
        )
        fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.07, wspace=0.24, hspace=0.42)
        part_stem = f"{stem}_recovery_part{part_index + 1}"
        outputs.extend(save_figure(fig, out_dir, part_stem, dpi=dpi))

    tables_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tables_dir / f"{stem}_recovery_manifest.tsv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, sep="\t", index=False)
    return outputs, manifest_path


def write_precision_burden_summary(
    *,
    precision: pd.DataFrame,
    best: pd.DataFrame,
    tool_ids: tuple[str, ...],
    labels: dict[str, str],
    colors: dict[str, str],
    out_dir: Path,
    stem: str,
    dpi: int,
) -> list[Path]:
    ordered_datasets = best.sort_values(
        ["precision", "false_positives_per_true_positive", "dataset_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )["dataset_id"].astype(str).tolist()
    x_base = {dataset_id: index for index, dataset_id in enumerate(ordered_datasets)}
    offsets = np.linspace(-0.28, 0.28, len(tool_ids)) if len(tool_ids) > 1 else np.array([0.0])
    recall_target = float(precision["recall_target"].dropna().iloc[0])

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), gridspec_kw={"width_ratios": [1.5, 1.5, 0.9]})
    ax_precision, ax_burden, ax_counts = axes
    for offset, tool_id in zip(offsets, tool_ids, strict=False):
        data = precision[precision["tool_id"].astype(str) == tool_id].copy()
        data = data[data["dataset_id"].astype(str).isin(x_base)]
        x = data["dataset_id"].astype(str).map(x_base).astype(float) + offset
        ax_precision.scatter(
            x,
            data["precision"],
            s=24,
            color=colors[tool_id],
            edgecolor="white",
            linewidth=0.35,
            label=labels[tool_id],
        )
        ax_burden.scatter(
            x,
            data["false_positives_per_true_positive"],
            s=24,
            color=colors[tool_id],
            edgecolor="white",
            linewidth=0.35,
        )

    ax_precision.set_title(f"Precision at {recall_target:.0%} recall", fontsize=TITLE_SIZE, fontweight="bold")
    ax_precision.set_ylabel("Precision", fontsize=LABEL_SIZE)
    ax_precision.set_xlabel("Perturbation experiments", fontsize=LABEL_SIZE)
    ax_precision.set_xticks([])
    ax_precision.set_ylim(0, max(0.05, float(precision["precision"].max()) * 1.18))
    style_axis(ax_precision)
    add_panel_label(ax_precision, "A")

    ax_burden.set_title(
        f"False positives per true positive at {recall_target:.0%} recall",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    ax_burden.set_ylabel("False positives per true positive", fontsize=LABEL_SIZE)
    ax_burden.set_xlabel("Perturbation experiments", fontsize=LABEL_SIZE)
    ax_burden.set_xticks([])
    ax_burden.set_yscale("log")
    ax_burden.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    ax_burden.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
    style_axis(ax_burden)
    add_panel_label(ax_burden, "B")

    count_data = (
        best["tool_id"]
        .astype(str)
        .value_counts()
        .reindex(tool_ids, fill_value=0)
        .rename_axis("tool_id")
        .reset_index(name="experiments")
    )
    bar_labels = [labels[tool_id] for tool_id in count_data["tool_id"]]
    bar_colors = [colors[tool_id] for tool_id in count_data["tool_id"]]
    y = np.arange(len(count_data))
    ax_counts.barh(y, count_data["experiments"], color=bar_colors, alpha=0.75)
    ax_counts.set_yticks(y, bar_labels)
    ax_counts.invert_yaxis()
    ax_counts.set_xlabel("Experiments", fontsize=LABEL_SIZE)
    ax_counts.set_title("Best predictor", fontsize=TITLE_SIZE, fontweight="bold")
    for value, y_pos in zip(count_data["experiments"], y, strict=False):
        ax_counts.text(value + 0.25, y_pos, str(int(value)), va="center", fontsize=TICK_SIZE)
    ax_counts.set_xlim(0, max(1, int(count_data["experiments"].max())) + 2)
    style_axis(ax_counts)
    add_panel_label(ax_counts, "C")

    handles, legend_labels = ax_precision.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(tool_ids),
        frameon=False,
        fontsize=LABEL_SIZE,
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.78, bottom=0.16, wspace=0.36)
    return save_figure(fig, out_dir, f"{stem}_precision_burden", dpi=dpi)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    recovery, precision, best = read_figure6_tables(args.tables_dir)
    tool_ids, labels, colors = ordered_tools(args.metadata, recovery)
    logger.info("Predictors: %s", ", ".join(labels[tool_id] for tool_id in tool_ids))

    recovery_paths, manifest_path = write_recovery_parts(
        recovery=recovery,
        tool_ids=tool_ids,
        labels=labels,
        colors=colors,
        best=best,
        out_dir=args.out_dir,
        tables_dir=args.tables_dir,
        stem=args.stem,
        columns=args.columns,
        plots_per_part=args.plots_per_part,
        dpi=args.dpi,
    )
    precision_paths = write_precision_burden_summary(
        precision=precision,
        best=best,
        tool_ids=tool_ids,
        labels=labels,
        colors=colors,
        out_dir=args.out_dir,
        stem=args.stem,
        dpi=args.dpi,
    )
    for path in [*recovery_paths, *precision_paths, manifest_path]:
        logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as error:
        logger.error("%s", error)
        raise SystemExit(1) from None
