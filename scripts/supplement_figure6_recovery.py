#!/usr/bin/env python3
"""Generate supplementary FPS recovery figures from Figure 6 tables."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

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
TOOL_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "*")


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
    parser.add_argument("--dpi", type=int, default=300, help="Raster output resolution.")
    parser.add_argument("--log-level", default="INFO", help="Logging level. Default: INFO.")
    return parser.parse_args()


def read_figure6_tables(tables_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = {
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


def tool_markers(tool_ids: tuple[str, ...]) -> dict[str, str]:
    return {
        tool_id: TOOL_MARKERS[index % len(TOOL_MARKERS)]
        for index, tool_id in enumerate(tool_ids)
    }


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, *, dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in DEFAULT_FORMATS:
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        if suffix == "svg":
            svg_text = "\n".join(
                line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
            )
            path.write_text(f"{svg_text}\n", encoding="utf-8")
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


def write_precision_burden_summary(
    *,
    precision: pd.DataFrame,
    best: pd.DataFrame,
    tool_ids: tuple[str, ...],
    labels: dict[str, str],
    colors: dict[str, str],
    markers: dict[str, str],
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

    fig = plt.figure(figsize=(13.8, 7.2))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(4.7, 0.75),
        height_ratios=(1, 1),
        wspace=0.02,
        hspace=0.42,
    )
    ax_precision = fig.add_subplot(grid[0, 0])
    ax_burden = fig.add_subplot(grid[1, 0])
    ax_legend = fig.add_subplot(grid[:, 1])
    for offset, tool_id in zip(offsets, tool_ids, strict=False):
        data = precision[precision["tool_id"].astype(str) == tool_id].copy()
        data = data[data["dataset_id"].astype(str).isin(x_base)]
        x = data["dataset_id"].astype(str).map(x_base).astype(float) + offset
        ax_precision.scatter(
            x,
            data["precision"],
            s=34,
            color=colors[tool_id],
            marker=markers[tool_id],
            edgecolor="#1F2937",
            linewidth=0.6,
            label=labels[tool_id],
        )
        ax_burden.scatter(
            x,
            data["false_positives_per_true_positive"],
            s=34,
            color=colors[tool_id],
            marker=markers[tool_id],
            edgecolor="#1F2937",
            linewidth=0.6,
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

    handles, legend_labels = ax_precision.get_legend_handles_labels()
    ax_legend.axis("off")
    legend = ax_legend.legend(
        handles,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        title="Predictor",
        title_fontsize=TITLE_SIZE,
        frameon=False,
        fontsize=LABEL_SIZE + 1,
        handlelength=2.4,
        labelspacing=1.1,
    )
    legend.get_title().set_fontweight("bold")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.09)
    return save_figure(fig, out_dir, f"{stem}_precision_burden", dpi=dpi)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    precision, best = read_figure6_tables(args.tables_dir)
    tool_ids, labels, colors = ordered_tools(args.metadata, precision)
    markers = tool_markers(tool_ids)
    logger.info("Predictors: %s", ", ".join(labels[tool_id] for tool_id in tool_ids))

    precision_paths = write_precision_burden_summary(
        precision=precision,
        best=best,
        tool_ids=tool_ids,
        labels=labels,
        colors=colors,
        markers=markers,
        out_dir=args.out_dir,
        stem=args.stem,
        dpi=args.dpi,
    )
    for path in precision_paths:
        logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as error:
        logger.error("%s", error)
        raise SystemExit(1) from None
