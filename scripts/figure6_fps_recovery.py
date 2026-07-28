#!/usr/bin/env python3
"""Generate Figure 6 FPS recovery and validation-burden outputs."""

from __future__ import annotations

import argparse
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, LogLocator

from figure_performance_common import (
    DEFAULT_MANUSCRIPT_TABLES_DIR,
    DEFAULT_METADATA_PATH,
    DEFAULT_RESULTS_DIR,
    UNIVERSE_FULL,
    PerformanceInputs,
    SVG_NS,
    evaluation_frame,
    find_latest_completed_run,
    load_run_thresholds,
    pad_to_shape,
    parse_svg_viewbox,
    prepare_inputs,
    rank_series,
    save_figure,
    trim_white_border,
)
from funmirbench.logger import setup_logging


logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path("manuscript_assets/figure6")
DEFAULT_BUDGETS = (10, 25, 50, 100, 250, 500, 1000, 2500, 5000)
RECALL_TARGET = 0.50
TITLE_SIZE = 14
LABEL_SIZE = 11
TICK_SIZE = 9
LEGEND_SIZE = 11
LEGEND_TITLE_SIZE = 12
PANEL_LABEL_SIZE = 15
PANEL_FIGSIZE = (7.2, 4.8)
OUTPUT_PREFIX = "figure6_fps_recovery"
TOOL_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "*")


@dataclass(frozen=True)
class Figure6Config:
    run_dir: Path
    metadata_path: Path
    out_dir: Path
    tables_dir: Path
    budgets: tuple[int, ...]
    recall_target: float
    dpi: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manuscript Figure 6 FPS recovery outputs."
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
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Stable manuscript figure output directory.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_MANUSCRIPT_TABLES_DIR,
        help="Stable manuscript table output directory.",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
        help="Prediction budgets for recovery analysis. Default: 10 25 50 100 250 500 1000 2500 5000.",
    )
    parser.add_argument(
        "--recall-target",
        type=float,
        default=RECALL_TARGET,
        help="Recall threshold for precision and false-positive burden panels. Default: 0.50.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster output resolution.")
    parser.add_argument("--log-level", default="INFO", help="Logging level. Default: INFO.")
    return parser.parse_args()


def tool_label(inputs: PerformanceInputs, tool_id: str) -> str:
    return inputs.tool_labels[tool_id]


def tool_color(inputs: PerformanceInputs, tool_id: str) -> str:
    return inputs.tool_colors[tool_id]


def tool_marker(inputs: PerformanceInputs, tool_id: str) -> str:
    marker_by_tool = {
        current_tool_id: TOOL_MARKERS[index % len(TOOL_MARKERS)]
        for index, current_tool_id in enumerate(inputs.tool_ids)
    }
    return marker_by_tool[tool_id]


def sorted_by_rank(frame: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    work = frame[["dataset_id", "gene_id", "is_positive"]].copy()
    work["score"] = scores.astype(float)
    work["gene_id"] = work["gene_id"].astype(str)
    return work.sort_values(
        ["score", "gene_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def recovery_rows(inputs: PerformanceInputs, budgets: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for dataset in inputs.datasets:
        positives_total = int(dataset.frame["is_positive"].sum())
        if positives_total == 0:
            continue
        for tool_id in inputs.tool_ids:
            work = evaluation_frame(dataset.frame, tool_id, inputs.tool_ids, UNIVERSE_FULL)
            scores = rank_series(
                work,
                tool_id,
                dataset_id=dataset.dataset_id,
                universe=UNIVERSE_FULL,
            )
            ordered = sorted_by_rank(work, scores)
            cumulative_tp = ordered["is_positive"].astype(int).cumsum()
            for budget in budgets:
                admitted = min(int(budget), len(ordered))
                positives_recovered = int(cumulative_tp.iloc[admitted - 1]) if admitted else 0
                rows.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "tool_id": tool_id,
                        "predictor": tool_label(inputs, tool_id),
                        "budget": int(budget),
                        "pairs_admitted": admitted,
                        "positives_total": positives_total,
                        "positives_recovered": positives_recovered,
                        "recall": positives_recovered / positives_total,
                    }
                )
    return pd.DataFrame(rows)


def precision_at_recall_threshold(
    frame: pd.DataFrame,
    scores: pd.Series,
    *,
    recall_target: float,
) -> dict[str, float | int | bool]:
    positives_total = int(frame["is_positive"].sum())
    if positives_total == 0:
        return {
            "threshold": np.nan,
            "pairs_admitted": 0,
            "true_positives": 0,
            "false_positives": 0,
            "recall": np.nan,
            "precision": np.nan,
            "false_positives_per_true_positive": np.nan,
            "threshold_includes_missing_block": False,
        }

    work = frame[["is_positive"]].copy()
    work["score"] = scores.astype(float)
    grouped = (
        work.groupby("score", as_index=False)
        .agg(
            pairs=("is_positive", "size"),
            true_positives=("is_positive", "sum"),
        )
        .sort_values("score", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    grouped["pairs_admitted"] = grouped["pairs"].cumsum()
    grouped["true_positives_admitted"] = grouped["true_positives"].cumsum()
    grouped["false_positives_admitted"] = (
        grouped["pairs_admitted"] - grouped["true_positives_admitted"]
    )
    grouped["recall"] = grouped["true_positives_admitted"] / positives_total
    reached = grouped[grouped["recall"] >= float(recall_target)]
    if reached.empty:
        return {
            "threshold": np.nan,
            "pairs_admitted": int(len(work)),
            "true_positives": int(grouped["true_positives_admitted"].iloc[-1]),
            "false_positives": int(grouped["false_positives_admitted"].iloc[-1]),
            "recall": float(grouped["recall"].iloc[-1]),
            "precision": np.nan,
            "false_positives_per_true_positive": np.nan,
            "threshold_includes_missing_block": False,
        }
    selected = reached.iloc[0]
    true_positives = int(selected["true_positives_admitted"])
    false_positives = int(selected["false_positives_admitted"])
    precision = true_positives / int(selected["pairs_admitted"])
    return {
        "threshold": float(selected["score"]),
        "pairs_admitted": int(selected["pairs_admitted"]),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "recall": float(selected["recall"]),
        "precision": precision,
        "false_positives_per_true_positive": false_positives / true_positives if true_positives else np.nan,
        "threshold_includes_missing_block": bool(np.isclose(float(selected["score"]), 0.0)),
    }


def threshold_rows(inputs: PerformanceInputs, recall_target: float) -> pd.DataFrame:
    rows = []
    for dataset in inputs.datasets:
        for tool_id in inputs.tool_ids:
            scores = rank_series(
                dataset.frame,
                tool_id,
                dataset_id=dataset.dataset_id,
                universe=UNIVERSE_FULL,
            )
            row = precision_at_recall_threshold(
                dataset.frame,
                scores,
                recall_target=recall_target,
            )
            rows.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "tool_id": tool_id,
                    "predictor": inputs.tool_labels[tool_id],
                    "recall_target": recall_target,
                    **row,
                }
            )
    return pd.DataFrame(rows)


def best_threshold_rows(thresholds: pd.DataFrame) -> pd.DataFrame:
    ranked = thresholds.sort_values(
        ["dataset_id", "precision", "false_positives_per_true_positive"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return ranked.groupby("dataset_id", as_index=False).head(1).reset_index(drop=True)


def write_tables(
    *,
    recovery: pd.DataFrame,
    thresholds: pd.DataFrame,
    best: pd.DataFrame,
    tables_dir: Path,
) -> dict[str, Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "recovery": tables_dir / "figure6_fps_recovery_by_budget.tsv",
        "precision": tables_dir / "figure6_fps_precision_at_recall.tsv",
        "best": tables_dir / "figure6_fps_best_precision_at_recall.tsv",
    }
    recovery.to_csv(paths["recovery"], sep="\t", index=False)
    thresholds.to_csv(paths["precision"], sep="\t", index=False)
    best.to_csv(paths["best"], sep="\t", index=False)
    return paths


def panel_a(ax: plt.Axes, inputs: PerformanceInputs, recovery: pd.DataFrame) -> None:
    summary = (
        recovery.groupby(["tool_id", "predictor", "budget"], as_index=False)
        .agg(mean_recall=("recall", "mean"), median_recall=("recall", "median"))
    )
    budgets = sorted(int(budget) for budget in summary["budget"].dropna().unique())
    for tool_id in inputs.tool_ids:
        data = summary[summary["tool_id"] == tool_id].sort_values("budget")
        if data.empty:
            continue
        ax.plot(
            data["budget"],
            data["mean_recall"],
            marker=tool_marker(inputs, tool_id),
            linewidth=2.0,
            linestyle="-",
            color=tool_color(inputs, tool_id),
            markeredgecolor="#1F2937",
            markeredgewidth=0.7,
            markersize=6,
            label=tool_label(inputs, tool_id),
        )
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(value):,}"))
    ax.set_xlabel("Prediction budget per experiment", fontsize=LABEL_SIZE)
    ax.set_ylabel("Mean positive-pair recall", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="both", alpha=0.22, linewidth=0.8)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_title("FPS recovery by prediction budget", fontsize=TITLE_SIZE, fontweight="bold")


def ordered_best(best: pd.DataFrame) -> pd.DataFrame:
    return best.sort_values(
        ["precision", "false_positives_per_true_positive", "dataset_id"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def panel_b(ax: plt.Axes, inputs: PerformanceInputs, best: pd.DataFrame, recall_target: float) -> None:
    ordered = ordered_best(best)
    x = np.arange(len(ordered))
    for tool_id in inputs.tool_ids:
        data = ordered[ordered["tool_id"] == tool_id]
        if data.empty:
            continue
        indices = data.index.to_numpy()
        ax.scatter(
            x[indices],
            data["precision"],
            s=70,
            color=tool_color(inputs, tool_id),
            marker=tool_marker(inputs, tool_id),
            edgecolor="#1F2937",
            linewidth=0.75,
            label=tool_label(inputs, tool_id),
        )
    ax.set_ylim(0, max(0.05, float(ordered["precision"].max()) * 1.18))
    ax.set_xticks([])
    ax.set_xlabel("Perturbation experiments", fontsize=LABEL_SIZE)
    ax.set_ylabel(f"Best precision at >= {recall_target:.0%} recall", fontsize=LABEL_SIZE)
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_title("Best precision at 50% recall", fontsize=TITLE_SIZE, fontweight="bold")


def legend_handles(inputs: PerformanceInputs) -> list[plt.Line2D]:
    return [
        plt.Line2D(
            [0],
            [0],
            marker=tool_marker(inputs, tool_id),
            linestyle="-",
            color=tool_color(inputs, tool_id),
            markerfacecolor=tool_color(inputs, tool_id),
            markeredgecolor="#1F2937",
            markeredgewidth=0.75,
            linewidth=2.0,
            markersize=7,
            label=tool_label(inputs, tool_id),
        )
        for tool_id in inputs.tool_ids
    ]


def panel_legend(ax: plt.Axes, inputs: PerformanceInputs) -> None:
    ax.axis("off")
    legend = ax.legend(
        handles=legend_handles(inputs),
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        title="Predictor",
        title_fontsize=LEGEND_TITLE_SIZE,
        handlelength=2.4,
        labelspacing=0.9,
        borderpad=1.0,
    )
    legend.get_title().set_fontweight("bold")


def add_panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.10,
        1.10,
        letter,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )


def finish_data_panel(ax: plt.Axes, letter: str) -> None:
    add_panel_label(ax, letter)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def write_panel_figures(
    *,
    inputs: PerformanceInputs,
    recovery: pd.DataFrame,
    best: pd.DataFrame,
    config: Figure6Config,
) -> dict[str, dict[str, Path]]:
    panel_outputs: dict[str, dict[str, Path]] = {}

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    panel_a(ax, inputs, recovery)
    finish_data_panel(ax, "A")
    fig.tight_layout()
    panel_outputs["A"] = save_figure(
        fig,
        config.out_dir,
        f"{OUTPUT_PREFIX}_panel_a_recovery_budget",
        dpi=config.dpi,
    )

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    panel_b(ax, inputs, best, config.recall_target)
    finish_data_panel(ax, "B")
    fig.tight_layout()
    panel_outputs["B"] = save_figure(
        fig,
        config.out_dir,
        f"{OUTPUT_PREFIX}_panel_b_best_precision",
        dpi=config.dpi,
    )

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    panel_legend(ax, inputs)
    fig.tight_layout()
    panel_outputs["D"] = save_figure(
        fig,
        config.out_dir,
        f"{OUTPUT_PREFIX}_panel_d_legend",
        dpi=config.dpi,
    )
    return panel_outputs


def write_combined_png(
    panel_outputs: dict[str, dict[str, Path]],
    *,
    out_dir: Path,
    dpi: int,
) -> Path:
    required = ("A", "B", "D")
    missing = [panel for panel in required if panel not in panel_outputs or "png" not in panel_outputs[panel]]
    if missing:
        raise ValueError(f"Missing PNG panel outputs: {missing}")

    panel_images = {
        panel: trim_white_border(plt.imread(panel_outputs[panel]["png"]))
        for panel in required
    }
    left_width = max(panel_images[panel].shape[1] for panel in ("A", "B"))
    gap_y = 26
    gap_x = 80
    left_rows = [
        pad_to_shape(
            panel_images[panel],
            (panel_images[panel].shape[0], left_width),
            vertical="center",
            horizontal="center",
        )
        for panel in ("A", "B")
    ]
    channels = left_rows[0].shape[2]
    dtype = left_rows[0].dtype
    left_column = np.vstack(
        [
            left_rows[0],
            np.ones((gap_y, left_width, channels), dtype=dtype),
            left_rows[1],
        ]
    )
    legend_column = pad_to_shape(
        panel_images["D"],
        (left_column.shape[0], panel_images["D"].shape[1]),
        vertical="center",
        horizontal="center",
    )
    combined = np.hstack(
        [
            left_column,
            np.ones((left_column.shape[0], gap_x, channels), dtype=dtype),
            legend_column,
        ]
    )
    fig_width = 14.5
    fig_height = fig_width * combined.shape[0] / combined.shape[1]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(combined)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{OUTPUT_PREFIX}_combined.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def write_combined_svg(
    panel_outputs: dict[str, dict[str, Path]],
    *,
    out_dir: Path,
) -> Path:
    required = ("A", "B", "D")
    missing = [panel for panel in required if panel not in panel_outputs or "svg" not in panel_outputs[panel]]
    if missing:
        raise ValueError(f"Missing SVG panel outputs: {missing}")

    ET.register_namespace("", SVG_NS)
    panel_svgs = {panel: parse_svg_viewbox(panel_outputs[panel]["svg"]) for panel in required}
    left_width = max(panel_svgs[panel][0] for panel in ("A", "B"))
    gap_y = 12.0
    gap_x = 22.0
    left_height = panel_svgs["A"][1] + gap_y + panel_svgs["B"][1]
    combined_width = left_width + gap_x + panel_svgs["D"][0]
    combined_height = max(left_height, panel_svgs["D"][1])
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
        {
            "x": "0",
            "y": "0",
            "width": f"{combined_width:g}",
            "height": f"{combined_height:g}",
            "fill": "#ffffff",
        },
    )
    placements = {}
    y_offset = (combined_height - left_height) / 2
    for panel in ("A", "B"):
        panel_width, panel_height, _panel_root = panel_svgs[panel]
        placements[panel] = ((left_width - panel_width) / 2, y_offset)
        y_offset += panel_height + gap_y
    placements["D"] = (
        left_width + gap_x,
        (combined_height - panel_svgs["D"][1]) / 2,
    )
    for panel in required:
        panel_width, panel_height, panel_root = panel_svgs[panel]
        x, y = placements[panel]
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
    svg_path = out_dir / f"{OUTPUT_PREFIX}_combined.svg"
    ET.ElementTree(root).write(svg_path, encoding="utf-8", xml_declaration=True)
    svg_text = "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
    svg_path.write_text(f"{svg_text}\n", encoding="utf-8")
    return svg_path


def write_combined_figure(
    panel_outputs: dict[str, dict[str, Path]],
    *,
    out_dir: Path,
    dpi: int,
) -> dict[str, Path]:
    return {
        "png": write_combined_png(panel_outputs, out_dir=out_dir, dpi=dpi),
        "svg": write_combined_svg(panel_outputs, out_dir=out_dir),
    }


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    run_dir = (
        args.run_dir.expanduser()
        if args.run_dir is not None
        else find_latest_completed_run(args.results_dir)
    )
    fdr_threshold, effect_threshold = load_run_thresholds(run_dir)
    inputs = prepare_inputs(
        run_dir=run_dir,
        metadata_path=args.metadata,
        fdr_threshold=fdr_threshold,
        effect_threshold=effect_threshold,
    )
    config = Figure6Config(
        run_dir=run_dir,
        metadata_path=args.metadata,
        out_dir=args.out_dir,
        tables_dir=args.tables_dir,
        budgets=tuple(sorted(set(int(budget) for budget in args.budgets))),
        recall_target=float(args.recall_target),
        dpi=int(args.dpi),
    )
    logger.info("Using run directory: %s", run_dir)
    logger.info("Using GT thresholds: fdr=%s effect=%s", fdr_threshold, effect_threshold)
    logger.info("Datasets: %d", len(inputs.datasets))
    recovery = recovery_rows(inputs, config.budgets)
    thresholds = threshold_rows(inputs, config.recall_target)
    best = best_threshold_rows(thresholds)
    table_paths = write_tables(
        recovery=recovery,
        thresholds=thresholds,
        best=best,
        tables_dir=config.tables_dir,
    )
    panel_outputs = write_panel_figures(
        inputs=inputs,
        recovery=recovery,
        best=best,
        config=config,
    )
    figure_paths = write_combined_figure(
        panel_outputs,
        out_dir=config.out_dir,
        dpi=config.dpi,
    )
    panel_paths = [
        path
        for panel_output in panel_outputs.values()
        for path in panel_output.values()
    ]
    for path in [*table_paths.values(), *panel_paths, *figure_paths.values()]:
        logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as error:
        logger.error("%s", error)
        raise SystemExit(1) from None
