#!/usr/bin/env python3
"""Generate Supplementary Figure 2 rank-distribution manuscript assets."""

from __future__ import annotations

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_performance_common import (
    DEFAULT_RESULTS_DIR,
    SVG_NS,
    find_latest_completed_run,
    pad_to_shape,
    parse_svg_viewbox,
    trim_white_border,
)
from funmirbench.logger import setup_logging


logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path("manuscript_assets/supplement")
OUTPUT_PREFIX = "supplement_figure2_rank_distributions"
PANEL_SPECS = (
    (
        "A",
        "positive_background_local_rank_distributions",
        "Rank within dataset",
    ),
    (
        "B",
        "positive_background_global_rank_distributions",
        "Rank across predictor file",
    ),
)
PANEL_LABEL_SIZE = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble combined local/global positive-background rank-distribution "
            "plots from a completed FuNmiRBench run into Supplementary Figure 2."
        )
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
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for Supplementary Figure 2 assets.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster output resolution.")
    parser.add_argument("--log-level", default="INFO", help="Logging level. Default: INFO.")
    return parser.parse_args()


def panel_path(run_dir: Path, stem: str, suffix: str) -> Path:
    return run_dir / "plots" / "combined" / "ranks" / f"{stem}.{suffix}"


def validate_inputs(run_dir: Path) -> None:
    missing = [
        str(panel_path(run_dir, stem, suffix))
        for _label, stem, _title in PANEL_SPECS
        for suffix in ("png", "svg")
        if not panel_path(run_dir, stem, suffix).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing rank-distribution plots. Regenerate the benchmark report first. "
            "Missing examples: " + "; ".join(missing[:4])
        )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
        color="black",
    )


def write_png_panels(run_dir: Path, out_dir: Path, *, dpi: int) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for label, stem, title in PANEL_SPECS:
        image = trim_white_border(plt.imread(panel_path(run_dir, stem, "png")), pad=10)
        fig_width = 7.2
        fig_height = fig_width * image.shape[0] / image.shape[1]
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.imshow(image)
        ax.axis("off")
        fig.text(
            0.01,
            0.985,
            label,
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
            va="top",
            ha="left",
        )
        fig.text(
            0.5,
            0.985,
            title,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="center",
        )
        fig.subplots_adjust(left=0, right=1, top=0.90, bottom=0)
        out_path = out_dir / f"{OUTPUT_PREFIX}_panel_{label.lower()}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        outputs[label] = out_path
    return outputs


def write_combined_png(panel_pngs: dict[str, Path], out_dir: Path, *, dpi: int) -> Path:
    images = {
        label: plt.imread(path)
        for label, path in panel_pngs.items()
    }
    target_height = max(image.shape[0] for image in images.values())
    target_width = max(image.shape[1] for image in images.values())
    cells = [
        pad_to_shape(images[label], (target_height, target_width), vertical="top")
        for label, _stem, _title in PANEL_SPECS
    ]
    channels = cells[0].shape[2]
    dtype = cells[0].dtype
    gap = np.ones((target_height, 60, channels), dtype=dtype)
    combined = np.hstack([cells[0], gap, cells[1]])
    fig_width = 14.0
    fig_height = fig_width * combined.shape[0] / combined.shape[1]
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(combined)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out_path = out_dir / f"{OUTPUT_PREFIX}_combined.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def write_svg_panel(run_dir: Path, out_dir: Path, label: str, stem: str) -> Path:
    source = panel_path(run_dir, stem, "svg")
    out_path = out_dir / f"{OUTPUT_PREFIX}_panel_{label.lower()}.svg"
    out_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return out_path


def write_combined_svg(run_dir: Path, out_dir: Path) -> Path:
    ET.register_namespace("", SVG_NS)
    panel_svgs = {
        label: parse_svg_viewbox(panel_path(run_dir, stem, "svg"))
        for label, stem, _title in PANEL_SPECS
    }
    gap_x = 24.0
    label_pad = 26.0
    title_pad = 24.0
    panel_widths = [panel_svgs[label][0] for label, _stem, _title in PANEL_SPECS]
    panel_heights = [panel_svgs[label][1] for label, _stem, _title in PANEL_SPECS]
    combined_width = sum(panel_widths) + gap_x
    combined_height = max(panel_heights) + label_pad + title_pad
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
    x = 0.0
    for label, _stem, title in PANEL_SPECS:
        width, height, panel_root = panel_svgs[label]
        ET.SubElement(
            root,
            f"{{{SVG_NS}}}text",
            {
                "x": f"{x + 4:g}",
                "y": "18",
                "font-size": "18",
                "font-weight": "700",
                "font-family": "DejaVu Sans, Arial, sans-serif",
                "fill": "#000000",
            },
        ).text = label
        ET.SubElement(
            root,
            f"{{{SVG_NS}}}text",
            {
                "x": f"{x + width / 2:g}",
                "y": "22",
                "font-size": "14",
                "font-weight": "700",
                "font-family": "DejaVu Sans, Arial, sans-serif",
                "text-anchor": "middle",
                "fill": "#000000",
            },
        ).text = title
        nested = ET.SubElement(
            root,
            f"{{{SVG_NS}}}svg",
            {
                "x": f"{x:g}",
                "y": f"{label_pad + title_pad:g}",
                "width": f"{width:g}",
                "height": f"{height:g}",
                "viewBox": panel_root.attrib["viewBox"],
            },
        )
        for child in list(panel_root):
            nested.append(child)
        x += width + gap_x

    out_path = out_dir / f"{OUTPUT_PREFIX}_combined.svg"
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=True)
    svg_text = "\n".join(line.rstrip() for line in out_path.read_text(encoding="utf-8").splitlines())
    out_path.write_text(f"{svg_text}\n", encoding="utf-8")
    return out_path


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    run_dir = (
        args.run_dir.expanduser()
        if args.run_dir is not None
        else find_latest_completed_run(args.results_dir)
    )
    validate_inputs(run_dir)
    logger.info("Using run directory: %s", run_dir)
    png_panels = write_png_panels(run_dir, args.out_dir, dpi=args.dpi)
    svg_panels = {
        label: write_svg_panel(run_dir, args.out_dir, label, stem)
        for label, stem, _title in PANEL_SPECS
    }
    combined_png = write_combined_png(png_panels, args.out_dir, dpi=args.dpi)
    combined_svg = write_combined_svg(run_dir, args.out_dir)
    for path in [*png_panels.values(), *svg_panels.values(), combined_png, combined_svg]:
        logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as error:
        logger.error("%s", error)
        raise SystemExit(1) from None
