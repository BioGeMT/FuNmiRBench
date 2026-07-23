#!/usr/bin/env python3
"""Assemble per-dataset top-prediction effect CDFs into a supplement figure."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from funmirbench.evaluate_common import TOP_PREDICTION_CDF_N
from funmirbench.logger import setup_logging


logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_MANUSCRIPT_OUTPUT_DIR = Path("manuscript_assets/supplement")
DEFAULT_MANUSCRIPT_TABLES_DIR = Path("manuscript_assets/tables")
DEFAULT_FORMATS = ("png", "svg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-dataset top-prediction effect CDF plots from a completed "
            "FuNmiRBench run into one supplementary manuscript figure."
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
        default=DEFAULT_MANUSCRIPT_OUTPUT_DIR,
        help="Output directory for the combined supplementary figure.",
    )
    parser.add_argument(
        "--stem",
        default=f"supplement_top_{TOP_PREDICTION_CDF_N}_effect_cdfs",
        help="Output filename stem.",
    )
    parser.add_argument("--columns", type=int, default=3, help="Panel columns. Default: 3.")
    parser.add_argument(
        "--plots-per-part",
        type=int,
        default=15,
        help="Number of dataset plots per supplement part. Default: 15.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_MANUSCRIPT_TABLES_DIR,
        help="Output directory for the panel manifest TSV.",
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


def ordered_dataset_ids(run_dir: Path) -> list[str]:
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        dataset_ids = summary.get("dataset_ids")
        if dataset_ids:
            return [str(dataset_id) for dataset_id in dataset_ids]
    return sorted(path.name for path in (run_dir / "datasets").iterdir() if path.is_dir())


def cdf_plot_path(run_dir: Path, dataset_id: str) -> Path:
    return (
        run_dir
        / "datasets"
        / dataset_id
        / "plots"
        / "comparisons"
        / f"top_{TOP_PREDICTION_CDF_N}_effect_cdfs.png"
    )


def trim_white_border(image: np.ndarray, *, threshold: float = 0.985, pad: int = 8) -> np.ndarray:
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


def write_combined_cdf_figure(
    *,
    images: list[tuple[str, np.ndarray]],
    out_dir: Path,
    stem: str,
    columns: int,
    dpi: int,
) -> list[Path]:
    n_panels = len(images)
    ncols = max(1, int(columns))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 4.8, nrows * 3.5),
        squeeze=False,
    )
    for index, (_dataset_id, image) in enumerate(images):
        ax = axes[index // ncols][index % ncols]
        ax.imshow(image)
        ax.axis("off")
    for index in range(n_panels, nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.015, hspace=0.04)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for suffix in DEFAULT_FORMATS:
        out_path = out_dir / f"{stem}.{suffix}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        output_paths.append(out_path)
    plt.close(fig)
    return output_paths


def write_supplement_parts(
    *,
    run_dir: Path,
    out_dir: Path,
    tables_dir: Path,
    stem: str,
    columns: int,
    plots_per_part: int,
    dpi: int,
) -> tuple[list[Path], Path]:
    dataset_ids = ordered_dataset_ids(run_dir)
    paths = [(dataset_id, cdf_plot_path(run_dir, dataset_id)) for dataset_id in dataset_ids]
    missing = [str(path) for _, path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing per-dataset CDF plots. Regenerate the benchmark with "
            "evaluation.write_top_prediction_cdfs enabled first. Missing examples: "
            + "; ".join(missing[:5])
        )

    images = [
        (dataset_id, trim_white_border(plt.imread(path)))
        for dataset_id, path in paths
    ]
    plots_per_part = max(1, int(plots_per_part))
    part_count = int(np.ceil(len(images) / plots_per_part))
    output_paths = []
    manifest_rows = []
    for part_index in range(part_count):
        start = part_index * plots_per_part
        end = min(start + plots_per_part, len(images))
        part_images = images[start:end]
        part_stem = f"{stem}_part{part_index + 1}"
        output_paths.extend(
            write_combined_cdf_figure(
                images=part_images,
                out_dir=out_dir,
                stem=part_stem,
                columns=columns,
                dpi=dpi,
            )
        )
        for local_index, (dataset_id, _image) in enumerate(part_images, start=1):
            global_index = start + local_index
            manifest_rows.append(
                {
                    "part": part_index + 1,
                    "panel_index": local_index,
                    "global_panel_index": global_index,
                    "dataset_id": dataset_id,
                    "cdf_plot_path": str(cdf_plot_path(run_dir, dataset_id)),
                }
            )

    tables_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = tables_dir / f"{stem}_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write("part\tpanel_index\tglobal_panel_index\tdataset_id\tcdf_plot_path\n")
        for row in manifest_rows:
            handle.write(
                f"{row['part']}\t{row['panel_index']}\t{row['global_panel_index']}\t"
                f"{row['dataset_id']}\t{row['cdf_plot_path']}\n"
            )
    return output_paths, manifest_path


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    run_dir = (
        args.run_dir.expanduser()
        if args.run_dir is not None
        else find_latest_completed_run(args.results_dir)
    )
    logger.info("Using run directory: %s", run_dir)
    output_paths, manifest_path = write_supplement_parts(
        run_dir=run_dir,
        out_dir=args.out_dir,
        tables_dir=args.tables_dir,
        stem=args.stem,
        columns=args.columns,
        plots_per_part=args.plots_per_part,
        dpi=args.dpi,
    )
    for path in output_paths:
        logger.info("Wrote %s", path)
    logger.info("Wrote %s", manifest_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as error:
        logger.error("%s", error)
        raise SystemExit(1) from None
