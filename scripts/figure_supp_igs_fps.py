#!/usr/bin/env python3
"""Generate supplementary IGS-restricted Full Pair Set performance outputs."""

from __future__ import annotations

from pathlib import Path

from figure_performance_common import (
    GENE_SET_FILTER_IGS,
    FigurePerformanceConfig,
    UNIVERSE_FULL,
    run_performance_figure,
)


CONFIG = FigurePerformanceConfig(
    figure_id="Supplementary IGS-restricted FPS",
    universe=UNIVERSE_FULL,
    title="IGS-restricted full pair set performance",
    output_prefix="supplement_igs_restricted_fps",
    include_random=True,
    default_out_dir=Path("manuscript_assets/supplement_igs_restricted_fps"),
    leaderboard_mode="winner_counts",
    include_random_in_leaderboard=False,
    gene_set_filter=GENE_SET_FILTER_IGS,
    panel_specs=(
        ("A", "aps", "boxplot"),
        ("B", "top_n_median_effect", "boxplot"),
        ("C", "auroc", "boxplot"),
        ("D", "spearman", "boxplot"),
        ("E", "winner_summary", "leaderboard"),
        ("F", "legend", "legend"),
    ),
    metric_ylim={"aps": (0.0, 0.6)},
)


if __name__ == "__main__":
    run_performance_figure(CONFIG)
