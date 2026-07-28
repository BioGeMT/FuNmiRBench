#!/usr/bin/env python3
"""Generate Figure 5 Full Pair Set performance outputs."""

from __future__ import annotations

from pathlib import Path

from figure_performance_common import (
    FigurePerformanceConfig,
    UNIVERSE_FULL,
    run_performance_figure,
)


CONFIG = FigurePerformanceConfig(
    figure_id="Figure 5",
    universe=UNIVERSE_FULL,
    title="Full pair set performance",
    output_prefix="figure5_fps",
    include_random=True,
    default_out_dir=Path("manuscript_assets/figure5"),
    leaderboard_mode="winner_counts",
    include_random_in_leaderboard=False,
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
