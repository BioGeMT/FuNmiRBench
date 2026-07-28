#!/usr/bin/env python3
"""Generate Figure 4 gene-set intersection performance outputs."""

from __future__ import annotations

from pathlib import Path

from figure_performance_common import (
    FigurePerformanceConfig,
    UNIVERSE_INTERSECTION,
    run_performance_figure,
)


CONFIG = FigurePerformanceConfig(
    figure_id="Figure 4 gene set",
    universe=UNIVERSE_INTERSECTION,
    title="Intersection gene-set performance",
    output_prefix="figure4_ips_gene_set",
    include_random=True,
    default_out_dir=Path("manuscript_assets/figure4_gene_set"),
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
