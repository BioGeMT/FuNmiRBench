#!/usr/bin/env python3
"""Generate Figure 3 gene-set algorithm-specific performance outputs."""

from __future__ import annotations

from pathlib import Path

from figure_performance_common import (
    FigurePerformanceConfig,
    UNIVERSE_ALGORITHM_SPECIFIC,
    run_performance_figure,
)


CONFIG = FigurePerformanceConfig(
    figure_id="Figure 3 gene set",
    universe=UNIVERSE_ALGORITHM_SPECIFIC,
    title="Algorithm-specific gene-set performance",
    output_prefix="figure3_algorithm_specific_gene_set",
    include_random=False,
    default_out_dir=Path("manuscript_assets/figure3_gene_set"),
    leaderboard_mode="winner_counts",
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
