#!/usr/bin/env python3
"""Generate Figure 3 algorithm-specific performance outputs."""

from __future__ import annotations

from pathlib import Path

from figure_performance_common import (
    FigurePerformanceConfig,
    UNIVERSE_ALGORITHM_SPECIFIC,
    run_performance_figure,
)


CONFIG = FigurePerformanceConfig(
    figure_id="Figure 3",
    universe=UNIVERSE_ALGORITHM_SPECIFIC,
    title="Algorithm-specific pair performance",
    output_prefix="figure3_algorithm_specific",
    include_random=True,
    default_out_dir=Path("manuscript_assets/figure3"),
)


if __name__ == "__main__":
    run_performance_figure(CONFIG)
