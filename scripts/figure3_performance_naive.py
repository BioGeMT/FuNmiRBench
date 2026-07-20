#!/usr/bin/env python3
"""Generate Figure 3 algorithm-specific performance outputs."""

from __future__ import annotations

from figure_performance_common import (
    FigurePerformanceConfig,
    UNIVERSE_ALGORITHM_SPECIFIC,
    run_performance_figure,
)


CONFIG = FigurePerformanceConfig(
    figure_id="Figure 3",
    universe=UNIVERSE_ALGORITHM_SPECIFIC,
    title="Cross-dataset algorithm-specific predictor distributions",
    output_prefix="figure3_algorithm_specific",
    include_random=False,
)


if __name__ == "__main__":
    run_performance_figure(CONFIG)
