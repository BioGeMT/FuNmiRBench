#!/usr/bin/env python3
"""Generate Figure 5 Full Pair Set performance outputs."""

from __future__ import annotations

from figure_performance_common import (
    FigurePerformanceConfig,
    UNIVERSE_FULL,
    run_performance_figure,
)


CONFIG = FigurePerformanceConfig(
    figure_id="Figure 5",
    universe=UNIVERSE_FULL,
    title="Cross-dataset full pair set predictor distributions",
    output_prefix="figure5_fps",
    include_random=True,
)


if __name__ == "__main__":
    run_performance_figure(CONFIG)
