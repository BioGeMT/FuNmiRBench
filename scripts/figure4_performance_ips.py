#!/usr/bin/env python3
"""Generate Figure 4 Intersection Pair Set performance outputs."""

from __future__ import annotations

from pathlib import Path

from figure_performance_common import (
    FigurePerformanceConfig,
    UNIVERSE_INTERSECTION,
    run_performance_figure,
)


CONFIG = FigurePerformanceConfig(
    figure_id="Figure 4",
    universe=UNIVERSE_INTERSECTION,
    title="Intersection pair set performance",
    output_prefix="figure4_ips",
    include_random=True,
    default_out_dir=Path("manuscript_assets/figure4"),
)


if __name__ == "__main__":
    run_performance_figure(CONFIG)
