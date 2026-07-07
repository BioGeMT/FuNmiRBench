"""Build manuscript assets with Figure 2 recovery curves extended to top 500.

This wrapper reuses ``build_manuscript_figures_tables.py`` but overrides the
fixed-budget recovery default from 300 to 500 predictions per dataset.

Example
-------
python scripts/build_manuscript_figures_tables_top500.py \
    --report-dir results/20260706_132519 \
    --out-dir manuscript_assets
"""

from __future__ import annotations

import build_manuscript_figures_tables as manuscript_assets

_original_recovery_table = manuscript_assets.recovery_table


def recovery_table(report_dir, *, fdr: float, effect_threshold: float, max_predictions: int = 500):
    """Return fixed-budget recovery curves up to top 500 predictions by default."""
    return _original_recovery_table(
        report_dir,
        fdr=fdr,
        effect_threshold=effect_threshold,
        max_predictions=max_predictions,
    )


manuscript_assets.recovery_table = recovery_table


if __name__ == "__main__":
    manuscript_assets.main()
