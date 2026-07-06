"""Create manuscript-ready figures and tables from a FuNmiRBench report.

This script is intentionally separated from the main benchmark run.  It consumes
an already generated FuNmiRBench report directory and writes only the figures and
TSV tables needed for the functional miRNA predictor manuscript.

Example
-------
python scripts/build_manuscript_figures_tables.py \
    --report-dir results/20260703_115539 \
    --out-dir manuscript_assets

Outputs
-------
figures/figure1_cross_dataset_distributions.png
figures/figure2_rank_enrichment_recovery.png
figures/figure3_targetscan_centered.png
figures/supplementary_mirdb_centered.png
tables/table1_cross_dataset_predictor_summary.tsv
tables/table_s1_detailed_predictor_summary.tsv
tables/table_s2_targetscan_centered_bins.tsv
tables/table_s3_mirdb_centered_bins.tsv
"""

from __future__ import annotations

import argparse
import math
import pathlib
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


TOOL_IDS = ["targetscan", "mirdb_mirtarget", "microt_cnn", "mirbind2", "miraw"]
TOOL_LABELS = {
    "targetscan": "TargetScan",
    "mirdb_mirtarget": "miRDB",
    "microt_cnn": "microT-CNN",
    "mirbind2": "miRBind2",
    "miraw": "miRAW",
}
METRIC_LABELS = {
    "coverage": "Coverage",
    "positive_coverage": "Positive coverage",
    "aps": "APS",
    "auroc": "AUROC",
}


def _style_axes(ax, *, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, alpha=0.25)
    ax.set_axisbelow(True)


def _save(fig, path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _candidate_tables_dirs(report_dir: pathlib.Path) -> list[pathlib.Path]:
    return [
        report_dir / "tables" / "combined",
        report_dir / "tables",
        report_dir,
    ]


def _find_table(report_dir: pathlib.Path, filename: str) -> pathlib.Path | None:
    for base in _candidate_tables_dirs(report_dir):
        path = base / filename
        if path.exists():
            return path
    matches = sorted(report_dir.glob(f"**/{filename}"))
    return matches[0] if matches else None


def _read_required_table(report_dir: pathlib.Path, filename: str) -> pd.DataFrame:
    path = _find_table(report_dir, filename)
    if path is None:
        raise FileNotFoundError(f"Could not find {filename} below {report_dir}")
    return pd.read_csv(path, sep="\t")


def _metric_long_from_wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    id_cols = [col for col in df.columns if col not in TOOL_IDS]
    long = df.melt(id_vars=id_cols, value_vars=[col for col in TOOL_IDS if col in df.columns], var_name="tool_id", value_name=metric)
    long[metric] = pd.to_numeric(long[metric], errors="coerce")
    return long.dropna(subset=[metric])


def _load_metric_rows(report_dir: pathlib.Path) -> pd.DataFrame:
    rows = []
    for metric in METRIC_LABELS:
        filename = f"{metric}_per_experiment.tsv"
        try:
            wide = _read_required_table(report_dir, filename)
        except FileNotFoundError:
            continue
        long = _metric_long_from_wide(wide, metric)[["tool_id", metric]].copy()
        long["metric"] = metric
        long = long.rename(columns={metric: "value"})
        rows.append(long)
    if not rows:
        raise FileNotFoundError("No per-experiment metric tables were found.")
    return pd.concat(rows, ignore_index=True)


def _write_predictor_summary_tables(metric_rows: pd.DataFrame, out_tables: pathlib.Path) -> dict[str, pathlib.Path]:
    summary = (
        metric_rows.groupby(["tool_id", "metric"])["value"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    wide = summary.pivot(index="tool_id", columns="metric")
    wide.columns = [f"{metric}_{stat}" for stat, metric in wide.columns]
    wide = wide.reset_index()
    wide.insert(1, "predictor", wide["tool_id"].map(TOOL_LABELS).fillna(wide["tool_id"]))
    table_s1 = out_tables / "table_s1_detailed_predictor_summary.tsv"
    table_s1.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(table_s1, sep="\t", index=False)

    keep_cols = ["tool_id", "predictor"]
    for metric in ("coverage", "positive_coverage", "aps", "auroc"):
        for stat in ("mean", "median"):
            col = f"{metric}_{stat}"
            if col in wide.columns:
                keep_cols.append(col)
    table1 = out_tables / "table1_cross_dataset_predictor_summary.tsv"
    wide[keep_cols].to_csv(table1, sep="\t", index=False)
    return {"table1": table1, "table_s1": table_s1}


def plot_cross_dataset_distributions(metric_rows: pd.DataFrame, out_path: pathlib.Path) -> pathlib.Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))
    axes = axes.ravel()
    for ax, metric in zip(axes, METRIC_LABELS):
        sub = metric_rows[metric_rows["metric"] == metric]
        data = []
        labels = []
        for tool_id in TOOL_IDS:
            values = sub.loc[sub["tool_id"] == tool_id, "value"].dropna().astype(float).tolist()
            if not values:
                continue
            data.append(values)
            labels.append(TOOL_LABELS.get(tool_id, tool_id))
        ax.boxplot(data, patch_artist=True, showfliers=False)
        for i, values in enumerate(data, start=1):
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(np.full(len(values), i) + jitter, values, s=18, alpha=0.65, edgecolors="white", linewidths=0.3)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0) if metric in {"coverage", "positive_coverage"} else ax.yaxis.get_major_formatter())
        _style_axes(ax)
    fig.suptitle("Cross-dataset predictor distributions", fontsize=15, y=0.995)
    fig.tight_layout()
    return _save(fig, out_path)


def _expected_effect_from_logfc(frame: pd.DataFrame) -> pd.Series:
    logfc = pd.to_numeric(frame["logFC"], errors="coerce")
    perturbation = frame.get("perturbation", pd.Series("", index=frame.index)).astype(str).str.upper().fillna("")
    effect = logfc.abs().copy()
    effect.loc[perturbation.eq("OE") | perturbation.eq("OVEREXPRESSION")] = -logfc.loc[perturbation.eq("OE") | perturbation.eq("OVEREXPRESSION")]
    kdko = perturbation.isin(["KO", "KD", "KNOCKOUT", "KNOCKDOWN"])
    effect.loc[kdko] = logfc.loc[kdko]
    return effect


def _rank_scale(scores: pd.Series) -> pd.Series:
    values = pd.to_numeric(scores, errors="coerce")
    ranks = values.rank(method="dense", ascending=True)
    max_rank = ranks.max(skipna=True)
    if pd.isna(max_rank):
        return pd.Series(np.nan, index=scores.index)
    if float(max_rank) <= 1.0:
        return pd.Series(1.0, index=scores.index, dtype=float)
    return (ranks - 1.0) / (float(max_rank) - 1.0)


def _rank_values(work: pd.DataFrame, tool_id: str, rank_type: str) -> pd.Series | None:
    rank_col = f"{rank_type}_rank_{tool_id}"
    score_col = f"score_{tool_id}"
    if rank_col in work.columns:
        return pd.to_numeric(work[rank_col], errors="coerce")
    if score_col in work.columns:
        return _rank_scale(work[score_col])
    return None


def _joined_paths(report_dir: pathlib.Path) -> list[pathlib.Path]:
    paths = sorted(report_dir.glob("**/joined.tsv"))
    if not paths:
        raise FileNotFoundError(f"No joined.tsv files found below {report_dir}")
    return paths


def _load_rank_rows(report_dir: pathlib.Path, *, rank_type: str, fdr: float, effect_threshold: float) -> pd.DataFrame:
    rows = []
    for joined_path in _joined_paths(report_dir):
        frame = pd.read_csv(joined_path, sep="\t")
        if not {"logFC", "FDR"}.issubset(frame.columns):
            continue
        frame["logFC_num"] = pd.to_numeric(frame["logFC"], errors="coerce")
        frame["FDR_num"] = pd.to_numeric(frame["FDR"], errors="coerce")
        usable = frame["logFC_num"].notna() & frame["FDR_num"].notna() & (frame["FDR_num"] >= 0.0) & (frame["FDR_num"] <= 1.0)
        work = frame.loc[usable].copy()
        if work.empty:
            continue
        work["expected_effect"] = _expected_effect_from_logfc(work)
        work["is_positive"] = (work["FDR_num"] < fdr) & (work["expected_effect"] > effect_threshold)
        for tool_id in TOOL_IDS:
            values = _rank_values(work, tool_id, rank_type)
            if values is None:
                continue
            tmp = pd.DataFrame({"tool_id": tool_id, "rank_value": values, "is_positive": work["is_positive"].astype(bool)})
            tmp = tmp.dropna(subset=["rank_value"])
            if not tmp.empty:
                rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["tool_id", "rank_value", "is_positive"])


def summarize_rank_fraction(rank_df: pd.DataFrame, *, n_bins: int, rank_type: str) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for tool_id in TOOL_IDS:
        sub = rank_df[rank_df["tool_id"] == tool_id]
        overall = float(sub["is_positive"].mean()) if len(sub) else np.nan
        for idx, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
            mask = (sub["rank_value"] >= left) & ((sub["rank_value"] <= right) if idx == n_bins - 1 else (sub["rank_value"] < right))
            in_bin = sub.loc[mask]
            total = int(len(in_bin))
            positives = int(in_bin["is_positive"].sum())
            rows.append({
                "rank_type": rank_type,
                "tool_id": tool_id,
                "predictor": TOOL_LABELS[tool_id],
                "rank_bin_start": left,
                "rank_bin_end": right,
                "rank_bin_mid": (left + right) / 2.0,
                "total_count": total,
                "positive_count": positives,
                "background_count": total - positives,
                "positive_fraction_within_bin": positives / total if total else np.nan,
                "overall_positive_fraction": overall,
            })
    return pd.DataFrame(rows)


def plot_rank_fraction(ax, summary: pd.DataFrame, *, title: str, xlabel: str) -> None:
    for tool_id in TOOL_IDS:
        sub = summary[summary["tool_id"] == tool_id].sort_values("rank_bin_mid")
        ax.plot(sub["rank_bin_mid"], sub["positive_fraction_within_bin"] * 100, marker="o", linewidth=2, label=TOOL_LABELS[tool_id])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("GT positives within bin (%)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    _style_axes(ax, grid_axis="both")


def _recovery_curves(report_dir: pathlib.Path, *, fdr: float, effect_threshold: float, max_predictions: int = 300) -> pd.DataFrame:
    curves = {tool_id: [] for tool_id in TOOL_IDS}
    for joined_path in _joined_paths(report_dir):
        frame = pd.read_csv(joined_path, sep="\t")
        if not {"gene_id", "logFC", "FDR"}.issubset(frame.columns):
            continue
        frame["logFC_num"] = pd.to_numeric(frame["logFC"], errors="coerce")
        frame["FDR_num"] = pd.to_numeric(frame["FDR"], errors="coerce")
        usable = frame["logFC_num"].notna() & frame["FDR_num"].notna() & (frame["FDR_num"] >= 0.0) & (frame["FDR_num"] <= 1.0)
        work = frame.loc[usable].copy()
        if work.empty:
            continue
        work["expected_effect"] = _expected_effect_from_logfc(work)
        work["is_positive"] = ((work["FDR_num"] < fdr) & (work["expected_effect"] > effect_threshold)).astype(int)
        positive_total = int(work["is_positive"].sum())
        if positive_total <= 0:
            continue
        for tool_id in TOOL_IDS:
            values = _rank_values(work, tool_id, "local") or _rank_values(work, tool_id, "global")
            if values is None:
                continue
            scored = work.loc[values.notna(), ["gene_id", "is_positive"]].copy()
            if scored.empty:
                continue
            scored["rank_value"] = values.loc[scored.index].astype(float)
            scored = scored.sort_values(["rank_value", "gene_id"], ascending=[False, True], kind="mergesort")
            cumulative = np.cumsum(scored["is_positive"].to_numpy(dtype=int)).astype(float) / positive_total
            curve = np.repeat(float(cumulative[-1]), max_predictions)
            observed = min(max_predictions, cumulative.size)
            curve[:observed] = cumulative[:observed]
            curves[tool_id].append(curve)
    rows = []
    for tool_id, items in curves.items():
        if not items:
            continue
        mean_curve = np.vstack(items).mean(axis=0)
        rows.extend({"tool_id": tool_id, "prediction_count": i + 1, "recovery_fraction": value} for i, value in enumerate(mean_curve))
    return pd.DataFrame(rows)


def plot_figure2(report_dir: pathlib.Path, out_figures: pathlib.Path, out_tables: pathlib.Path, *, fdr: float, effect_threshold: float) -> pathlib.Path:
    local = summarize_rank_fraction(_load_rank_rows(report_dir, rank_type="local", fdr=fdr, effect_threshold=effect_threshold), n_bins=10, rank_type="local")
    global_ = summarize_rank_fraction(_load_rank_rows(report_dir, rank_type="global", fdr=fdr, effect_threshold=effect_threshold), n_bins=5, rank_type="global")
    local.to_csv(out_tables / "table_s4_local_rank_bins.tsv", sep="\t", index=False)
    global_.to_csv(out_tables / "table_s5_global_rank_bins.tsv", sep="\t", index=False)
    recovery = _recovery_curves(report_dir, fdr=fdr, effect_threshold=effect_threshold)
    recovery.to_csv(out_tables / "table_s6_recovery_curves.tsv", sep="\t", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))
    plot_rank_fraction(axes[0], local, title="Local rank enrichment", xlabel="Local rank bin")
    plot_rank_fraction(axes[1], global_, title="Global rank enrichment", xlabel="Global rank bin")
    for tool_id in TOOL_IDS:
        sub = recovery[recovery["tool_id"] == tool_id]
        if not sub.empty:
            axes[2].plot(sub["prediction_count"], sub["recovery_fraction"] * 100, linewidth=2, label=TOOL_LABELS[tool_id])
    axes[2].set_title("GT-positive recovery")
    axes[2].set_xlabel("Predicted targets per dataset")
    axes[2].set_ylabel("Mean GT-positive recovery (%)")
    axes[2].set_ylim(bottom=0)
    _style_axes(axes[2], grid_axis="both")
    axes[2].legend(frameon=False, fontsize=9)
    fig.suptitle("Functional target enrichment and recovery across predictor ranks", fontsize=15, y=1.02)
    fig.tight_layout()
    return _save(fig, out_figures / "figure2_rank_enrichment_recovery.png")


def _centered_summary(report_dir: pathlib.Path, *, anchor_tool: str, fdr: float, effect_threshold: float, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for joined_path in _joined_paths(report_dir):
        frame = pd.read_csv(joined_path, sep="\t")
        if not {"logFC", "FDR"}.issubset(frame.columns):
            continue
        frame["logFC_num"] = pd.to_numeric(frame["logFC"], errors="coerce")
        frame["FDR_num"] = pd.to_numeric(frame["FDR"], errors="coerce")
        usable = frame["logFC_num"].notna() & frame["FDR_num"].notna() & (frame["FDR_num"] >= 0.0) & (frame["FDR_num"] <= 1.0)
        work = frame.loc[usable].copy()
        if work.empty:
            continue
        work["expected_effect"] = _expected_effect_from_logfc(work)
        work["is_positive"] = (work["FDR_num"] < fdr) & (work["expected_effect"] > effect_threshold)
        anchor_values = _rank_values(work, anchor_tool, "local")
        if anchor_values is None:
            continue
        work = work.loc[anchor_values.notna()].copy()
        if work.empty:
            continue
        work["anchor_rank"] = anchor_values.loc[work.index].astype(float)
        for tool_id in TOOL_IDS:
            values = _rank_values(work, tool_id, "local")
            scored = values.notna() if values is not None else pd.Series(False, index=work.index)
            tmp = pd.DataFrame({"tool_id": tool_id, "anchor_rank": work["anchor_rank"], "is_positive": work["is_positive"].astype(bool), "scored": scored.astype(bool)})
            rows.append(tmp)
    all_rows = pd.concat(rows, ignore_index=True)
    summary = []
    for tool_id in TOOL_IDS:
        sub = all_rows[all_rows["tool_id"] == tool_id]
        for idx, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
            mask = (sub["anchor_rank"] >= left) & ((sub["anchor_rank"] <= right) if idx == n_bins - 1 else (sub["anchor_rank"] < right))
            in_bin = sub.loc[mask]
            scored = in_bin.loc[in_bin["scored"]]
            total = int(len(in_bin))
            scored_count = int(len(scored))
            positives = int(scored["is_positive"].sum())
            summary.append({
                "anchor_tool": anchor_tool,
                "anchor_label": TOOL_LABELS[anchor_tool],
                "tool_id": tool_id,
                "predictor": TOOL_LABELS[tool_id],
                "rank_bin_start": left,
                "rank_bin_end": right,
                "rank_bin_mid": (left + right) / 2.0,
                "anchor_bin_total": total,
                "predictor_scored_count": scored_count,
                "positive_count": positives,
                "background_count": scored_count - positives,
                "coverage_fraction": scored_count / total if total else np.nan,
                "positive_fraction_within_scored": positives / scored_count if scored_count else np.nan,
            })
    return pd.DataFrame(summary)


def plot_centered(summary: pd.DataFrame, out_path: pathlib.Path) -> pathlib.Path:
    anchor_label = str(summary["anchor_label"].iloc[0])
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 9.6), sharex=True)
    for tool_id in TOOL_IDS:
        sub = summary[summary["tool_id"] == tool_id].sort_values("rank_bin_mid")
        axes[0].plot(sub["rank_bin_mid"], sub["positive_fraction_within_scored"] * 100, marker="o", linewidth=2, label=TOOL_LABELS[tool_id])
        axes[1].plot(sub["rank_bin_mid"], sub["coverage_fraction"] * 100, marker="o", linewidth=2, label=TOOL_LABELS[tool_id])
    axes[0].set_title(f"{anchor_label}-centered GT-positive enrichment")
    axes[0].set_ylabel("GT positives among scored genes (%)")
    axes[0].set_ylim(bottom=0)
    _style_axes(axes[0], grid_axis="both")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    axes[1].set_title(f"Predictor coverage across {anchor_label} bins")
    axes[1].set_xlabel(f"{anchor_label} local rank bin (0 = weakest, 1 = strongest)")
    axes[1].set_ylabel("Genes in bin scored by predictor (%)")
    axes[1].set_ylim(0, 105)
    _style_axes(axes[1], grid_axis="both")
    fig.tight_layout()
    return _save(fig, out_path)


def build_assets(report_dir: pathlib.Path, out_dir: pathlib.Path, *, fdr: float = 0.05, effect_threshold: float = 1.0) -> dict[str, pathlib.Path]:
    out_figures = out_dir / "figures"
    out_tables = out_dir / "tables"
    out_figures.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)
    written: dict[str, pathlib.Path] = {}

    metric_rows = _load_metric_rows(report_dir)
    metric_rows.to_csv(out_tables / "per_experiment_metric_rows.tsv", sep="\t", index=False)
    written.update(_write_predictor_summary_tables(metric_rows, out_tables))
    written["figure1"] = plot_cross_dataset_distributions(metric_rows, out_figures / "figure1_cross_dataset_distributions.png")
    written["figure2"] = plot_figure2(report_dir, out_figures, out_tables, fdr=fdr, effect_threshold=effect_threshold)

    targetscan = _centered_summary(report_dir, anchor_tool="targetscan", fdr=fdr, effect_threshold=effect_threshold)
    targetscan_table = out_tables / "table_s2_targetscan_centered_bins.tsv"
    targetscan.to_csv(targetscan_table, sep="\t", index=False)
    written["table_s2"] = targetscan_table
    written["figure3"] = plot_centered(targetscan, out_figures / "figure3_targetscan_centered.png")

    mirdb = _centered_summary(report_dir, anchor_tool="mirdb_mirtarget", fdr=fdr, effect_threshold=effect_threshold)
    mirdb_table = out_tables / "table_s3_mirdb_centered_bins.tsv"
    mirdb.to_csv(mirdb_table, sep="\t", index=False)
    written["table_s3"] = mirdb_table
    written["supplementary_mirdb"] = plot_centered(mirdb, out_figures / "supplementary_mirdb_centered.png")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manuscript figures and tables from a FuNmiRBench report directory.")
    parser.add_argument("--report-dir", type=pathlib.Path, required=True, help="FuNmiRBench report directory containing tables and joined.tsv files.")
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("manuscript_assets"), help="Output directory for manuscript assets.")
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--effect-threshold", type=float, default=1.0)
    args = parser.parse_args()
    written = build_assets(args.report_dir, args.out_dir, fdr=args.fdr_threshold, effect_threshold=args.effect_threshold)
    for key, path in written.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
