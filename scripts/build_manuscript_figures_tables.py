"""Build manuscript figures and tables from a FuNmiRBench report directory.

This post-processing script creates only the manuscript assets we currently plan
to include: three main figures, one main summary table, and one supplementary
per-dataset metrics table. It does not rerun the benchmark.

Example
-------
python scripts/build_manuscript_figures_tables.py \
    --report-dir results/20260703_115539 \
    --out-dir manuscript_assets
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, FancyBboxPatch
from matplotlib.ticker import PercentFormatter

TOOL_IDS = ["targetscan", "mirdb_mirtarget", "microt_cnn", "mirbind2", "miraw"]
TOOL_LABELS = {
    "targetscan": "TargetScan",
    "mirdb_mirtarget": "miRDB",
    "microt_cnn": "microT-CNN",
    "mirbind2": "miRBind2",
    "miraw": "miRAW",
}
MAIN_METRICS = ["coverage", "positive_coverage", "aps", "auroc"]
SUPPLEMENTARY_METRICS = ["coverage", "positive_coverage", "aps", "pr_auc", "auroc", "spearman"]
METRIC_LABELS = {
    "coverage": "Coverage",
    "positive_coverage": "Positive coverage",
    "aps": "APS",
    "pr_auc": "PR-AUC",
    "auroc": "AUROC",
    "spearman": "Spearman",
}
TOOL_PALETTE = [
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]


def tool_color(tool_id: str, tool_ids=TOOL_IDS) -> str:
    try:
        index = list(tool_ids).index(tool_id)
    except ValueError:
        index = sum(ord(char) for char in str(tool_id))
    return TOOL_PALETTE[index % len(TOOL_PALETTE)]


def style_axes(ax, *, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, alpha=0.25)
    ax.set_axisbelow(True)


def save_figure(fig, out_path: pathlib.Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def find_table(report_dir: pathlib.Path, filename: str) -> pathlib.Path:
    candidates = [
        report_dir / "tables" / "combined" / filename,
        report_dir / "tables" / filename,
        report_dir / filename,
    ]
    candidates.extend(sorted(report_dir.glob(f"**/{filename}")))
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {filename} below {report_dir}")


def joined_paths(report_dir: pathlib.Path) -> list[pathlib.Path]:
    paths = sorted(report_dir.glob("**/joined.tsv"))
    if not paths:
        raise FileNotFoundError(f"No joined.tsv files found below {report_dir}")
    return paths


def metric_table_long(report_dir: pathlib.Path, metric: str) -> pd.DataFrame:
    table = pd.read_csv(find_table(report_dir, f"{metric}_per_experiment.tsv"), sep="\t")
    id_cols = [col for col in table.columns if col not in TOOL_IDS]
    available_tools = [tool_id for tool_id in TOOL_IDS if tool_id in table.columns]
    long = table.melt(id_vars=id_cols, value_vars=available_tools, var_name="tool_id", value_name=metric)
    long[metric] = pd.to_numeric(long[metric], errors="coerce")
    return long


def load_per_dataset_metrics(report_dir: pathlib.Path) -> pd.DataFrame:
    merged = None
    key_cols = None
    for metric in SUPPLEMENTARY_METRICS:
        try:
            long = metric_table_long(report_dir, metric)
        except FileNotFoundError:
            continue
        id_cols = [col for col in long.columns if col not in {"tool_id", metric}]
        keys = [*id_cols, "tool_id"]
        if merged is None:
            merged = long
            key_cols = keys
        else:
            merged = merged.merge(long[keys + [metric]], on=key_cols, how="outer")
    if merged is None:
        raise FileNotFoundError("No per-experiment metric tables were found.")
    merged.insert(merged.columns.get_loc("tool_id") + 1, "predictor", merged["tool_id"].map(TOOL_LABELS).fillna(merged["tool_id"]))
    return merged


def metric_rows_for_plots(per_dataset: pd.DataFrame) -> pd.DataFrame:
    id_cols = [col for col in per_dataset.columns if col not in set(SUPPLEMENTARY_METRICS)]
    long = per_dataset.melt(id_vars=id_cols, value_vars=[m for m in MAIN_METRICS if m in per_dataset.columns], var_name="metric", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.dropna(subset=["value"])


def write_tables(per_dataset: pd.DataFrame, tables_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    supp = tables_dir / "table_s1_per_dataset_predictor_metrics.tsv"
    per_dataset.to_csv(supp, sep="\t", index=False)

    metric_cols = [m for m in MAIN_METRICS if m in per_dataset.columns]
    summary = per_dataset.groupby(["tool_id", "predictor"])[metric_cols].agg(["mean", "median"]).reset_index()
    summary.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns]
    table1 = tables_dir / "table1_cross_dataset_predictor_summary.tsv"
    summary.to_csv(table1, sep="\t", index=False)
    return {"table1": table1, "table_s1": supp}


def plot_figure1(metric_rows: pd.DataFrame, figures_dir: pathlib.Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for ax, metric in zip(axes.ravel(), MAIN_METRICS):
        sub = metric_rows[metric_rows["metric"] == metric]
        data = [sub.loc[sub["tool_id"] == tool_id, "value"].dropna().to_numpy(float) for tool_id in TOOL_IDS]
        labels = [TOOL_LABELS[tool_id] for tool_id in TOOL_IDS]
        box = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, tool_id in zip(box["boxes"], TOOL_IDS):
            color = tool_color(tool_id)
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.28)
        for median in box["medians"]:
            median.set_color("#22303C")
            median.set_linewidth(1.4)
        for idx, (tool_id, values) in enumerate(zip(TOOL_IDS, data), start=1):
            if values.size:
                jitter = np.linspace(-0.08, 0.08, values.size) if values.size > 1 else np.array([0.0])
                ax.scatter(np.full(values.size, idx) + jitter, values, s=18, alpha=0.72, color=tool_color(tool_id), edgecolors="white", linewidths=0.3)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0, 1.02)
        if metric in {"coverage", "positive_coverage"}:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        style_axes(ax)
    fig.suptitle("Cross-dataset predictor distributions", fontsize=15, y=1.0)
    fig.tight_layout()
    return save_figure(fig, figures_dir / "figure1_cross_dataset_distributions.png")


def score_columns(frame: pd.DataFrame) -> list[str]:
    return [f"score_{tool_id}" for tool_id in TOOL_IDS if f"score_{tool_id}" in frame.columns]


def gene_universe_count_table(report_dir: pathlib.Path) -> pd.DataFrame:
    """Summarize full-gene-set and intersection-gene-set sizes per dataset.

    FGS is the full usable joined table for a dataset. IGS is the subset of those
    genes with non-missing scores from every selected predictor. The individual
    predictor columns count genes scored by each method inside the FGS universe.
    """
    rows = []
    for path in joined_paths(report_dir):
        frame = pd.read_csv(path, sep="\t")
        if "dataset_id" in frame.columns and not frame.empty:
            dataset_id = str(frame["dataset_id"].iloc[0])
        else:
            dataset_id = path.parent.name
        frame = frame.copy()
        frame["logFC_num"] = pd.to_numeric(frame.get("logFC"), errors="coerce")
        usable = frame.loc[frame["logFC_num"].notna()].copy()
        cols = score_columns(usable)
        row = {
            "dataset_id": dataset_id,
            "fgs_gene_count": int(len(usable)),
            "available_predictor_count": int(len(cols)),
        }
        if cols:
            scored = usable[cols].notna()
            row["igs_gene_count"] = int(scored.all(axis=1).sum())
            row["union_scored_gene_count"] = int(scored.any(axis=1).sum())
            for tool_id in TOOL_IDS:
                col = f"score_{tool_id}"
                if col in scored.columns:
                    row[f"{tool_id}_scored_gene_count"] = int(scored[col].sum())
        else:
            row["igs_gene_count"] = 0
            row["union_scored_gene_count"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


def write_gene_universe_table(report_dir: pathlib.Path, tables_dir: pathlib.Path) -> pathlib.Path:
    table = gene_universe_count_table(report_dir)
    out_path = tables_dir / "figure1_panel_c_gene_universe_counts.tsv"
    table.to_csv(out_path, sep="\t", index=False)
    return out_path


def summarize_gene_universes(counts: pd.DataFrame) -> dict[str, int]:
    if counts.empty:
        return {"fgs": 0, "igs": 0, "union": 0, "datasets": 0}
    return {
        "fgs": int(round(pd.to_numeric(counts["fgs_gene_count"], errors="coerce").median())),
        "igs": int(round(pd.to_numeric(counts["igs_gene_count"], errors="coerce").median())),
        "union": int(round(pd.to_numeric(counts["union_scored_gene_count"], errors="coerce").median())),
        "datasets": int(counts["dataset_id"].nunique()) if "dataset_id" in counts.columns else int(len(counts)),
    }


def plot_figure1_panel_c(report_dir: pathlib.Path, figures_dir: pathlib.Path):
    """Draw the manuscript panel C schematic for FGS versus IGS.

    The five translucent ellipses represent the gene sets scored by each
    predictor. Their central overlap is the IGS. The full plotting field is the
    FGS, i.e. all usable genes retained in the joined DE table, including genes
    not scored by one or more methods.
    """
    summary = summarize_gene_universes(gene_universe_count_table(report_dir))
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.2)
    ax.set_aspect("equal")
    ax.axis("off")

    fgs_box = FancyBboxPatch(
        (0.35, 0.45),
        9.3,
        6.25,
        boxstyle="round,pad=0.18,rounding_size=0.28",
        facecolor="#F7F8FB",
        edgecolor="#B8C4D6",
        linewidth=1.1,
        zorder=0,
    )
    ax.add_patch(fgs_box)
    ax.text(0.65, 6.45, "FGS: full usable gene set", fontsize=11, weight="bold", ha="left", va="center")
    ax.text(
        0.65,
        6.08,
        f"median {summary['fgs']:,} genes across {summary['datasets']:,} datasets; unscored pairs kept as lowest confidence",
        fontsize=8.7,
        color="#44546A",
        ha="left",
        va="center",
    )

    ellipse_specs = [
        ("targetscan", 3.85, 4.05, 6.3, 2.45, 0),
        ("mirdb_mirtarget", 6.15, 4.05, 6.3, 2.45, 0),
        ("microt_cnn", 5.00, 4.95, 2.95, 5.45, 0),
        ("mirbind2", 3.95, 2.85, 2.95, 5.45, -52),
        ("miraw", 6.05, 2.85, 2.95, 5.45, 52),
    ]
    label_positions = {
        "targetscan": (1.25, 4.12),
        "mirdb_mirtarget": (8.75, 4.12),
        "microt_cnn": (5.00, 6.33),
        "mirbind2": (2.15, 1.12),
        "miraw": (7.85, 1.12),
    }
    for tool_id, x, y, width, height, angle in ellipse_specs:
        patch = Ellipse(
            (x, y),
            width,
            height,
            angle=angle,
            facecolor=tool_color(tool_id),
            edgecolor=tool_color(tool_id),
            alpha=0.34,
            linewidth=1.1,
            zorder=1,
        )
        ax.add_patch(patch)
        lx, ly = label_positions[tool_id]
        ax.text(lx, ly, TOOL_LABELS[tool_id], fontsize=9.2, weight="bold", ha="center", va="center", color="#25364A")

    igs_box = FancyBboxPatch(
        (3.77, 3.05),
        2.46,
        1.18,
        boxstyle="round,pad=0.15,rounding_size=0.16",
        facecolor="white",
        edgecolor="#5B6577",
        linewidth=1.0,
        alpha=0.84,
        zorder=4,
    )
    ax.add_patch(igs_box)
    ax.text(5.0, 3.77, "IGS", fontsize=18, color="#D62728", style="italic", weight="bold", ha="center", va="center", zorder=5)
    ax.text(5.0, 3.34, f"all predictors scored\nmedian {summary['igs']:,} genes", fontsize=8.7, color="#25364A", ha="center", va="center", zorder=5)
    ax.text(5.0, 0.62, f"Union scored by at least one predictor: median {summary['union']:,} genes", fontsize=8.5, color="#44546A", ha="center", va="center")

    return save_figure(fig, figures_dir / "figure1_panel_c_gene_universes.png")


def expected_effect(frame: pd.DataFrame) -> pd.Series:
    logfc = pd.to_numeric(frame["logFC"], errors="coerce")
    perturbation = frame.get("perturbation", pd.Series("", index=frame.index)).astype(str).str.upper().fillna("")
    effect = logfc.abs().copy()
    oe = perturbation.isin(["OE", "OVEREXPRESSION"])
    kdko = perturbation.isin(["KO", "KD", "KNOCKOUT", "KNOCKDOWN"])
    effect.loc[oe] = -logfc.loc[oe]
    effect.loc[kdko] = logfc.loc[kdko]
    return effect


def dense_rank_scale(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    ranks = values.rank(method="dense", ascending=True)
    max_rank = ranks.max(skipna=True)
    if pd.isna(max_rank):
        return pd.Series(np.nan, index=values.index)
    if float(max_rank) <= 1.0:
        return pd.Series(1.0, index=values.index, dtype=float)
    return (ranks - 1.0) / (float(max_rank) - 1.0)


def rank_values(frame: pd.DataFrame, tool_id: str, rank_type: str) -> pd.Series | None:
    col = f"{rank_type}_rank_{tool_id}"
    score = f"score_{tool_id}"
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce")
    if score in frame.columns:
        return dense_rank_scale(frame[score])
    return None


def usable_joined(frame: pd.DataFrame, *, fdr: float, effect_threshold: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["logFC_num"] = pd.to_numeric(frame["logFC"], errors="coerce")
    frame["FDR_num"] = pd.to_numeric(frame["FDR"], errors="coerce")
    keep = frame["logFC_num"].notna() & frame["FDR_num"].notna() & (frame["FDR_num"] >= 0.0) & (frame["FDR_num"] <= 1.0)
    frame = frame.loc[keep].copy()
    frame["expected_effect"] = expected_effect(frame)
    frame["is_positive"] = (frame["FDR_num"] < fdr) & (frame["expected_effect"] > effect_threshold)
    return frame


def rank_fraction_table(report_dir: pathlib.Path, *, rank_type: str, bins: int, fdr: float, effect_threshold: float) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    stacked = []
    for path in joined_paths(report_dir):
        frame = usable_joined(pd.read_csv(path, sep="\t"), fdr=fdr, effect_threshold=effect_threshold)
        for tool_id in TOOL_IDS:
            ranks = rank_values(frame, tool_id, rank_type)
            if ranks is not None:
                stacked.append(pd.DataFrame({"tool_id": tool_id, "rank": ranks, "is_positive": frame["is_positive"]}).dropna(subset=["rank"]))
    data = pd.concat(stacked, ignore_index=True)
    rows = []
    for tool_id in TOOL_IDS:
        sub = data[data["tool_id"] == tool_id]
        for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (sub["rank"] >= left) & ((sub["rank"] <= right) if i == bins - 1 else (sub["rank"] < right))
            in_bin = sub.loc[mask]
            total = int(len(in_bin))
            positives = int(in_bin["is_positive"].sum())
            rows.append({"tool_id": tool_id, "predictor": TOOL_LABELS[tool_id], "rank_bin_mid": (left + right) / 2, "total_count": total, "positive_count": positives, "positive_fraction_within_bin": positives / total if total else np.nan})
    return pd.DataFrame(rows)


def recovery_table(report_dir: pathlib.Path, *, fdr: float, effect_threshold: float, max_predictions: int = 300) -> pd.DataFrame:
    curves = {tool_id: [] for tool_id in TOOL_IDS}
    for path in joined_paths(report_dir):
        frame = usable_joined(pd.read_csv(path, sep="\t"), fdr=fdr, effect_threshold=effect_threshold)
        positive_total = int(frame["is_positive"].sum())
        if positive_total <= 0:
            continue
        for tool_id in TOOL_IDS:
            ranks = rank_values(frame, tool_id, "local")
            if ranks is None:
                ranks = rank_values(frame, tool_id, "global")
            if ranks is None:
                continue
            scored = frame.loc[ranks.notna(), ["gene_id", "is_positive"]].copy()
            if scored.empty:
                continue
            scored["rank"] = ranks.loc[scored.index].astype(float)
            scored = scored.sort_values(["rank", "gene_id"], ascending=[False, True], kind="mergesort")
            cumulative = np.cumsum(scored["is_positive"].to_numpy(int)).astype(float) / positive_total
            curve = np.repeat(cumulative[-1], max_predictions)
            observed = min(max_predictions, cumulative.size)
            curve[:observed] = cumulative[:observed]
            curves[tool_id].append(curve)
    rows = []
    for tool_id, tool_curves in curves.items():
        if not tool_curves:
            continue
        mean_curve = np.vstack(tool_curves).mean(axis=0)
        for i, value in enumerate(mean_curve, start=1):
            rows.append({"tool_id": tool_id, "predictor": TOOL_LABELS[tool_id], "prediction_count": i, "recovery_fraction": value})
    return pd.DataFrame(rows)


def plot_fraction(ax, table: pd.DataFrame, *, title: str, xlabel: str):
    for tool_id in TOOL_IDS:
        sub = table[table["tool_id"] == tool_id].sort_values("rank_bin_mid")
        ax.plot(sub["rank_bin_mid"], sub["positive_fraction_within_bin"] * 100, marker="o", linewidth=2, color=tool_color(tool_id), label=TOOL_LABELS[tool_id])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("GT positives within bin (%)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    style_axes(ax, grid_axis="both")


def plot_figure2(local: pd.DataFrame, global_: pd.DataFrame, recovery: pd.DataFrame, figures_dir: pathlib.Path):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))
    plot_fraction(axes[0], local, title="Local rank enrichment", xlabel="Local normalized rank bin")
    plot_fraction(axes[1], global_, title="Global rank enrichment", xlabel="Global normalized rank bin")
    for tool_id in TOOL_IDS:
        sub = recovery[recovery["tool_id"] == tool_id]
        axes[2].plot(sub["prediction_count"], sub["recovery_fraction"] * 100, linewidth=2, color=tool_color(tool_id), label=TOOL_LABELS[tool_id])
    axes[2].set_title("GT-positive recovery")
    axes[2].set_xlabel("Predicted targets per dataset")
    axes[2].set_ylabel("Mean GT-positive recovery (%)")
    axes[2].set_ylim(bottom=0)
    style_axes(axes[2], grid_axis="both")
    axes[2].legend(frameon=False, fontsize=9)
    fig.suptitle("Functional target enrichment and recovery across predictor ranks", fontsize=15, y=1.02)
    fig.tight_layout()
    return save_figure(fig, figures_dir / "figure2_rank_enrichment_recovery.png")


def centered_table(report_dir: pathlib.Path, *, anchor_tool: str, fdr: float, effect_threshold: float, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    stacked = []
    for path in joined_paths(report_dir):
        frame = usable_joined(pd.read_csv(path, sep="\t"), fdr=fdr, effect_threshold=effect_threshold)
        anchor = rank_values(frame, anchor_tool, "local")
        if anchor is None:
            continue
        frame = frame.loc[anchor.notna()].copy()
        anchor = anchor.loc[frame.index].astype(float)
        for tool_id in TOOL_IDS:
            ranks = rank_values(frame, tool_id, "local")
            scored = ranks.notna() if ranks is not None else pd.Series(False, index=frame.index)
            stacked.append(pd.DataFrame({"tool_id": tool_id, "anchor_rank": anchor, "is_positive": frame["is_positive"], "scored": scored.astype(bool)}))
    data = pd.concat(stacked, ignore_index=True)
    rows = []
    for tool_id in TOOL_IDS:
        sub = data[data["tool_id"] == tool_id]
        for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (sub["anchor_rank"] >= left) & ((sub["anchor_rank"] <= right) if i == bins - 1 else (sub["anchor_rank"] < right))
            in_bin = sub.loc[mask]
            scored = in_bin[in_bin["scored"]]
            total = int(len(in_bin))
            scored_count = int(len(scored))
            positives = int(scored["is_positive"].sum())
            rows.append({"tool_id": tool_id, "predictor": TOOL_LABELS[tool_id], "rank_bin_mid": (left + right) / 2, "anchor_bin_total": total, "predictor_scored_count": scored_count, "coverage_fraction": scored_count / total if total else np.nan, "positive_fraction_within_scored": positives / scored_count if scored_count else np.nan})
    return pd.DataFrame(rows)


def plot_figure3(table: pd.DataFrame, figures_dir: pathlib.Path):
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 9.6), sharex=True)
    for tool_id in TOOL_IDS:
        sub = table[table["tool_id"] == tool_id].sort_values("rank_bin_mid")
        axes[0].plot(sub["rank_bin_mid"], sub["positive_fraction_within_scored"] * 100, marker="o", linewidth=2, color=tool_color(tool_id), label=TOOL_LABELS[tool_id])
        axes[1].plot(sub["rank_bin_mid"], sub["coverage_fraction"] * 100, marker="o", linewidth=2, color=tool_color(tool_id), label=TOOL_LABELS[tool_id])
    axes[0].set_title("TargetScan-centered GT-positive enrichment")
    axes[0].set_ylabel("GT positives among scored genes (%)")
    axes[0].set_ylim(bottom=0)
    style_axes(axes[0], grid_axis="both")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    axes[1].set_title("Predictor coverage across TargetScan bins")
    axes[1].set_xlabel("TargetScan local rank bin (0 = weakest, 1 = strongest)")
    axes[1].set_ylabel("Genes in bin scored by predictor (%)")
    axes[1].set_ylim(0, 105)
    style_axes(axes[1], grid_axis="both")
    fig.tight_layout()
    return save_figure(fig, figures_dir / "figure3_targetscan_centered.png")


def build_assets(report_dir: pathlib.Path, out_dir: pathlib.Path, *, fdr: float, effect_threshold: float):
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    per_dataset = load_per_dataset_metrics(report_dir)
    metrics_long = metric_rows_for_plots(per_dataset)
    outputs.update(write_tables(per_dataset, tables_dir))
    outputs["figure1"] = plot_figure1(metrics_long, figures_dir)
    outputs["figure1_panel_c_table"] = write_gene_universe_table(report_dir, tables_dir)
    outputs["figure1_panel_c"] = plot_figure1_panel_c(report_dir, figures_dir)

    local = rank_fraction_table(report_dir, rank_type="local", bins=10, fdr=fdr, effect_threshold=effect_threshold)
    global_ = rank_fraction_table(report_dir, rank_type="global", bins=5, fdr=fdr, effect_threshold=effect_threshold)
    recovery = recovery_table(report_dir, fdr=fdr, effect_threshold=effect_threshold)
    outputs["figure2"] = plot_figure2(local, global_, recovery, figures_dir)

    targetscan = centered_table(report_dir, anchor_tool="targetscan", fdr=fdr, effect_threshold=effect_threshold)
    outputs["figure3"] = plot_figure3(targetscan, figures_dir)
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Create manuscript figures and tables from a FuNmiRBench report.")
    parser.add_argument("--report-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("manuscript_assets"))
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--effect-threshold", type=float, default=1.0)
    args = parser.parse_args()
    outputs = build_assets(args.report_dir, args.out_dir, fdr=args.fdr_threshold, effect_threshold=args.effect_threshold)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
