"""Build manuscript figures and tables from a FuNmiRBench report directory.

This post-processing script creates only the manuscript assets we currently plan
to include: main figures, one main summary table, and one supplementary
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
from matplotlib.ticker import MaxNLocator, PercentFormatter

from funmirbench.gene_conservation import load_utr3_conservation
from funmirbench.gene_lengths import load_utr3_lengths

TOOL_IDS = ["targetscan", "mirdb_mirtarget", "microt_cnn", "mirbind2", "miraw"]
TOOL_LABELS = {
    "targetscan": "TargetScan",
    "mirdb_mirtarget": "miRDB",
    "microt_cnn": "microT-CNN",
    "mirbind2": "miRBind2-3UTR",
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


def strip_ensembl_version(value) -> str:
    return str(value).strip().split(".", 1)[0]


def find_repo_root(report_dir: pathlib.Path) -> pathlib.Path:
    """Find the repository root used for cached Ensembl/UCSC resources."""
    checked: set[pathlib.Path] = set()
    starts = [pathlib.Path.cwd(), report_dir.resolve()]
    for start in starts:
        for candidate in (start, *start.parents):
            if candidate in checked:
                continue
            checked.add(candidate)
            if (candidate / "pyproject.toml").exists() and (candidate / "benchmark.yaml").exists():
                return candidate
    return pathlib.Path.cwd().resolve()


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
    """Summarize full-gene-set and intersection-gene-set sizes per dataset."""
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
    """Draw the manuscript panel C schematic for FGS versus IGS."""
    counts = gene_universe_count_table(report_dir)
    summary = summarize_gene_universes(counts)
    predictor_medians = {
        tool_id: int(round(pd.to_numeric(counts[f"{tool_id}_scored_gene_count"], errors="coerce").median()))
        for tool_id in TOOL_IDS
        if f"{tool_id}_scored_gene_count" in counts.columns
    }

    fig, ax = plt.subplots(figsize=(5.8, 4.35))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.4)
    ax.set_aspect("equal")
    ax.axis("off")

    ellipse_specs = [
        ("targetscan", "A", 4.95, 4.85, 3.65, 5.75, 0),
        ("mirdb_mirtarget", "B", 6.85, 4.10, 4.85, 2.75, 0),
        ("microt_cnn", "C", 6.25, 2.72, 5.00, 2.85, -23),
        ("mirbind2", "D", 3.78, 2.70, 4.90, 2.85, 23),
        ("miraw", "E", 3.05, 4.10, 4.85, 2.75, 0),
    ]
    letter_positions = {
        "targetscan": (4.95, 7.03),
        "mirdb_mirtarget": (8.97, 4.72),
        "microt_cnn": (6.88, 1.42),
        "mirbind2": (3.12, 1.42),
        "miraw": (1.03, 4.72),
    }
    for tool_id, letter, x, y, width, height, angle in ellipse_specs:
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
        lx, ly = letter_positions[tool_id]
        ax.text(lx, ly, letter, fontsize=9.5, weight="bold", ha="center", va="center", color="#25364A")

    igs_box = FancyBboxPatch(
        (3.55, 3.32),
        2.9,
        1.1,
        boxstyle="round,pad=0.12,rounding_size=0.08",
        facecolor=(1, 1, 1, 0.72),
        edgecolor="#4A5568",
        linewidth=0.9,
        zorder=4,
    )
    ax.add_patch(igs_box)
    ax.text(5.0, 4.08, "IGS", fontsize=17, color="#D62728", style="italic", weight="bold", ha="center", va="center", zorder=5)
    ax.text(5.0, 3.63, f"all five predictors\nmedian {summary['igs']:,} genes", fontsize=8.4, color="#25364A", ha="center", va="center", zorder=5)

    legend_rows = [
        ("A", "TargetScan", "targetscan"),
        ("B", "miRDB", "mirdb_mirtarget"),
        ("C", "microT-CNN", "microt_cnn"),
        ("D", "miRBind2", "mirbind2"),
        ("E", "miRAW", "miraw"),
    ]
    x0, y0 = 0.42, 0.77
    for idx, (letter, label, tool_id) in enumerate(legend_rows):
        y = y0 - idx * 0.25
        ax.text(x0, y, letter, fontsize=7.4, weight="bold", color=tool_color(tool_id), ha="left", va="center")
        ax.text(x0 + 0.25, y, f"{label}: median {predictor_medians.get(tool_id, 0):,}", fontsize=7.3, color="#25364A", ha="left", va="center")

    ax.text(5.0, 0.56, f"FGS median {summary['fgs']:,} genes across {summary['datasets']:,} datasets", fontsize=8.1, color="#25364A", ha="center", va="center")
    ax.text(5.0, 0.27, f"Union scored by >=1 predictor: median {summary['union']:,} genes", fontsize=7.8, color="#44546A", ha="center", va="center")

    return save_figure(fig, figures_dir / "figure1_panel_c_gene_universes.png")


def membership_table(report_dir: pathlib.Path, *, membership_mode: str = "dataset_gene") -> pd.DataFrame:
    """Return IGS/non-IGS membership by dataset-gene or unique gene."""
    if membership_mode not in {"dataset_gene", "unique_gene"}:
        raise ValueError("membership_mode must be 'dataset_gene' or 'unique_gene'")
    rows = []
    for path in joined_paths(report_dir):
        frame = pd.read_csv(path, sep="\t")
        if "gene_id" not in frame.columns:
            continue
        if "dataset_id" in frame.columns and not frame.empty:
            dataset_id = str(frame["dataset_id"].iloc[0])
        else:
            dataset_id = path.parent.name
        frame = frame.copy()
        frame["gene_id"] = frame["gene_id"].map(strip_ensembl_version)
        frame["logFC_num"] = pd.to_numeric(frame.get("logFC"), errors="coerce")
        usable = frame.loc[frame["logFC_num"].notna()].dropna(subset=["gene_id"]).copy()
        cols = score_columns(usable)
        if not cols:
            continue
        scored = usable[cols].notna()
        rows.append(pd.DataFrame({
            "dataset_id": dataset_id,
            "gene_id": usable["gene_id"].to_numpy(),
            "is_igs": scored.all(axis=1).to_numpy(bool),
        }))
    if not rows:
        raise ValueError(f"No joined rows with predictor scores found below {report_dir}")
    membership = pd.concat(rows, ignore_index=True).drop_duplicates()
    if membership_mode == "unique_gene":
        membership = membership.groupby("gene_id", as_index=False)["is_igs"].any()
        membership.insert(0, "dataset_id", "unique_gene")
    membership["gene_set"] = np.where(membership["is_igs"], "IGS genes", "non-IGS genes")
    return membership[["dataset_id", "gene_id", "gene_set"]]


def panel_d_gene_length_table(report_dir: pathlib.Path, gene_lengths: pd.DataFrame, *, membership_mode: str = "dataset_gene") -> pd.DataFrame:
    """Build the Panel D source table from joined files and computed 3'UTR lengths."""
    merged = membership_table(report_dir, membership_mode=membership_mode).merge(gene_lengths, on="gene_id", how="left")
    return merged[["dataset_id", "gene_id", "gene_set", "utr3_length_bp"]].copy()


def panel_d_qc_table(panel_d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene_set, sub in panel_d.groupby("gene_set", sort=False):
        lengths = pd.to_numeric(sub["utr3_length_bp"], errors="coerce")
        matched = lengths.notna()
        rows.append({
            "gene_set": gene_set,
            "n_rows": int(len(sub)),
            "n_rows_with_length": int(matched.sum()),
            "length_match_fraction": float(matched.mean()) if len(sub) else np.nan,
            "n_unique_genes": int(sub["gene_id"].nunique()),
            "median_utr3_length_bp": float(lengths.dropna().median()) if matched.any() else np.nan,
            "mean_utr3_length_bp": float(lengths.dropna().mean()) if matched.any() else np.nan,
        })
    return pd.DataFrame(rows)


def write_panel_d_tables(panel_d: pd.DataFrame, tables_dir: pathlib.Path, *, name_suffix: str = "") -> dict[str, pathlib.Path]:
    source_path = tables_dir / f"figure1_panel_d_gene_lengths{name_suffix}.tsv"
    qc_path = tables_dir / f"figure1_panel_d_gene_length_qc{name_suffix}.tsv"
    key_suffix = name_suffix.lstrip("_")
    key_prefix = "figure1_panel_d" if not key_suffix else f"figure1_panel_d_{key_suffix}"
    panel_d.to_csv(source_path, sep="\t", index=False)
    panel_d_qc_table(panel_d).to_csv(qc_path, sep="\t", index=False)
    return {f"{key_prefix}_table": source_path, f"{key_prefix}_qc": qc_path}


def kde_manual(values, grid, bandwidth: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros_like(grid, dtype=float)
    if bandwidth is None:
        std = np.std(values, ddof=1) if values.size > 1 else max(float(values[0]) * 0.1, 1.0)
        bandwidth = 1.06 * std * (values.size ** (-1.0 / 5.0)) if std > 0 else max(float(np.median(values)) * 0.1, 1.0)
    bandwidth = max(float(bandwidth), 1.0)
    density = np.zeros_like(grid, dtype=float)
    chunk_size = 2500
    for start in range(0, values.size, chunk_size):
        chunk = values[start:start + chunk_size]
        scaled = (grid[:, None] - chunk[None, :]) / bandwidth
        density += np.exp(-0.5 * scaled * scaled).sum(axis=1)
    density /= values.size * bandwidth * np.sqrt(2.0 * np.pi)
    return density


def plot_mirrored_density(
    table: pd.DataFrame,
    figures_dir: pathlib.Path,
    *,
    value_col: str,
    xlabel: str,
    out_name: str,
    label_mode: str = "dataset_gene",
):
    plot_data = table.dropna(subset=[value_col]).copy()
    if plot_data.empty:
        raise ValueError(f"No matched values available for {out_name}.")
    plot_data[value_col] = pd.to_numeric(plot_data[value_col], errors="coerce")
    plot_data = plot_data.dropna(subset=[value_col])
    if plot_data["gene_set"].nunique() < 2:
        raise ValueError(f"{out_name} requires both IGS genes and non-IGS genes.")

    lower = float(plot_data[value_col].quantile(0.01))
    upper = float(plot_data[value_col].quantile(0.99))
    if upper <= lower:
        lower = float(plot_data[value_col].min())
        upper = float(plot_data[value_col].max())
    grid = np.linspace(lower, upper, 500)
    igs = plot_data.loc[plot_data["gene_set"] == "IGS genes", value_col].to_numpy(float)
    non_igs = plot_data.loc[plot_data["gene_set"] == "non-IGS genes", value_col].to_numpy(float)
    igs_density = kde_manual(igs, grid)
    non_igs_density = kde_manual(non_igs, grid)
    max_density = max(float(igs_density.max()), float(non_igs_density.max()), 1e-12)
    igs_y = igs_density / max_density
    non_igs_y = non_igs_density / max_density
    igs_peak_x = float(grid[int(np.argmax(igs_density))])
    non_igs_peak_x = float(grid[int(np.argmax(non_igs_density))])

    if label_mode == "unique_gene":
        top_label = f"IGS Genes\nn={len(igs):,}"
        bottom_label = f"non-IGS Genes\nn={len(non_igs):,}"
    else:
        top_label = f"IGS dataset-gene pairs\nn={len(igs):,}"
        bottom_label = f"non-IGS dataset-gene pairs\nn={len(non_igs):,}"

    fig, ax = plt.subplots(figsize=(7.2, 2.55))
    igs_color = "#5DA5DA"
    non_igs_color = "#F17C7E"
    ax.fill_between(grid, 0, igs_y, alpha=0.82, color=igs_color, label="IGS Genes")
    ax.fill_between(grid, 0, -non_igs_y, alpha=0.82, color=non_igs_color, label="non-IGS Genes")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.vlines(igs_peak_x, 0, 0.28, color="#245C8A", linewidth=1.2)
    ax.vlines(non_igs_peak_x, 0, -0.28, color="#A83234", linewidth=1.2)
    ax.text(lower, 0.74, top_label, ha="left", va="center", fontsize=8)
    ax.text(lower, -0.74, bottom_label, ha="left", va="center", fontsize=8)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_yticks([])
    ax.set_xlim(lower, upper)
    ax.set_ylim(-1.08, 1.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["bottom"].set_position(("outward", 5))
    ax.xaxis.set_visible(True)
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True, length=4, width=0.8, pad=4, labelsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    return save_figure(fig, figures_dir / out_name)


def plot_figure1_panel_d_gene_lengths(
    panel_d: pd.DataFrame,
    figures_dir: pathlib.Path,
    *,
    out_name: str = "figure1_panel_d_gene_lengths.png",
    label_mode: str = "dataset_gene",
):
    return plot_mirrored_density(
        panel_d,
        figures_dir,
        value_col="utr3_length_bp",
        xlabel="3'UTR Length (bp)",
        out_name=out_name,
        label_mode=label_mode,
    )


def build_panel_d_assets(
    report_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    tables_dir: pathlib.Path,
    gene_lengths: pd.DataFrame,
    *,
    membership_mode: str,
    name_suffix: str,
) -> dict[str, pathlib.Path]:
    panel_d = panel_d_gene_length_table(report_dir, gene_lengths, membership_mode=membership_mode)
    outputs = write_panel_d_tables(panel_d, tables_dir, name_suffix=name_suffix)
    figure_key = "figure1_panel_d" if not name_suffix else f"figure1_panel_d{name_suffix}"
    outputs[figure_key] = plot_figure1_panel_d_gene_lengths(
        panel_d,
        figures_dir,
        out_name=f"figure1_panel_d_gene_lengths{name_suffix}.png",
        label_mode=membership_mode,
    )
    return outputs


def panel_e_conservation_table(report_dir: pathlib.Path, gene_conservation: pd.DataFrame, *, membership_mode: str = "dataset_gene") -> pd.DataFrame:
    """Build the Panel E source table from joined files and computed 3'UTR conservation."""
    cols = [col for col in ["gene_id", "mean_phyloP", "mean_phastCons", "utr3_scored_bases_phyloP", "utr3_scored_bases_phastCons"] if col in gene_conservation.columns]
    merged = membership_table(report_dir, membership_mode=membership_mode).merge(gene_conservation[cols], on="gene_id", how="left")
    return merged


def panel_e_qc_table(panel_e: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene_set, sub in panel_e.groupby("gene_set", sort=False):
        values = pd.to_numeric(sub["mean_phyloP"], errors="coerce")
        matched = values.notna()
        rows.append({
            "gene_set": gene_set,
            "n_rows": int(len(sub)),
            "n_rows_with_conservation": int(matched.sum()),
            "conservation_match_fraction": float(matched.mean()) if len(sub) else np.nan,
            "n_unique_genes": int(sub["gene_id"].nunique()),
            "median_mean_phyloP": float(values.dropna().median()) if matched.any() else np.nan,
            "mean_mean_phyloP": float(values.dropna().mean()) if matched.any() else np.nan,
        })
    return pd.DataFrame(rows)


def write_panel_e_tables(panel_e: pd.DataFrame, tables_dir: pathlib.Path, *, name_suffix: str = "") -> dict[str, pathlib.Path]:
    source_path = tables_dir / f"figure1_panel_e_conservation{name_suffix}.tsv"
    qc_path = tables_dir / f"figure1_panel_e_conservation_qc{name_suffix}.tsv"
    key_suffix = name_suffix.lstrip("_")
    key_prefix = "figure1_panel_e" if not key_suffix else f"figure1_panel_e_{key_suffix}"
    panel_e.to_csv(source_path, sep="\t", index=False)
    panel_e_qc_table(panel_e).to_csv(qc_path, sep="\t", index=False)
    return {f"{key_prefix}_table": source_path, f"{key_prefix}_qc": qc_path}


def plot_figure1_panel_e_conservation(
    panel_e: pd.DataFrame,
    figures_dir: pathlib.Path,
    *,
    out_name: str = "figure1_panel_e_conservation.png",
    label_mode: str = "dataset_gene",
):
    return plot_mirrored_density(
        panel_e,
        figures_dir,
        value_col="mean_phyloP",
        xlabel="Mean phyloP Conservation",
        out_name=out_name,
        label_mode=label_mode,
    )


def build_panel_e_assets(
    report_dir: pathlib.Path,
    figures_dir: pathlib.Path,
    tables_dir: pathlib.Path,
    gene_conservation: pd.DataFrame,
    *,
    membership_mode: str,
    name_suffix: str,
) -> dict[str, pathlib.Path]:
    panel_e = panel_e_conservation_table(report_dir, gene_conservation, membership_mode=membership_mode)
    outputs = write_panel_e_tables(panel_e, tables_dir, name_suffix=name_suffix)
    figure_key = "figure1_panel_e" if not name_suffix else f"figure1_panel_e{name_suffix}"
    outputs[figure_key] = plot_figure1_panel_e_conservation(
        panel_e,
        figures_dir,
        out_name=f"figure1_panel_e_conservation{name_suffix}.png",
        label_mode=membership_mode,
    )
    return outputs


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


def build_assets(
    report_dir: pathlib.Path,
    out_dir: pathlib.Path,
    *,
    fdr: float,
    effect_threshold: float,
):
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

    repo_root = find_repo_root(report_dir)
    gene_lengths = load_utr3_lengths(root=repo_root)
    outputs.update(
        build_panel_d_assets(
            report_dir,
            figures_dir,
            tables_dir,
            gene_lengths,
            membership_mode="dataset_gene",
            name_suffix="",
        )
    )
    outputs.update(
        build_panel_d_assets(
            report_dir,
            figures_dir,
            tables_dir,
            gene_lengths,
            membership_mode="unique_gene",
            name_suffix="_unique_gene",
        )
    )

    gene_conservation = load_utr3_conservation(root=repo_root)
    outputs.update(
        build_panel_e_assets(
            report_dir,
            figures_dir,
            tables_dir,
            gene_conservation,
            membership_mode="dataset_gene",
            name_suffix="",
        )
    )
    outputs.update(
        build_panel_e_assets(
            report_dir,
            figures_dir,
            tables_dir,
            gene_conservation,
            membership_mode="unique_gene",
            name_suffix="_unique_gene",
        )
    )

    local = rank_fraction_table(report_dir, rank_type="local", bins=10, fdr=fdr, effect_threshold=effect_threshold)
    global_ = rank_fraction_table(report_dir, rank_type="global", bins=5, fdr=fdr, effect_threshold=effect_threshold)
    recovery = recovery_table(report_dir, fdr=fdr, effect_threshold=effect_threshold)
    outputs["figure2"] = plot_figure2(local, global_, recovery, figures_dir)

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Create manuscript figures and tables from a FuNmiRBench report.")
    parser.add_argument("--report-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("manuscript_assets"))
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--effect-threshold", type=float, default=1.0)
    args = parser.parse_args()
    outputs = build_assets(
        args.report_dir,
        args.out_dir,
        fdr=args.fdr_threshold,
        effect_threshold=args.effect_threshold,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
