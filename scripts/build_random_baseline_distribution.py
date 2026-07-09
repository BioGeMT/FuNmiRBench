"""Build cross-dataset metric distributions including a random baseline.

The random baseline assigns deterministic U(0, 1) scores to every usable gene
inside each joined.tsv dataset and evaluates those scores with the same
perturbation-aware GT-positive rule used by the manuscript asset builder.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import average_precision_score, roc_auc_score

TOOL_IDS = ["targetscan", "mirdb_mirtarget", "microt_cnn", "mirbind2", "miraw"]
TOOL_LABELS = {
    "targetscan": "TargetScan",
    "mirdb_mirtarget": "miRDB",
    "microt_cnn": "microT-CNN",
    "mirbind2": "miRBind2-3UTR",
    "miraw": "miRAW",
    "random_baseline": "Random baseline",
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
    "#7F7F7F",
]


def tool_color(tool_id: str, tool_ids=None) -> str:
    if tool_ids is None:
        tool_ids = [*TOOL_IDS, "random_baseline"]
    try:
        index = list(tool_ids).index(tool_id)
    except ValueError:
        index = sum(ord(char) for char in str(tool_id))
    return TOOL_PALETTE[index % len(TOOL_PALETTE)]


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
    return merged


def expected_effect(frame: pd.DataFrame) -> pd.Series:
    logfc = pd.to_numeric(frame["logFC"], errors="coerce")
    perturbation = frame.get("perturbation", pd.Series("", index=frame.index)).astype(str).str.upper().fillna("")
    effect = logfc.abs().copy()
    oe = perturbation.isin(["OE", "OVEREXPRESSION"])
    kdko = perturbation.isin(["KO", "KD", "KNOCKOUT", "KNOCKDOWN"])
    effect.loc[oe] = -logfc.loc[oe]
    effect.loc[kdko] = logfc.loc[kdko]
    return effect


def usable_joined(frame: pd.DataFrame, *, fdr: float, effect_threshold: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["logFC_num"] = pd.to_numeric(frame["logFC"], errors="coerce")
    frame["FDR_num"] = pd.to_numeric(frame["FDR"], errors="coerce")
    keep = frame["logFC_num"].notna() & frame["FDR_num"].notna() & (frame["FDR_num"] >= 0.0) & (frame["FDR_num"] <= 1.0)
    frame = frame.loc[keep].copy()
    frame["expected_effect"] = expected_effect(frame)
    frame["is_positive"] = (frame["FDR_num"] < fdr) & (frame["expected_effect"] > effect_threshold)
    return frame


def seed_for_dataset(dataset_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{dataset_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def dataset_id_from_joined(frame: pd.DataFrame, path: pathlib.Path) -> str:
    if "dataset_id" in frame.columns and not frame.empty:
        return str(frame["dataset_id"].iloc[0])
    return path.parent.name


def random_baseline_metrics(report_dir: pathlib.Path, *, fdr: float, effect_threshold: float, seed: int) -> pd.DataFrame:
    rows = []
    for path in joined_paths(report_dir):
        raw = pd.read_csv(path, sep="\t")
        dataset_id = dataset_id_from_joined(raw, path)
        frame = usable_joined(raw, fdr=fdr, effect_threshold=effect_threshold)
        if frame.empty:
            continue
        rng = np.random.default_rng(seed_for_dataset(dataset_id, seed))
        scores = rng.random(len(frame))
        positives = frame["is_positive"].to_numpy(bool)
        n_positive = int(positives.sum())
        n_negative = int((~positives).sum())
        if n_positive and n_negative:
            auroc = float(roc_auc_score(positives, scores))
        else:
            auroc = np.nan
        if n_positive:
            aps = float(average_precision_score(positives, scores))
            pr_auc = aps
            positive_coverage = 1.0
        else:
            aps = np.nan
            pr_auc = np.nan
            positive_coverage = np.nan
        spearman = float(pd.Series(scores).corr(pd.Series(frame["expected_effect"].to_numpy(float)), method="spearman"))
        row = {
            "dataset_id": dataset_id,
            "tool_id": "random_baseline",
            "predictor": TOOL_LABELS["random_baseline"],
            "coverage": 1.0,
            "positive_coverage": positive_coverage,
            "aps": aps,
            "pr_auc": pr_auc,
            "auroc": auroc,
            "spearman": spearman,
            "random_seed": seed,
            "n_usable_genes": int(len(frame)),
            "n_positive_genes": n_positive,
        }
        rows.append(row)
    if not rows:
        raise ValueError(f"No usable joined rows found below {report_dir}")
    return pd.DataFrame(rows)


def add_random_baseline(per_dataset: pd.DataFrame, random_rows: pd.DataFrame) -> pd.DataFrame:
    per_dataset = per_dataset.copy()
    if "predictor" not in per_dataset.columns:
        per_dataset.insert(per_dataset.columns.get_loc("tool_id") + 1, "predictor", per_dataset["tool_id"].map(TOOL_LABELS).fillna(per_dataset["tool_id"]))
    random_rows = random_rows.reindex(columns=per_dataset.columns.union(random_rows.columns, sort=False))
    per_dataset = per_dataset.reindex(columns=random_rows.columns)
    return pd.concat([per_dataset, random_rows], ignore_index=True)


def metric_rows_for_plots(per_dataset: pd.DataFrame) -> pd.DataFrame:
    id_cols = [col for col in per_dataset.columns if col not in set(SUPPLEMENTARY_METRICS)]
    long = per_dataset.melt(id_vars=id_cols, value_vars=[m for m in MAIN_METRICS if m in per_dataset.columns], var_name="metric", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.dropna(subset=["value"])


def save_figure(fig, out_path: pathlib.Path) -> pathlib.Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_cross_dataset_distributions(metric_rows: pd.DataFrame, figures_dir: pathlib.Path) -> pathlib.Path:
    plot_tools = [*TOOL_IDS, "random_baseline"]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2))
    for ax, metric in zip(axes.ravel(), MAIN_METRICS):
        sub = metric_rows[metric_rows["metric"] == metric]
        data = [sub.loc[sub["tool_id"] == tool_id, "value"].dropna().to_numpy(float) for tool_id in plot_tools]
        labels = [TOOL_LABELS[tool_id] for tool_id in plot_tools]
        box = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, tool_id in zip(box["boxes"], plot_tools):
            color = tool_color(tool_id, plot_tools)
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.28)
        for median in box["medians"]:
            median.set_color("#22303C")
            median.set_linewidth(1.4)
        for idx, (tool_id, values) in enumerate(zip(plot_tools, data), start=1):
            if values.size:
                jitter = np.linspace(-0.08, 0.08, values.size) if values.size > 1 else np.array([0.0])
                ax.scatter(np.full(values.size, idx) + jitter, values, s=18, alpha=0.72, color=tool_color(tool_id, plot_tools), edgecolors="white", linewidths=0.3)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.set_ylim(0, 1.02)
        if metric in {"coverage", "positive_coverage"}:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("Cross-dataset predictor distributions with random baseline", fontsize=15, y=1.0)
    fig.tight_layout()
    return save_figure(fig, figures_dir / "figure1_cross_dataset_distributions_with_random_baseline.png")


def write_tables(per_dataset_with_random: pd.DataFrame, tables_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    full_path = tables_dir / "table_s1_per_dataset_predictor_metrics_with_random_baseline.tsv"
    per_dataset_with_random.to_csv(full_path, sep="\t", index=False)
    metric_cols = [m for m in MAIN_METRICS if m in per_dataset_with_random.columns]
    summary = per_dataset_with_random.groupby(["tool_id", "predictor"])[metric_cols].agg(["mean", "median"]).reset_index()
    summary.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns]
    summary_path = tables_dir / "table1_cross_dataset_predictor_summary_with_random_baseline.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    return {"per_dataset_with_random": full_path, "summary_with_random": summary_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create metric distributions with a random baseline predictor.")
    parser.add_argument("--report-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("manuscript_assets"))
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--effect-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    figures_dir = args.out_dir / "figures"
    tables_dir = args.out_dir / "tables"
    per_dataset = load_per_dataset_metrics(args.report_dir)
    random_rows = random_baseline_metrics(
        args.report_dir,
        fdr=args.fdr_threshold,
        effect_threshold=args.effect_threshold,
        seed=args.seed,
    )
    combined = add_random_baseline(per_dataset, random_rows)
    outputs = write_tables(combined, tables_dir)
    outputs["figure_with_random"] = plot_cross_dataset_distributions(metric_rows_for_plots(combined), figures_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
