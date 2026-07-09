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
BOXPLOT_METRICS = ["coverage", "positive_coverage", "aps", "pr_auc", "auroc", "spearman"]
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


def cdf_usable_joined(frame: pd.DataFrame) -> pd.DataFrame:
    """Match the top_100_effect_cdfs input rule: usable logFC rows, no FDR cutoff."""
    frame = frame.copy()
    frame["logFC_num"] = pd.to_numeric(frame["logFC"], errors="coerce")
    frame = frame.loc[frame["logFC_num"].notna()].copy()
    frame["expected_effect"] = expected_effect(frame)
    return frame


def seed_for_dataset(dataset_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{dataset_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def dataset_id_from_joined(frame: pd.DataFrame, path: pathlib.Path) -> str:
    if "dataset_id" in frame.columns and not frame.empty:
        return str(frame["dataset_id"].iloc[0])
    return path.parent.name


def random_scores_for_dataset(dataset_id: str, n_rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed_for_dataset(dataset_id, seed))
    return rng.random(n_rows)


def random_baseline_metrics(report_dir: pathlib.Path, *, fdr: float, effect_threshold: float, seed: int) -> pd.DataFrame:
    rows = []
    for path in joined_paths(report_dir):
        raw = pd.read_csv(path, sep="\t")
        dataset_id = dataset_id_from_joined(raw, path)
        frame = usable_joined(raw, fdr=fdr, effect_threshold=effect_threshold)
        if frame.empty:
            continue
        scores = random_scores_for_dataset(dataset_id, len(frame), seed)
        positives = frame["is_positive"].to_numpy(bool)
        n_positive = int(positives.sum())
        n_negative = int((~positives).sum())
        auroc = float(roc_auc_score(positives, scores)) if n_positive and n_negative else np.nan
        if n_positive:
            aps = float(average_precision_score(positives, scores))
            pr_auc = aps
            positive_coverage = 1.0
        else:
            aps = np.nan
            pr_auc = np.nan
            positive_coverage = np.nan
        spearman = float(pd.Series(scores).corr(pd.Series(frame["expected_effect"].to_numpy(float)), method="spearman"))
        rows.append(
            {
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
        )
    if not rows:
        raise ValueError(f"No usable joined rows found below {report_dir}")
    return pd.DataFrame(rows)


def _top_effect_row(dataset_id: str, tool_id: str, values: pd.Series, *, top_k: int, n_scored_genes: int) -> dict:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "dataset_id": dataset_id,
        "tool_id": tool_id,
        "predictor": TOOL_LABELS[tool_id],
        "top_k": int(top_k),
        "n_top_genes": int(values.size),
        "n_scored_genes": int(n_scored_genes),
        "top_effect_mean": float(values.mean()) if values.size else np.nan,
        "top_effect_med": float(values.median()) if values.size else np.nan,
        "top_effect_q25": float(values.quantile(0.25)) if values.size else np.nan,
        "top_effect_q75": float(values.quantile(0.75)) if values.size else np.nan,
    }


def topk_effect_summary_rows(report_dir: pathlib.Path, *, seed: int, top_k: int) -> pd.DataFrame:
    """Summarize the same top-N effect distributions shown in top_100_effect_cdfs.

    The `top_effect_med` column is the median perturbation-aware expected effect
    among the top-N scored genes for one dataset and one predictor. This is the
    scalar summary shown as "med" in each per-dataset CDF legend.
    """
    rows = []
    for path in joined_paths(report_dir):
        raw = pd.read_csv(path, sep="\t")
        dataset_id = dataset_id_from_joined(raw, path)
        frame = cdf_usable_joined(raw)
        if frame.empty:
            continue
        for tool_id in TOOL_IDS:
            score_col = f"score_{tool_id}"
            if score_col not in frame.columns:
                continue
            scored = frame.loc[frame[score_col].notna(), ["gene_id", "expected_effect", score_col]].copy()
            if scored.empty:
                continue
            scored[score_col] = pd.to_numeric(scored[score_col], errors="coerce")
            scored = scored.dropna(subset=[score_col])
            if scored.empty:
                continue
            sort_cols = [score_col]
            ascending = [False]
            if "gene_id" in scored.columns:
                sort_cols.append("gene_id")
                ascending.append(True)
            scored = scored.sort_values(sort_cols, ascending=ascending, kind="mergesort")
            top_values = scored["expected_effect"].head(int(top_k))
            rows.append(_top_effect_row(dataset_id, tool_id, top_values, top_k=top_k, n_scored_genes=len(scored)))

        random_scores = pd.Series(random_scores_for_dataset(dataset_id, len(frame), seed), index=frame.index)
        random_scored = frame.loc[:, ["gene_id", "expected_effect"]].copy()
        random_scored["random_score"] = random_scores
        random_scored = random_scored.sort_values(["random_score", "gene_id"], ascending=[False, True], kind="mergesort")
        random_top_values = random_scored["expected_effect"].head(int(top_k))
        rows.append(
            _top_effect_row(
                dataset_id,
                "random_baseline",
                random_top_values,
                top_k=top_k,
                n_scored_genes=len(random_scored),
            )
        )
    if not rows:
        raise ValueError(f"No top-{top_k} effect summary rows were created below {report_dir}")
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
    value_vars = [metric for metric in BOXPLOT_METRICS if metric in per_dataset.columns]
    long = per_dataset.melt(id_vars=id_cols, value_vars=value_vars, var_name="metric", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.dropna(subset=["value"])


def save_figure(fig, out_path: pathlib.Path) -> pathlib.Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_box_data(ax, data: list[np.ndarray], labels: list[str], plot_tools: list[str]) -> None:
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
            ax.scatter(
                np.full(values.size, idx) + jitter,
                values,
                s=18,
                alpha=0.72,
                color=tool_color(tool_id, plot_tools),
                edgecolors="white",
                linewidths=0.3,
            )
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def plot_cross_dataset_distributions(metric_rows: pd.DataFrame, figures_dir: pathlib.Path) -> pathlib.Path:
    plot_tools = [*TOOL_IDS, "random_baseline"]
    labels = [TOOL_LABELS[tool_id] for tool_id in plot_tools]
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.8))
    layout = {
        "coverage": axes[0, 0],
        "positive_coverage": axes[1, 0],
        "aps": axes[0, 1],
        "pr_auc": axes[1, 1],
        "auroc": axes[0, 2],
        "spearman": axes[1, 2],
    }
    for metric, ax in layout.items():
        sub = metric_rows[metric_rows["metric"] == metric]
        data = [sub.loc[sub["tool_id"] == tool_id, "value"].dropna().to_numpy(float) for tool_id in plot_tools]
        plot_box_data(ax, data, labels, plot_tools)
        ax.set_title(METRIC_LABELS[metric])
        if metric == "spearman":
            ax.axhline(0, color="#555555", linewidth=0.8, alpha=0.8)
            ax.set_ylim(-1.02, 1.02)
        else:
            ax.set_ylim(0, 1.02)
        if metric in {"coverage", "positive_coverage"}:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    fig.suptitle("Cross-dataset predictor metric distributions with random baseline", fontsize=15, y=1.0)
    fig.tight_layout()
    return save_figure(fig, figures_dir / "figure1_cross_dataset_six_metric_boxplots_with_random_baseline.png")


def plot_topk_effect_median_boxplot(topk_rows: pd.DataFrame, figures_dir: pathlib.Path, *, top_k: int) -> pathlib.Path:
    plot_tools = [*TOOL_IDS, "random_baseline"]
    labels = [TOOL_LABELS[tool_id] for tool_id in plot_tools]
    data = [topk_rows.loc[topk_rows["tool_id"] == tool_id, "top_effect_med"].dropna().to_numpy(float) for tool_id in plot_tools]
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    plot_box_data(ax, data, labels, plot_tools)
    ax.axhline(0, color="#555555", linewidth=0.8, alpha=0.8)
    ax.set_title(f"Median perturbation-aware effect among top {top_k} predictions")
    ax.set_ylabel(f"Median expected effect in top {top_k}")
    fig.tight_layout()
    return save_figure(fig, figures_dir / f"top{top_k}_effect_median_boxplot_with_random_baseline.png")


def write_tables(per_dataset_with_random: pd.DataFrame, topk_effect_rows: pd.DataFrame, tables_dir: pathlib.Path, *, top_k: int) -> dict[str, pathlib.Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    full_path = tables_dir / "table_s1_per_dataset_predictor_metrics_with_random_baseline.tsv"
    per_dataset_with_random.to_csv(full_path, sep="\t", index=False)

    metric_cols = [metric for metric in BOXPLOT_METRICS if metric in per_dataset_with_random.columns]
    summary = per_dataset_with_random.groupby(["tool_id", "predictor"])[metric_cols].agg(["mean", "median"]).reset_index()
    summary.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns]
    summary_path = tables_dir / "table1_cross_dataset_predictor_summary_with_random_baseline.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)

    topk_path = tables_dir / f"top{top_k}_effect_summary_per_dataset_with_random_baseline.tsv"
    topk_effect_rows.to_csv(topk_path, sep="\t", index=False)
    topk_summary = topk_effect_rows.groupby(["tool_id", "predictor"])[
        ["top_effect_mean", "top_effect_med", "top_effect_q25", "top_effect_q75"]
    ].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    topk_summary.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in topk_summary.columns]
    topk_summary_path = tables_dir / f"top{top_k}_effect_summary_with_random_baseline.tsv"
    topk_summary.to_csv(topk_summary_path, sep="\t", index=False)

    return {
        "per_dataset_with_random": full_path,
        "summary_with_random": summary_path,
        "topk_effect_per_dataset": topk_path,
        "topk_effect_summary": topk_summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create metric distributions with a random baseline predictor.")
    parser.add_argument("--report-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("manuscript_assets"))
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--effect-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=100)
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
    topk_effect_rows = topk_effect_summary_rows(args.report_dir, seed=args.seed, top_k=args.top_k)

    outputs = write_tables(combined, topk_effect_rows, tables_dir, top_k=args.top_k)
    outputs["six_metric_boxplots"] = plot_cross_dataset_distributions(metric_rows_for_plots(combined), figures_dir)
    outputs["topk_effect_median_boxplot"] = plot_topk_effect_median_boxplot(topk_effect_rows, figures_dir, top_k=args.top_k)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
