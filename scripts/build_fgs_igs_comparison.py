"""Compare predictor performance on the Full Gene Set (FGS) and Intersection Gene Set (IGS).

FGS retains every usable gene in each dataset. Missing predictor scores are kept
and assigned the lowest rank so that incomplete coverage is penalized.

IGS restricts each dataset to genes scored by all selected predictors. It is a
controlled comparison on a shared gene universe, but it can be compositionally
biased relative to the full usable gene set.
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

TOOL_IDS = ["targetscan", "mirdb_mirtarget", "microt_cnn", "mirbind2", "miraw"]
TOOL_LABELS = {
    "targetscan": "TargetScan",
    "mirdb_mirtarget": "miRDB",
    "microt_cnn": "microT-CNN",
    "mirbind2": "miRBind2-3UTR",
    "miraw": "miRAW",
}
METRICS = ["aps", "pr_auc", "auroc", "spearman"]
METRIC_LABELS = {
    "aps": "APS",
    "pr_auc": "PR-AUC",
    "auroc": "AUROC",
    "spearman": "Spearman",
}
TOOL_PALETTE = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD"]


def joined_paths(report_dir: pathlib.Path) -> list[pathlib.Path]:
    paths = sorted(report_dir.glob("**/joined.tsv"))
    if not paths:
        raise FileNotFoundError(f"No joined.tsv files found below {report_dir}")
    return paths


def dataset_id_from_joined(frame: pd.DataFrame, path: pathlib.Path) -> str:
    if "dataset_id" in frame.columns and not frame.empty:
        return str(frame["dataset_id"].iloc[0])
    return path.parent.name


def expected_effect(frame: pd.DataFrame) -> pd.Series:
    logfc = pd.to_numeric(frame["logFC"], errors="coerce")
    perturbation = frame.get("perturbation", pd.Series("", index=frame.index)).astype(str).str.upper().fillna("")
    effect = logfc.abs().copy()
    oe = perturbation.isin(["OE", "OVEREXPRESSION"])
    kdko = perturbation.isin(["KO", "KD", "KNOCKOUT", "KNOCKDOWN"])
    effect.loc[oe] = -logfc.loc[oe]
    effect.loc[kdko] = logfc.loc[kdko]
    return effect


def usable_joined(frame: pd.DataFrame, *, fdr_threshold: float, effect_threshold: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["logFC_num"] = pd.to_numeric(frame["logFC"], errors="coerce")
    frame["FDR_num"] = pd.to_numeric(frame["FDR"], errors="coerce")
    keep = frame["logFC_num"].notna() & frame["FDR_num"].notna() & (frame["FDR_num"] >= 0.0) & (frame["FDR_num"] <= 1.0)
    frame = frame.loc[keep].copy()
    frame["expected_effect"] = expected_effect(frame)
    frame["is_positive"] = ((frame["FDR_num"] < fdr_threshold) & (frame["expected_effect"] > effect_threshold)).astype(int)
    return frame


def fgs_scores(frame: pd.DataFrame, score_col: str) -> pd.Series:
    scores = pd.to_numeric(frame[score_col], errors="coerce")
    observed = scores.dropna()
    if observed.empty:
        return scores
    lowest = float(observed.min())
    span = max(float(observed.max()) - lowest, 1.0)
    missing_score = lowest - span * 1e-6
    return scores.fillna(missing_score)


def metric_values(y_true: pd.Series, scores: pd.Series) -> dict[str, float]:
    y = pd.Series(y_true).astype(int)
    s = pd.to_numeric(scores, errors="coerce")
    keep = y.notna() & s.notna()
    y = y.loc[keep]
    s = s.loc[keep]
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if len(y) == 0 or positives == 0 or negatives == 0:
        return {"aps": np.nan, "pr_auc": np.nan, "auroc": np.nan, "spearman": np.nan}
    precision, recall, _ = precision_recall_curve(y, s)
    return {
        "aps": float(average_precision_score(y, s)),
        "pr_auc": float(auc(recall, precision)),
        "auroc": float(roc_auc_score(y, s)),
        "spearman": float(s.corr(pd.Series(y_true.index.map(lambda idx: np.nan), index=y_true.index), method="spearman")) if False else np.nan,
    }


def evaluate_score_frame(frame: pd.DataFrame, *, dataset_id: str, tool_id: str, universe: str, score_col: str, scores: pd.Series) -> dict[str, float | str | int]:
    y_true = frame["is_positive"].astype(int)
    scores = pd.Series(scores, index=frame.index)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    out = {
        "dataset_id": dataset_id,
        "tool_id": tool_id,
        "predictor": TOOL_LABELS[tool_id],
        "universe": universe,
        "n_genes": int(len(frame)),
        "n_positive": positives,
        "n_negative": negatives,
        "n_observed_scores": int(pd.to_numeric(frame[score_col], errors="coerce").notna().sum()),
        "n_missing_scores": int(pd.to_numeric(frame[score_col], errors="coerce").isna().sum()),
    }
    y = y_true
    s = pd.to_numeric(scores, errors="coerce")
    keep = y.notna() & s.notna()
    y = y.loc[keep]
    s = s.loc[keep]
    if len(y) == 0 or int(y.sum()) == 0 or int(len(y) - y.sum()) == 0:
        out.update({metric: np.nan for metric in METRICS})
        return out
    precision, recall, _ = precision_recall_curve(y, s)
    out["aps"] = float(average_precision_score(y, s))
    out["pr_auc"] = float(auc(recall, precision))
    out["auroc"] = float(roc_auc_score(y, s))
    out["spearman"] = float(s.corr(frame.loc[s.index, "expected_effect"], method="spearman"))
    return out


def fgs_igs_rows(report_dir: pathlib.Path, *, fdr_threshold: float, effect_threshold: float) -> pd.DataFrame:
    rows = []
    score_cols = [f"score_{tool_id}" for tool_id in TOOL_IDS]
    for path in joined_paths(report_dir):
        raw = pd.read_csv(path, sep="\t")
        dataset_id = dataset_id_from_joined(raw, path)
        missing_cols = [col for col in score_cols if col not in raw.columns]
        if missing_cols:
            continue
        frame = usable_joined(raw, fdr_threshold=fdr_threshold, effect_threshold=effect_threshold)
        if frame.empty:
            continue
        igs = frame.loc[frame[score_cols].notna().all(axis=1)].copy()
        for tool_id, score_col in zip(TOOL_IDS, score_cols):
            rows.append(
                evaluate_score_frame(
                    frame,
                    dataset_id=dataset_id,
                    tool_id=tool_id,
                    universe="FGS",
                    score_col=score_col,
                    scores=fgs_scores(frame, score_col),
                )
            )
            if not igs.empty:
                rows.append(
                    evaluate_score_frame(
                        igs,
                        dataset_id=dataset_id,
                        tool_id=tool_id,
                        universe="IGS",
                        score_col=score_col,
                        scores=pd.to_numeric(igs[score_col], errors="coerce"),
                    )
                )
    if not rows:
        raise ValueError(f"No FGS/IGS comparison rows were produced below {report_dir}")
    return pd.DataFrame(rows)


def paired_delta_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["dataset_id", "tool_id", "predictor"]
    rows = []
    for metric in METRICS:
        wide = metrics.pivot_table(index=key_cols, columns="universe", values=metric, aggfunc="first").reset_index()
        if "FGS" not in wide.columns or "IGS" not in wide.columns:
            continue
        wide[f"{metric}_delta_IGS_minus_FGS"] = wide["IGS"] - wide["FGS"]
        wide["metric"] = metric
        wide = wide.rename(columns={"FGS": "fgs_value", "IGS": "igs_value", f"{metric}_delta_IGS_minus_FGS": "delta_IGS_minus_FGS"})
        rows.append(wide[[*key_cols, "metric", "fgs_value", "igs_value", "delta_IGS_minus_FGS"]])
    if not rows:
        return pd.DataFrame(columns=[*key_cols, "metric", "fgs_value", "igs_value", "delta_IGS_minus_FGS"])
    return pd.concat(rows, ignore_index=True)


def summary_table(metrics: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    metric_summary = metrics.groupby(["universe", "tool_id", "predictor"])[METRICS].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    metric_summary.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in metric_summary.columns]
    if deltas.empty:
        return metric_summary
    delta_summary = deltas.groupby(["metric", "tool_id", "predictor"])["delta_IGS_minus_FGS"].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    delta_summary = delta_summary.rename(columns={stat: f"delta_{stat}" for stat in ["count", "mean", "median", "std", "min", "max"]})
    return metric_summary, delta_summary


def tool_color(tool_id: str) -> str:
    return TOOL_PALETTE[TOOL_IDS.index(tool_id) % len(TOOL_PALETTE)]


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def plot_metric_pair(ax, metrics: pd.DataFrame, *, metric: str) -> None:
    positions = np.arange(len(TOOL_IDS), dtype=float)
    width = 0.27
    fgs_data = [metrics.loc[(metrics["tool_id"] == tool_id) & (metrics["universe"] == "FGS"), metric].dropna().to_numpy(float) for tool_id in TOOL_IDS]
    igs_data = [metrics.loc[(metrics["tool_id"] == tool_id) & (metrics["universe"] == "IGS"), metric].dropna().to_numpy(float) for tool_id in TOOL_IDS]
    fgs_box = ax.boxplot(fgs_data, positions=positions - width / 2, widths=0.22, patch_artist=True, showfliers=False)
    igs_box = ax.boxplot(igs_data, positions=positions + width / 2, widths=0.22, patch_artist=True, showfliers=False)
    for box, alpha in ((fgs_box, 0.22), (igs_box, 0.48)):
        for patch, tool_id in zip(box["boxes"], TOOL_IDS):
            color = tool_color(tool_id)
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(alpha)
        for median in box["medians"]:
            median.set_color("#22303C")
            median.set_linewidth(1.35)
    for idx, tool_id in enumerate(TOOL_IDS):
        pairs = metrics.loc[metrics["tool_id"] == tool_id].pivot_table(index="dataset_id", columns="universe", values=metric, aggfunc="first")
        if "FGS" in pairs.columns and "IGS" in pairs.columns:
            pairs = pairs.dropna(subset=["FGS", "IGS"])
            for _, row in pairs.iterrows():
                ax.plot([positions[idx] - width / 2, positions[idx] + width / 2], [row["FGS"], row["IGS"]], color=tool_color(tool_id), alpha=0.20, linewidth=0.8)
        for side, universe in ((-width / 2, "FGS"), (width / 2, "IGS")):
            vals = metrics.loc[(metrics["tool_id"] == tool_id) & (metrics["universe"] == universe), metric].dropna().to_numpy(float)
            if vals.size:
                jitter = np.linspace(-0.035, 0.035, vals.size) if vals.size > 1 else np.array([0.0])
                ax.scatter(np.full(vals.size, positions[idx] + side) + jitter, vals, s=15, alpha=0.65, color=tool_color(tool_id), edgecolors="white", linewidths=0.25, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels([TOOL_LABELS[tool_id] for tool_id in TOOL_IDS], rotation=18, ha="right")
    ax.set_title(METRIC_LABELS[metric])
    if metric == "spearman":
        ax.axhline(0, color="#555555", linewidth=0.8, alpha=0.75)
        ax.set_ylim(-1.02, 1.02)
    else:
        ax.set_ylim(0, 1.02)
    style_axis(ax)


def plot_fgs_igs_comparison(metrics: pd.DataFrame, figures_dir: pathlib.Path) -> pathlib.Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.3))
    for ax, metric in zip(axes.ravel(), ["aps", "pr_auc", "auroc", "spearman"]):
        plot_metric_pair(ax, metrics, metric=metric)
    handles = [
        plt.Line2D([0], [0], color="#22303C", marker="s", linestyle="", markersize=8, markerfacecolor="#AEBBD0", alpha=0.55, label="FGS"),
        plt.Line2D([0], [0], color="#22303C", marker="s", linestyle="", markersize=8, markerfacecolor="#637FA6", alpha=0.75, label="IGS"),
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("FGS versus IGS benchmark performance", fontsize=15, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = figures_dir / "fgs_vs_igs_metric_bump.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def write_outputs(metrics: pd.DataFrame, out_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tables_dir / "fgs_vs_igs_per_dataset_metrics.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    deltas = paired_delta_rows(metrics)
    deltas_path = tables_dir / "fgs_vs_igs_per_dataset_metric_deltas.tsv"
    deltas.to_csv(deltas_path, sep="\t", index=False)
    metric_summary, delta_summary = summary_table(metrics, deltas)
    summary_path = tables_dir / "fgs_vs_igs_metric_summary.tsv"
    metric_summary.to_csv(summary_path, sep="\t", index=False)
    delta_summary_path = tables_dir / "fgs_vs_igs_metric_delta_summary.tsv"
    delta_summary.to_csv(delta_summary_path, sep="\t", index=False)
    figure_path = plot_fgs_igs_comparison(metrics, figures_dir)
    return {
        "per_dataset_metrics": metrics_path,
        "per_dataset_deltas": deltas_path,
        "metric_summary": summary_path,
        "metric_delta_summary": delta_summary_path,
        "fgs_vs_igs_figure": figure_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FGS and IGS benchmark metrics.")
    parser.add_argument("--report-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("manuscript_assets"))
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--effect-threshold", type=float, default=1.0)
    args = parser.parse_args()

    metrics = fgs_igs_rows(args.report_dir, fdr_threshold=args.fdr_threshold, effect_threshold=args.effect_threshold)
    outputs = write_outputs(metrics, args.out_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
