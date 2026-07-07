"""Build gene-level predictability plots across FuNmiRBench datasets.

This exploratory post-processing analysis asks whether some genes are
recurrently easier to prioritize as functional miRNA targets across perturbation
experiments.

For each dataset and predictor, local ranks are computed only among scored
miRNA-gene pairs in that dataset. The script then aggregates ranks by gene and
predictor, focusing on occurrences where the gene is GT-positive for the
perturbed miRNA.

Outputs
-------
- tables/gene_predictability_by_predictor.tsv
- tables/gene_predictability_overall.tsv
- figures/gene_predictability_scatter.png/.svg
- figures/gene_predictability_heatmap.png/.svg

Example
-------
python scripts/build_gene_predictability.py \
    --report-dir results/20260706_132519 \
    --out-dir manuscript_assets \
    --min-positive-occurrences 2
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import funmirbench.evaluate as ev

TOOL_IDS = ["targetscan", "mirdb_mirtarget", "microt_cnn", "mirbind2", "miraw"]
TOOL_LABELS = {
    "targetscan": "TargetScan",
    "mirdb_mirtarget": "miRDB",
    "microt_cnn": "microT-CNN",
    "mirbind2": "miRBind2",
    "miraw": "miRAW",
}
TOOL_COLORS = {
    "targetscan": "#9467BD",
    "mirdb_mirtarget": "#FF7F0E",
    "microt_cnn": "#1F77B4",
    "mirbind2": "#D62728",
    "miraw": "#2CA02C",
}


def joined_paths(report_dir: pathlib.Path) -> list[pathlib.Path]:
    paths = sorted(report_dir.glob("**/joined.tsv"))
    if not paths:
        raise FileNotFoundError(f"No joined.tsv files found below {report_dir}")
    return paths


def save_figure(fig, out_path: pathlib.Path) -> pathlib.Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def style_axes(ax, *, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, alpha=0.25)
    ax.set_axisbelow(True)


def usable_joined(frame: pd.DataFrame, *, fdr: float, effect_threshold: float) -> pd.DataFrame:
    work = ev._filter_usable_gt_rows(frame, fdr_threshold=fdr)
    work = ev._annotate_ground_truth(work)
    work["is_positive"] = ev._positive_mask(
        work,
        fdr_threshold=fdr,
        abs_logfc_threshold=effect_threshold,
    ).astype(bool)
    return work


def local_rank_from_score(score: pd.Series) -> pd.Series:
    """Return local normalized rank in one dataset, using average ranks for ties."""
    values = pd.to_numeric(score, errors="coerce")
    ranks = values.rank(method="average", ascending=True)
    count = int(values.notna().sum())
    if count <= 0:
        return pd.Series(np.nan, index=score.index)
    if count == 1:
        return pd.Series(1.0, index=score.index, dtype=float)
    return (ranks - 1.0) / (float(count) - 1.0)


def gene_identifier(frame: pd.DataFrame) -> pd.Series:
    if "gene_symbol" in frame.columns:
        symbol = frame["gene_symbol"].astype("string")
        if "gene_id" in frame.columns:
            gene_id = frame["gene_id"].astype("string")
            return symbol.where(symbol.notna() & (symbol.str.len() > 0), gene_id)
        return symbol
    if "gene_name" in frame.columns:
        return frame["gene_name"].astype("string")
    if "gene_id" in frame.columns:
        return frame["gene_id"].astype("string")
    raise KeyError("Expected one of gene_symbol, gene_name, or gene_id in joined.tsv")


def collect_rank_occurrences(report_dir: pathlib.Path, *, fdr: float, effect_threshold: float) -> pd.DataFrame:
    rows = []
    for path in joined_paths(report_dir):
        dataset_id = path.parent.name
        frame = usable_joined(pd.read_csv(path, sep="\t"), fdr=fdr, effect_threshold=effect_threshold)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["gene"] = gene_identifier(frame)
        for tool_id in TOOL_IDS:
            score_col = f"score_{tool_id}"
            if score_col not in frame.columns:
                continue
            scored = frame[pd.to_numeric(frame[score_col], errors="coerce").notna()].copy()
            if scored.empty:
                continue
            scored["local_rank"] = local_rank_from_score(scored[score_col])
            scored = scored.dropna(subset=["local_rank", "gene"])
            if scored.empty:
                continue
            tmp = scored[["gene", "is_positive", "expected_effect", "local_rank"]].copy()
            tmp["dataset_id"] = dataset_id
            tmp["tool_id"] = tool_id
            tmp["predictor"] = TOOL_LABELS[tool_id]
            rows.append(tmp)
    if not rows:
        raise RuntimeError("No scored rank occurrences could be collected.")
    return pd.concat(rows, ignore_index=True)


def summarize_by_gene(occurrences: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def q25(x: pd.Series) -> float:
        return float(x.quantile(0.25))

    def q75(x: pd.Series) -> float:
        return float(x.quantile(0.75))

    per_predictor_rows = []
    for (tool_id, predictor, gene), sub in occurrences.groupby(["tool_id", "predictor", "gene"], dropna=False):
        pos = sub[sub["is_positive"]]
        bg = sub[~sub["is_positive"]]
        per_predictor_rows.append(
            {
                "tool_id": tool_id,
                "predictor": predictor,
                "gene": gene,
                "n_scored_occurrences": int(len(sub)),
                "n_datasets_scored": int(sub["dataset_id"].nunique()),
                "n_gt_positive_occurrences": int(len(pos)),
                "n_background_occurrences": int(len(bg)),
                "mean_local_rank_when_positive": float(pos["local_rank"].mean()) if len(pos) else np.nan,
                "median_local_rank_when_positive": float(pos["local_rank"].median()) if len(pos) else np.nan,
                "q25_local_rank_when_positive": q25(pos["local_rank"]) if len(pos) else np.nan,
                "q75_local_rank_when_positive": q75(pos["local_rank"]) if len(pos) else np.nan,
                "top10_fraction_when_positive": float((pos["local_rank"] >= 0.90).mean()) if len(pos) else np.nan,
                "top20_fraction_when_positive": float((pos["local_rank"] >= 0.80).mean()) if len(pos) else np.nan,
                "mean_local_rank_when_background": float(bg["local_rank"].mean()) if len(bg) else np.nan,
                "rank_delta_positive_minus_background": (
                    float(pos["local_rank"].mean() - bg["local_rank"].mean()) if len(pos) and len(bg) else np.nan
                ),
                "mean_expected_effect_when_positive": float(pos["expected_effect"].mean()) if len(pos) else np.nan,
            }
        )
    per_predictor = pd.DataFrame(per_predictor_rows)

    overall_rows = []
    for gene, sub in occurrences.groupby("gene", dropna=False):
        pos = sub[sub["is_positive"]]
        overall_rows.append(
            {
                "gene": gene,
                "n_predictor_scored_occurrences": int(len(sub)),
                "n_predictors": int(sub["tool_id"].nunique()),
                "n_datasets_scored_any_predictor": int(sub["dataset_id"].nunique()),
                "n_gt_positive_predictor_occurrences": int(len(pos)),
                "n_gt_positive_datasets_any_predictor": int(pos["dataset_id"].nunique()),
                "mean_local_rank_when_positive_across_predictors": float(pos["local_rank"].mean()) if len(pos) else np.nan,
                "top20_fraction_when_positive_across_predictors": float((pos["local_rank"] >= 0.80).mean()) if len(pos) else np.nan,
                "mean_expected_effect_when_positive": float(pos["expected_effect"].mean()) if len(pos) else np.nan,
            }
        )
    overall = pd.DataFrame(overall_rows)
    return per_predictor, overall


def plot_gene_predictability_scatter(per_predictor: pd.DataFrame, figures_dir: pathlib.Path, *, min_positive_occurrences: int, label_top_n: int) -> pathlib.Path:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), sharey=True)
    axes_flat = axes.ravel()
    for idx, tool_id in enumerate(TOOL_IDS):
        ax = axes_flat[idx]
        sub = per_predictor[(per_predictor["tool_id"] == tool_id) & (per_predictor["n_gt_positive_occurrences"] >= min_positive_occurrences)].copy()
        color = TOOL_COLORS[tool_id]
        if sub.empty:
            ax.text(0.5, 0.5, "No genes pass filter", ha="center", va="center", transform=ax.transAxes)
        else:
            size = 20 + 10 * np.sqrt(sub["n_scored_occurrences"].astype(float))
            ax.scatter(
                sub["n_gt_positive_occurrences"],
                sub["mean_local_rank_when_positive"],
                s=size,
                alpha=0.55,
                color=color,
                edgecolors="white",
                linewidths=0.4,
            )
            labels = sub.sort_values(
                ["mean_local_rank_when_positive", "n_gt_positive_occurrences"],
                ascending=[False, False],
            ).head(label_top_n)
            for _, row in labels.iterrows():
                ax.text(
                    row["n_gt_positive_occurrences"] + 0.03,
                    row["mean_local_rank_when_positive"],
                    str(row["gene"]),
                    fontsize=7.5,
                    alpha=0.85,
                )
        ax.set_title(TOOL_LABELS[tool_id])
        ax.set_xlabel("GT-positive occurrences")
        ax.set_ylim(0, 1.02)
        style_axes(ax)
    axes_flat[0].set_ylabel("Mean local rank when GT-positive\n(0 = weakest, 1 = strongest)")
    axes_flat[3].set_ylabel("Mean local rank when GT-positive\n(0 = weakest, 1 = strongest)")
    axes_flat[-1].axis("off")
    fig.suptitle("Gene-level predictability across perturbation datasets", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return save_figure(fig, figures_dir / "gene_predictability_scatter.png")


def plot_gene_predictability_heatmap(per_predictor: pd.DataFrame, overall: pd.DataFrame, figures_dir: pathlib.Path, *, min_positive_occurrences: int, top_genes: int) -> pathlib.Path:
    eligible = overall[overall["n_gt_positive_predictor_occurrences"] >= min_positive_occurrences].copy()
    eligible = eligible.sort_values(
        ["n_gt_positive_datasets_any_predictor", "top20_fraction_when_positive_across_predictors", "mean_local_rank_when_positive_across_predictors"],
        ascending=[False, False, False],
    ).head(top_genes)
    genes = eligible["gene"].astype(str).tolist()
    matrix = np.full((len(genes), len(TOOL_IDS)), np.nan)
    annotations = np.empty((len(genes), len(TOOL_IDS)), dtype=object)
    annotations[:] = ""
    for i, gene in enumerate(genes):
        for j, tool_id in enumerate(TOOL_IDS):
            row = per_predictor[(per_predictor["gene"].astype(str) == gene) & (per_predictor["tool_id"] == tool_id)]
            if not row.empty:
                value = row["mean_local_rank_when_positive"].iloc[0]
                if pd.notna(value):
                    matrix[i, j] = float(value)
                    annotations[i, j] = str(int(row["n_gt_positive_occurrences"].iloc[0]))

    fig_height = max(6.0, 0.32 * max(len(genes), 1) + 1.8)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(TOOL_IDS)))
    ax.set_xticklabels([TOOL_LABELS[tool_id] for tool_id in TOOL_IDS], rotation=30, ha="right")
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes)
    ax.set_title("Recurrent GT-positive genes: mean local rank by predictor")
    for i in range(len(genes)):
        for j in range(len(TOOL_IDS)):
            if annotations[i, j]:
                ax.text(j, i, annotations[i, j], ha="center", va="center", fontsize=7, color="white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean local rank when GT-positive")
    ax.set_xlabel("Number in cells = GT-positive occurrences for that predictor")
    fig.tight_layout()
    return save_figure(fig, figures_dir / "gene_predictability_heatmap.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create gene-level predictability plots across FuNmiRBench datasets.")
    parser.add_argument("--report-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("manuscript_assets"))
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--effect-threshold", type=float, default=1.0)
    parser.add_argument("--min-positive-occurrences", type=int, default=2)
    parser.add_argument("--label-top-n", type=int, default=8)
    parser.add_argument("--heatmap-top-genes", type=int, default=30)
    args = parser.parse_args()

    figures_dir = args.out_dir / "figures"
    tables_dir = args.out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    occurrences = collect_rank_occurrences(
        args.report_dir,
        fdr=args.fdr_threshold,
        effect_threshold=args.effect_threshold,
    )
    occurrence_path = tables_dir / "gene_predictability_rank_occurrences.tsv"
    occurrences.to_csv(occurrence_path, sep="\t", index=False)

    per_predictor, overall = summarize_by_gene(occurrences)
    per_predictor_path = tables_dir / "gene_predictability_by_predictor.tsv"
    overall_path = tables_dir / "gene_predictability_overall.tsv"
    per_predictor.to_csv(per_predictor_path, sep="\t", index=False)
    overall.to_csv(overall_path, sep="\t", index=False)

    scatter_path = plot_gene_predictability_scatter(
        per_predictor,
        figures_dir,
        min_positive_occurrences=args.min_positive_occurrences,
        label_top_n=args.label_top_n,
    )
    heatmap_path = plot_gene_predictability_heatmap(
        per_predictor,
        overall,
        figures_dir,
        min_positive_occurrences=args.min_positive_occurrences,
        top_genes=args.heatmap_top_genes,
    )

    print(f"occurrences: {occurrence_path}")
    print(f"by_predictor: {per_predictor_path}")
    print(f"overall: {overall_path}")
    print(f"scatter: {scatter_path}")
    print(f"heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
