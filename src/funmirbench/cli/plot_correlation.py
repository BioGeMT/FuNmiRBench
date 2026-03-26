"""
Evaluate joined GT/prediction tables and generate downstream benchmark outputs.

Supported outputs:
1. Score vs logFC scatter + correlation, one per predictor
2. PR curve / PR-AUC, one per predictor
3. ROC curve / AUROC, one per predictor
4. Algorithms vs genes heatmap, one per dataset
5. Predictor x predictor correlation heatmap, one per dataset
6. APS per experiment table
7. Spearman per experiment table
8. AUROC per experiment table
"""

from __future__ import annotations

import argparse
import math
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


SCORE_PREFIX = "score_"
DEFAULT_SCORE_CANDIDATES = [
    "score",
    "prediction_score",
    "canonical_score",
    "mock_score",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate downstream benchmark plots and reports from a joined TSV."
    )
    p.add_argument("--joined-tsv", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--score-col", default=None, help="Optional single score column to evaluate.")
    p.add_argument("--fdr-col", default="FDR", help="FDR column name (default: FDR).")
    p.add_argument("--logfc-col", default="logFC", help="logFC column name (default: logFC).")
    p.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Unused legacy option retained for CLI compatibility.",
    )
    p.add_argument("--fdr-threshold", type=float, default=0.05)
    p.add_argument("--abs-logfc-threshold", type=float, default=1.0)
    p.add_argument("--predictor-top-fraction", type=float, default=0.10)
    p.add_argument("--dataset-id", default=None)
    p.add_argument("--mirna", default=None)
    p.add_argument("--cell-line", default=None)
    p.add_argument("--perturbation", default=None)
    p.add_argument("--geo-accession", default=None)
    return p.parse_args()


def _safe_neglog10(series: pd.Series) -> pd.Series:
    clipped = series.astype(float).clip(lower=1e-300)
    return -clipped.map(math.log10)


def _normalize_scores(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    min_value = values.min(skipna=True)
    max_value = values.max(skipna=True)
    if pd.isna(min_value) or pd.isna(max_value):
        return pd.Series(float("nan"), index=series.index)
    if max_value == min_value:
        return pd.Series(0.0, index=series.index)
    return (values - min_value) / (max_value - min_value)


def _tool_id_from_score_col(score_col: str) -> str:
    if score_col.startswith(SCORE_PREFIX):
        return score_col[len(SCORE_PREFIX):]
    return score_col


def _detect_score_cols(df: pd.DataFrame, requested: str | None) -> list[str]:
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(f"Requested score column {requested!r} not found.")
        return [requested]

    prefixed = [col for col in df.columns if str(col).startswith(SCORE_PREFIX)]
    if prefixed:
        return sorted(prefixed)

    for col in DEFAULT_SCORE_CANDIDATES:
        if col in df.columns:
            return [col]

    numeric_candidates = [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
        and col not in {"logFC", "FDR", "PValue", "pvalue", "adj.P.Val"}
    ]
    if len(numeric_candidates) == 1:
        return numeric_candidates

    raise ValueError(
        "Could not detect prediction score column automatically. "
        f"Available columns: {list(df.columns)}"
    )


def _prepare_scored_frame(
    joined: pd.DataFrame,
    *,
    score_col: str,
    fdr_threshold: float,
    abs_logfc_threshold: float,
) -> tuple[pd.DataFrame, int]:
    required_cols = {"gene_id", "logFC", "FDR", score_col}
    missing = [col for col in required_cols if col not in joined.columns]
    if missing:
        raise ValueError(f"Joined table missing required columns: {missing}")

    keep_cols = ["gene_id", "logFC", "FDR", score_col]
    for optional in ("dataset_id", "mirna", "PValue"):
        if optional in joined.columns:
            keep_cols.append(optional)

    keep = joined[keep_cols].copy()
    keep = keep[keep["logFC"].notna() & keep["FDR"].notna()].copy()
    keep = keep[keep["FDR"].astype(float) > 0].copy()
    if keep.empty:
        raise ValueError(f"No usable rows remain for {score_col}.")

    filled_zero_count = int(keep[score_col].isna().sum())
    keep[score_col] = keep[score_col].fillna(0.0).astype(float)
    keep["logFC"] = keep["logFC"].astype(float)
    keep["FDR"] = keep["FDR"].astype(float)
    keep["abs_logFC"] = keep["logFC"].abs()
    keep["neglog10_FDR"] = _safe_neglog10(keep["FDR"])
    keep["is_positive"] = (
        (keep["FDR"] < fdr_threshold)
        & (keep["abs_logFC"] > abs_logfc_threshold)
    ).astype(int)

    positives = int(keep["is_positive"].sum())
    negatives = int(len(keep) - positives)
    if positives == 0:
        raise ValueError(f"No positives remain for {score_col}.")
    if negatives == 0:
        raise ValueError(f"No negatives remain for {score_col}.")
    return keep, filled_zero_count


def _plot_scatter_with_correlation(
    df: pd.DataFrame,
    *,
    score_col: str,
    dataset_id: str,
    tool_id: str,
    out_path: pathlib.Path,
) -> tuple[float, float]:
    pearson = float(df[score_col].corr(df["logFC"], method="pearson"))
    spearman = float(df[score_col].corr(df["logFC"], method="spearman"))

    plt.figure(figsize=(7, 5))
    plt.scatter(df[score_col], df["logFC"], alpha=0.5, s=12)
    plt.xlabel(f"{tool_id} score")
    plt.ylabel("logFC")
    plt.title(f"{dataset_id}: {tool_id} score vs logFC")
    plt.text(
        0.02,
        0.98,
        f"pearson={pearson:.4f}\nspearman={spearman:.4f}",
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return pearson, spearman


def _plot_pr_curve(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    dataset_id: str,
    tool_id: str,
    out_path: pathlib.Path,
) -> tuple[float, float]:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    aps = average_precision_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AUPRC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{dataset_id}: {tool_id} PR curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return float(pr_auc), float(aps)


def _plot_roc_curve(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    dataset_id: str,
    tool_id: str,
    out_path: pathlib.Path,
) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUROC = {auroc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{dataset_id}: {tool_id} ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return float(auroc)


def _plot_algorithms_vs_genes_heatmap(
    joined: pd.DataFrame,
    *,
    score_cols: list[str],
    tool_ids: list[str],
    dataset_id: str,
    out_path: pathlib.Path,
    fdr_threshold: float,
    abs_logfc_threshold: float,
) -> None:
    work = joined[["gene_id", "logFC", "FDR", *score_cols]].copy()
    work = work[work["logFC"].notna() & work["FDR"].notna()].copy()
    work["logFC"] = work["logFC"].astype(float)
    work["FDR"] = work["FDR"].astype(float)
    work["abs_logFC"] = work["logFC"].abs()
    work["is_positive"] = (
        (work["FDR"] < fdr_threshold)
        & (work["abs_logFC"] > abs_logfc_threshold)
    ).astype(int)
    work = work.sort_values(
        ["is_positive", "FDR", "abs_logFC"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    normalized = pd.DataFrame(
        {
            score_col: _normalize_scores(work[score_col]).fillna(0.0)
            for score_col in score_cols
        }
    )

    max_abs_logfc = max(float(work["abs_logFC"].max()), 1.0)
    figure_height = max(6, min(22, 0.16 * len(work)))
    figure_width = max(8, 3 + len(tool_ids))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(figure_width, figure_height),
        gridspec_kw={"width_ratios": [0.4, 0.5, max(2, len(tool_ids))]},
    )

    axes[0].imshow(work["is_positive"].to_numpy().reshape(-1, 1), aspect="auto", cmap="Greys")
    axes[0].set_title("GT")
    axes[0].set_xticks([0])
    axes[0].set_xticklabels(["pos"], rotation=90)

    axes[1].imshow(
        work["logFC"].to_numpy().reshape(-1, 1),
        aspect="auto",
        cmap="coolwarm",
        vmin=-max_abs_logfc,
        vmax=max_abs_logfc,
    )
    axes[1].set_title("logFC")
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(["logFC"], rotation=90)

    heat = axes[2].imshow(normalized.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title(f"{dataset_id}: algorithms vs genes")
    axes[2].set_xticks(range(len(tool_ids)))
    axes[2].set_xticklabels(tool_ids, rotation=45, ha="right")

    if len(work) <= 40:
        labels = work["gene_id"].tolist()
        for axis in axes:
            axis.set_yticks(range(len(labels)))
            axis.set_yticklabels(labels, fontsize=7)
    else:
        for axis in axes:
            axis.set_yticks([])

    fig.colorbar(heat, ax=axes[2], fraction=0.046, pad=0.04, label="normalized score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _top_fraction_mask(series: pd.Series, fraction: float) -> pd.Series:
    threshold = series.quantile(1.0 - fraction)
    return series >= threshold


def _plot_predictor_correlation_heatmap(
    joined: pd.DataFrame,
    *,
    score_cols: list[str],
    tool_ids: list[str],
    dataset_id: str,
    out_path: pathlib.Path,
    top_fraction: float,
) -> pd.DataFrame:
    normalized = {
        tool_id: _normalize_scores(joined[score_col]).fillna(0.0)
        for tool_id, score_col in zip(tool_ids, score_cols)
    }
    top_masks = {
        tool_id: _top_fraction_mask(normalized[tool_id], top_fraction)
        for tool_id in tool_ids
    }

    matrix = pd.DataFrame(index=tool_ids, columns=tool_ids, dtype=float)
    for tool_a in tool_ids:
        for tool_b in tool_ids:
            union_mask = top_masks[tool_a] | top_masks[tool_b]
            if int(union_mask.sum()) < 2:
                corr = 1.0 if tool_a == tool_b else float("nan")
            else:
                corr = float(
                    normalized[tool_a][union_mask].corr(
                        normalized[tool_b][union_mask],
                        method="spearman",
                    )
                )
            matrix.loc[tool_a, tool_b] = corr

    plt.figure(figsize=(max(5, len(tool_ids) * 1.5), max(4, len(tool_ids) * 1.2)))
    image = plt.imshow(matrix.astype(float).to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(tool_ids)), tool_ids, rotation=45, ha="right")
    plt.yticks(range(len(tool_ids)), tool_ids)
    plt.title(f"{dataset_id}: predictor correlation (top {int(top_fraction * 100)}%)")
    for i, tool_a in enumerate(tool_ids):
        for j, tool_b in enumerate(tool_ids):
            value = matrix.loc[tool_a, tool_b]
            label = "nan" if pd.isna(value) else f"{value:.2f}"
            plt.text(j, i, label, ha="center", va="center", color="black")
    plt.colorbar(image, label="Spearman")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return matrix


def _write_tool_report(
    *,
    dataset_id: str,
    mirna: str | None,
    cell_line: str | None,
    perturbation: str | None,
    geo_accession: str | None,
    de_table_path: str | None,
    joined_tsv: pathlib.Path | None,
    tool_id: str,
    canonical_tsv_path: str | None,
    metrics: dict[str, float],
    out_path: pathlib.Path,
    scatter_png: pathlib.Path,
    pr_png: pathlib.Path,
    roc_png: pathlib.Path,
    filled_zero_count: int,
) -> None:
    lines = [
        f"dataset_id: {dataset_id}",
        f"mirna: {mirna or 'NA'}",
        f"cell_line: {cell_line or 'NA'}",
        f"perturbation: {perturbation or 'NA'}",
        f"geo_accession: {geo_accession or 'NA'}",
        f"de_table_path: {de_table_path or 'NA'}",
        f"joined_tsv: {joined_tsv or 'NA'}",
        f"tool_id: {tool_id}",
        f"canonical_tsv_path: {canonical_tsv_path or 'NA'}",
        f"filled_missing_scores_with_zero: {filled_zero_count}",
        "",
        "metrics:",
        f"  rows_used: {int(metrics['rows_used'])}",
        f"  positives: {int(metrics['positives'])}",
        f"  negatives: {int(metrics['negatives'])}",
        f"  pearson: {metrics['pearson']:.6f}",
        f"  spearman: {metrics['spearman']:.6f}",
        f"  aps: {metrics['aps']:.6f}",
        f"  pr_auc: {metrics['pr_auc']:.6f}",
        f"  auroc: {metrics['auroc']:.6f}",
        "",
        "plots:",
        f"  scatter: {scatter_png}",
        f"  pr_curve: {pr_png}",
        f"  roc_curve: {roc_png}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_joined_dataframe(
    joined: pd.DataFrame,
    *,
    plots_dir: pathlib.Path,
    reports_dir: pathlib.Path,
    fdr_threshold: float,
    abs_logfc_threshold: float,
    predictor_top_fraction: float,
    dataset_id: str | None = None,
    mirna: str | None = None,
    cell_line: str | None = None,
    perturbation: str | None = None,
    geo_accession: str | None = None,
    de_table_path: str | None = None,
    joined_tsv: pathlib.Path | None = None,
    canonical_paths: dict[str, str] | None = None,
    score_cols: list[str] | None = None,
) -> dict[str, Any]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if score_cols is None:
        score_cols = _detect_score_cols(joined, None)

    dataset_id = dataset_id or (
        str(joined["dataset_id"].iloc[0]) if "dataset_id" in joined.columns else "NA"
    )
    mirna = mirna or (
        str(joined["mirna"].iloc[0]) if "mirna" in joined.columns else None
    )
    tool_ids = [_tool_id_from_score_col(score_col) for score_col in score_cols]

    metric_rows: list[dict[str, Any]] = []
    dataset_plots: dict[str, str] = {}

    for score_col, tool_id in zip(score_cols, tool_ids):
        scored, filled_zero_count = _prepare_scored_frame(
            joined,
            score_col=score_col,
            fdr_threshold=fdr_threshold,
            abs_logfc_threshold=abs_logfc_threshold,
        )

        scatter_png = plots_dir / f"{dataset_id}__{tool_id}_score_vs_logFC.png"
        pr_png = plots_dir / f"{dataset_id}__{tool_id}_pr_curve.png"
        roc_png = plots_dir / f"{dataset_id}__{tool_id}_roc_curve.png"

        pearson, spearman = _plot_scatter_with_correlation(
            scored,
            score_col=score_col,
            dataset_id=dataset_id,
            tool_id=tool_id,
            out_path=scatter_png,
        )
        pr_auc, aps = _plot_pr_curve(
            scored["is_positive"],
            scored[score_col],
            dataset_id=dataset_id,
            tool_id=tool_id,
            out_path=pr_png,
        )
        auroc = _plot_roc_curve(
            scored["is_positive"],
            scored[score_col],
            dataset_id=dataset_id,
            tool_id=tool_id,
            out_path=roc_png,
        )

        report_txt = reports_dir / f"{dataset_id}__{tool_id}_evaluation_report.txt"
        metrics = {
            "rows_used": float(len(scored)),
            "positives": float(scored["is_positive"].sum()),
            "negatives": float(len(scored) - int(scored["is_positive"].sum())),
            "pearson": pearson,
            "spearman": spearman,
            "aps": aps,
            "pr_auc": pr_auc,
            "auroc": auroc,
        }
        _write_tool_report(
            dataset_id=dataset_id,
            mirna=mirna,
            cell_line=cell_line,
            perturbation=perturbation,
            geo_accession=geo_accession,
            de_table_path=de_table_path,
            joined_tsv=joined_tsv,
            tool_id=tool_id,
            canonical_tsv_path=(canonical_paths or {}).get(tool_id),
            metrics=metrics,
            out_path=report_txt,
            scatter_png=scatter_png,
            pr_png=pr_png,
            roc_png=roc_png,
            filled_zero_count=filled_zero_count,
        )

        metric_rows.append(
            {
                "dataset_id": dataset_id,
                "mirna": mirna,
                "cell_line": cell_line,
                "perturbation": perturbation,
                "geo_accession": geo_accession,
                "tool_id": tool_id,
                "aps": aps,
                "spearman": spearman,
                "auroc": auroc,
                "pr_auc": pr_auc,
                "report_txt": str(report_txt),
            }
        )
        dataset_plots[f"{tool_id}_scatter"] = str(scatter_png)
        dataset_plots[f"{tool_id}_pr_curve"] = str(pr_png)
        dataset_plots[f"{tool_id}_roc_curve"] = str(roc_png)

    heatmap_png = plots_dir / f"{dataset_id}__algorithms_vs_genes_heatmap.png"
    _plot_algorithms_vs_genes_heatmap(
        joined,
        score_cols=score_cols,
        tool_ids=tool_ids,
        dataset_id=dataset_id,
        out_path=heatmap_png,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
    )
    dataset_plots["algorithms_vs_genes_heatmap"] = str(heatmap_png)

    predictor_corr_tsv: str | None = None
    if len(score_cols) >= 2:
        predictor_corr_png = plots_dir / f"{dataset_id}__predictor_correlation_heatmap.png"
        predictor_corr_path = reports_dir / f"{dataset_id}__predictor_correlation.tsv"
        corr_matrix = _plot_predictor_correlation_heatmap(
            joined,
            score_cols=score_cols,
            tool_ids=tool_ids,
            dataset_id=dataset_id,
            out_path=predictor_corr_png,
            top_fraction=predictor_top_fraction,
        )
        corr_matrix.to_csv(predictor_corr_path, sep="\t")
        dataset_plots["predictor_correlation_heatmap"] = str(predictor_corr_png)
        predictor_corr_tsv = str(predictor_corr_path)

    return {
        "metric_rows": metric_rows,
        "plots": dataset_plots,
        "predictor_correlation_tsv": predictor_corr_tsv,
        "tool_ids": tool_ids,
        "score_cols": score_cols,
    }


def evaluate_joined_dataset(
    joined_tsv: pathlib.Path,
    *,
    plots_dir: pathlib.Path,
    reports_dir: pathlib.Path,
    fdr_threshold: float,
    abs_logfc_threshold: float,
    predictor_top_fraction: float,
    dataset_id: str | None = None,
    mirna: str | None = None,
    cell_line: str | None = None,
    perturbation: str | None = None,
    geo_accession: str | None = None,
    de_table_path: str | None = None,
    canonical_paths: dict[str, str] | None = None,
    score_cols: list[str] | None = None,
) -> dict[str, Any]:
    df = pd.read_csv(joined_tsv, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    return evaluate_joined_dataframe(
        df,
        plots_dir=plots_dir,
        reports_dir=reports_dir,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        predictor_top_fraction=predictor_top_fraction,
        dataset_id=dataset_id,
        mirna=mirna,
        cell_line=cell_line,
        perturbation=perturbation,
        geo_accession=geo_accession,
        de_table_path=de_table_path,
        joined_tsv=joined_tsv,
        canonical_paths=canonical_paths,
        score_cols=score_cols,
    )


def write_metric_tables(metric_rows: list[dict[str, Any]], tables_dir: pathlib.Path) -> dict[str, str]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metric_rows)
    if metrics_df.empty:
        raise ValueError("No metric rows were produced.")

    id_cols = ["dataset_id", "mirna", "cell_line", "perturbation", "geo_accession"]
    out_paths: dict[str, str] = {}

    for metric_name, filename in [
        ("aps", "aps_per_experiment.tsv"),
        ("spearman", "spearman_per_experiment.tsv"),
        ("auroc", "auroc_per_experiment.tsv"),
    ]:
        wide = metrics_df.pivot_table(
            index=id_cols,
            columns="tool_id",
            values=metric_name,
            aggfunc="first",
        ).reset_index()
        wide.columns.name = None
        out_path = tables_dir / filename
        wide.to_csv(out_path, sep="\t", index=False)
        out_paths[metric_name] = str(out_path)

    return out_paths


def main() -> None:
    args = parse_args()
    joined_tsv = args.joined_tsv.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not joined_tsv.exists():
        raise FileNotFoundError(f"Joined TSV not found: {joined_tsv}")

    df = pd.read_csv(joined_tsv, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]

    if args.logfc_col not in df.columns:
        raise ValueError(f"Column {args.logfc_col!r} not found in joined TSV.")
    if args.fdr_col not in df.columns:
        raise ValueError(f"Column {args.fdr_col!r} not found in joined TSV.")

    if args.logfc_col != "logFC":
        df = df.rename(columns={args.logfc_col: "logFC"})
    if args.fdr_col != "FDR":
        df = df.rename(columns={args.fdr_col: "FDR"})

    score_cols = _detect_score_cols(df, args.score_col)
    results = evaluate_joined_dataframe(
        df,
        plots_dir=out_dir,
        reports_dir=out_dir,
        fdr_threshold=args.fdr_threshold,
        abs_logfc_threshold=args.abs_logfc_threshold,
        predictor_top_fraction=args.predictor_top_fraction,
        dataset_id=args.dataset_id,
        mirna=args.mirna,
        cell_line=args.cell_line,
        perturbation=args.perturbation,
        geo_accession=args.geo_accession,
        joined_tsv=joined_tsv,
        score_cols=score_cols,
    )

    print(f"joined_tsv: {joined_tsv}")
    print(f"dataset_id: {args.dataset_id or 'NA'}")
    print(f"score_cols: {', '.join(results['score_cols'])}")
    if results["predictor_correlation_tsv"] is not None:
        print(f"predictor_correlation_tsv: {results['predictor_correlation_tsv']}")
    for metric_row in results["metric_rows"]:
        print(
            f"tool={metric_row['tool_id']} "
            f"aps={metric_row['aps']:.6f} "
            f"spearman={metric_row['spearman']:.6f} "
            f"auroc={metric_row['auroc']:.6f}"
        )


if __name__ == "__main__":
    main()
