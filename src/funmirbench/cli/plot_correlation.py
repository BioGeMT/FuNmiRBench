"""
FuNmiRBench combined evaluation plotting helper

Generates from a joined experiment/prediction TSV:
1) score vs logFC scatter
2) score vs -log10(FDR) scatter
3) top-N vs background boxplots:
   - logFC
   - abs(logFC)
   - -log10(FDR)
4) CDF plots:
   - abs(logFC)
   - -log10(FDR)
5) Precision-Recall curve
6) ROC curve
7) GSEA-style enrichment curve
8) a combined text summary report

Positive genes are defined as:
- FDR < fdr_threshold
- abs(logFC) > abs_logfc_threshold
"""

from __future__ import annotations

import argparse
import math
import pathlib
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


DEFAULT_SCORE_CANDIDATES = [
    "score",
    "prediction_score",
    "canonical_score",
    "mock_score",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate combined evaluation plots and a report from a joined experiment/prediction TSV."
    )
    p.add_argument("--joined-tsv", type=pathlib.Path, required=True, help="Joined TSV produced by join_experiment_predictions.")
    p.add_argument("--out-dir", type=pathlib.Path, required=True, help="Directory where plots and report will be written.")
    p.add_argument("--score-col", default=None, help="Prediction score column name. If omitted, try to detect it.")
    p.add_argument("--fdr-col", default="FDR", help="FDR column name (default: FDR).")
    p.add_argument("--logfc-col", default="logFC", help="logFC column name (default: logFC).")
    p.add_argument("--top-n", type=int, default=100, help="Number of top predicted genes to compare against background (default: 100).")
    p.add_argument("--fdr-threshold", type=float, default=0.05, help="Positive-set FDR threshold (default: 0.05).")
    p.add_argument("--abs-logfc-threshold", type=float, default=1.0, help="Positive-set abs(logFC) threshold (default: 1.0).")
    p.add_argument("--dataset-id", default=None, help="Optional dataset ID to include in the report.")
    p.add_argument("--mirna", default=None, help="Optional miRNA label to include in the report.")
    p.add_argument("--cell-line", default=None, help="Optional cell line label to include in the report.")
    p.add_argument("--perturbation", default=None, help="Optional perturbation label to include in the report.")
    p.add_argument("--geo-accession", default=None, help="Optional GEO accession to include in the report.")
    return p.parse_args()


def _detect_score_col(df: pd.DataFrame, requested: Optional[str]) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested score column {requested!r} not found in joined TSV.")
        return requested

    for c in DEFAULT_SCORE_CANDIDATES:
        if c in df.columns:
            return c

    numeric_candidates = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in {"logFC", "FDR", "PValue", "pvalue", "adj.P.Val"}
    ]
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]

    raise ValueError(
        "Could not detect prediction score column automatically. "
        f"Available columns: {list(df.columns)}. Pass --score-col explicitly."
    )


def _safe_neglog10(series: pd.Series) -> pd.Series:
    clipped = series.clip(lower=1e-300)
    return -clipped.map(math.log10)


def _pearson(x: pd.Series, y: pd.Series) -> float:
    return float(x.corr(y, method="pearson"))


def _spearman(x: pd.Series, y: pd.Series) -> float:
    return float(x.corr(y, method="spearman"))


def _base_name(joined_tsv: pathlib.Path) -> str:
    return joined_tsv.stem


def _scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: pathlib.Path,
    title: str,
    x_label: str,
    y_label: str,
) -> Tuple[float, float]:
    x = df[x_col]
    y = df[y_col]

    pearson = _pearson(x, y)
    spearman = _spearman(x, y)

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.5, s=12)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return pearson, spearman


def _make_groups(df: pd.DataFrame, score_col: str, top_n: int) -> pd.DataFrame:
    ranked = df.sort_values(score_col, ascending=False).copy()
    ranked["group"] = "background"
    ranked.iloc[:top_n, ranked.columns.get_loc("group")] = f"top_{top_n}"
    return ranked


def _boxplot(
    df: pd.DataFrame,
    value_col: str,
    out_path: pathlib.Path,
    title: str,
    y_label: str,
    top_label: str,
) -> None:
    top_values = df.loc[df["group"] == top_label, value_col].dropna().tolist()
    bg_values = df.loc[df["group"] == "background", value_col].dropna().tolist()

    if not top_values or not bg_values:
        raise ValueError(f"Cannot plot {value_col}: one of the groups is empty.")

    plt.figure(figsize=(6, 5))
    plt.boxplot([top_values, bg_values], tick_labels=[top_label, "background"])
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _ecdf(values):
    s = pd.Series(values).dropna().sort_values().to_numpy()
    y = [i / len(s) for i in range(1, len(s) + 1)]
    return s, y


def _cdf_plot(
    df: pd.DataFrame,
    value_col: str,
    out_path: pathlib.Path,
    title: str,
    x_label: str,
    top_label: str,
) -> None:
    top_values = df.loc[df["group"] == top_label, value_col].dropna()
    bg_values = df.loc[df["group"] == "background", value_col].dropna()

    if top_values.empty or bg_values.empty:
        raise ValueError(f"Cannot plot CDF for {value_col}: one of the groups is empty.")

    top_x, top_y = _ecdf(top_values)
    bg_x, bg_y = _ecdf(bg_values)

    plt.figure(figsize=(6, 5))
    plt.plot(top_x, top_y, label=top_label)
    plt.plot(bg_x, bg_y, label="background")
    plt.xlabel(x_label)
    plt.ylabel("CDF")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_pr_curve(y_true, y_score, out_path: pathlib.Path) -> Tuple[float, float]:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    baseline = float(sum(y_true)) / float(len(y_true))

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AUPRC = {pr_auc:.4f}")
    plt.axhline(baseline, linestyle="--", label=f"Baseline = {baseline:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return pr_auc, baseline


def _plot_roc_curve(y_true, y_score, out_path: pathlib.Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUROC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return roc_auc


def _plot_enrichment_curve(df: pd.DataFrame, score_col: str, out_path: pathlib.Path) -> float:
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    hits = ranked["is_positive"].astype(int)
    n = len(ranked)
    nh = int(hits.sum())
    nm = n - nh

    if nh == 0 or nm == 0:
        raise ValueError("Enrichment curve requires both positive and negative genes.")

    running = []
    rs = 0.0
    hit_step = 1.0 / nh
    miss_step = 1.0 / nm

    for hit in hits:
        rs += hit_step if hit == 1 else -miss_step
        running.append(rs)

    enrichment_score = max(running, key=abs)

    x = list(range(1, n + 1))
    hit_positions = [i + 1 for i, hit in enumerate(hits) if hit == 1]

    plt.figure(figsize=(8, 5))
    plt.plot(x, running, label=f"ES = {enrichment_score:.4f}")
    plt.axhline(0, linestyle="--")
    if hit_positions:
        ymin = min(running)
        ymax = max(running)
        tick_bottom = ymin - (ymax - ymin) * 0.08 if ymax > ymin else ymin - 0.1
        tick_top = ymin
        plt.vlines(hit_positions, tick_bottom, tick_top, linewidth=0.5)
        plt.ylim(bottom=tick_bottom)
    plt.xlabel("Genes ranked by prediction score (high → low)")
    plt.ylabel("Running enrichment score")
    plt.title("GSEA-style enrichment curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return float(enrichment_score)


def main() -> None:
    args = parse_args()

    joined_tsv = args.joined_tsv.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not joined_tsv.exists():
        raise FileNotFoundError(f"Joined TSV not found: {joined_tsv}")

    df = pd.read_csv(joined_tsv, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]

    if args.logfc_col not in df.columns:
        raise ValueError(f"Column {args.logfc_col!r} not found in joined TSV.")
    if args.fdr_col not in df.columns:
        raise ValueError(f"Column {args.fdr_col!r} not found in joined TSV.")

    score_col = _detect_score_col(df, args.score_col)

    keep = df[[score_col, args.logfc_col, args.fdr_col]].copy()
    keep = keep.dropna()
    keep = keep[keep[args.fdr_col] > 0]

    if keep.empty:
        raise ValueError("No usable rows remain after dropping NA values and non-positive FDR values.")

    if len(keep) <= args.top_n:
        raise ValueError(
            f"top_n={args.top_n} is too large for the available rows ({len(keep)})."
        )

    keep["abs_logFC"] = keep[args.logfc_col].abs()
    keep["neglog10_FDR"] = _safe_neglog10(keep[args.fdr_col])
    keep["is_positive"] = (
        (keep[args.fdr_col] < args.fdr_threshold) &
        (keep["abs_logFC"] > args.abs_logfc_threshold)
    ).astype(int)

    positives = int(keep["is_positive"].sum())
    negatives = int(len(keep) - positives)

    if positives == 0:
        raise ValueError(
            "No positive genes found under the current thresholds. "
            "Try a less strict FDR or abs(logFC) threshold."
        )
    if negatives == 0:
        raise ValueError(
            "All genes are positive under the current thresholds. "
            "Try a stricter FDR or abs(logFC) threshold."
        )

    base = _base_name(joined_tsv)

    score_vs_logfc_png = out_dir / f"{base}_score_vs_logFC.png"
    score_vs_fdr_png = out_dir / f"{base}_score_vs_neglog10FDR.png"

    pearson_logfc, spearman_logfc = _scatter_plot(
        keep,
        x_col=score_col,
        y_col=args.logfc_col,
        out_path=score_vs_logfc_png,
        title="Prediction score vs logFC",
        x_label=score_col,
        y_label=args.logfc_col,
    )

    pearson_fdr, spearman_fdr = _scatter_plot(
        keep,
        x_col=score_col,
        y_col="neglog10_FDR",
        out_path=score_vs_fdr_png,
        title="Prediction score vs -log10(FDR)",
        x_label=score_col,
        y_label="-log10(FDR)",
    )

    grouped = _make_groups(keep, score_col, args.top_n)
    top_label = f"top_{args.top_n}"

    logfc_boxplot_png = out_dir / f"{base}_top{args.top_n}_logFC_boxplot.png"
    abslogfc_boxplot_png = out_dir / f"{base}_top{args.top_n}_abslogFC_boxplot.png"
    neglog10fdr_boxplot_png = out_dir / f"{base}_top{args.top_n}_neglog10FDR_boxplot.png"

    _boxplot(grouped, args.logfc_col, logfc_boxplot_png,
             f"Top {args.top_n} prediction scores vs background: logFC", "logFC", top_label)
    _boxplot(grouped, "abs_logFC", abslogfc_boxplot_png,
             f"Top {args.top_n} prediction scores vs background: abs(logFC)", "abs(logFC)", top_label)
    _boxplot(grouped, "neglog10_FDR", neglog10fdr_boxplot_png,
             f"Top {args.top_n} prediction scores vs background: -log10(FDR)", "-log10(FDR)", top_label)

    cdf_abslogfc_png = out_dir / f"{base}_top{args.top_n}_abslogFC_cdf.png"
    cdf_neglog10fdr_png = out_dir / f"{base}_top{args.top_n}_neglog10FDR_cdf.png"

    _cdf_plot(grouped, "abs_logFC", cdf_abslogfc_png,
              f"CDF of abs(logFC): top {args.top_n} vs background", "abs(logFC)", top_label)
    _cdf_plot(grouped, "neglog10_FDR", cdf_neglog10fdr_png,
              f"CDF of -log10(FDR): top {args.top_n} vs background", "-log10(FDR)", top_label)

    top_df = grouped[grouped["group"] == top_label]
    bg_df = grouped[grouped["group"] == "background"]

    pr_png = out_dir / f"{base}_pr_curve.png"
    roc_png = out_dir / f"{base}_roc_curve.png"
    enrichment_png = out_dir / f"{base}_enrichment_curve.png"

    y_true = keep["is_positive"].astype(int).tolist()
    y_score = keep[score_col].astype(float).tolist()

    pr_auc, pr_baseline = _plot_pr_curve(y_true, y_score, pr_png)
    roc_auc = _plot_roc_curve(y_true, y_score, roc_png)
    enrichment_score = _plot_enrichment_curve(keep, score_col, enrichment_png)

    report_txt = out_dir / f"{base}_evaluation_report.txt"
    report_lines = [
        f"joined_tsv: {joined_tsv}",
        f"rows_used: {len(keep)}",
        f"score_col: {score_col}",
        f"logfc_col: {args.logfc_col}",
        f"fdr_col: {args.fdr_col}",
        f"top_n: {args.top_n}",
        f"fdr_threshold: {args.fdr_threshold}",
        f"abs_logfc_threshold: {args.abs_logfc_threshold}",
        "",
        "dataset_metadata:",
        f"  dataset_id: {args.dataset_id or 'NA'}",
        f"  mirna: {args.mirna or 'NA'}",
        f"  cell_line: {args.cell_line or 'NA'}",
        f"  perturbation: {args.perturbation or 'NA'}",
        f"  geo_accession: {args.geo_accession or 'NA'}",
        "",
        "class_balance:",
        f"  positives: {positives}",
        f"  negatives: {negatives}",
        f"  positive_fraction: {positives / len(keep):.6f}",
        f"  significant_genes_fdr_lt_{args.fdr_threshold}: {int((keep[args.fdr_col] < args.fdr_threshold).sum())}",
        "",
        "score_vs_logFC:",
        f"  pearson: {pearson_logfc:.6f}",
        f"  spearman: {spearman_logfc:.6f}",
        f"  png: {score_vs_logfc_png}",
        "",
        "score_vs_neglog10FDR:",
        f"  pearson: {pearson_fdr:.6f}",
        f"  spearman: {spearman_fdr:.6f}",
        f"  png: {score_vs_fdr_png}",
        "",
        "topn_boxplots_medians:",
        f"  {top_label}_logFC: {top_df[args.logfc_col].median():.6f}",
        f"  background_logFC: {bg_df[args.logfc_col].median():.6f}",
        f"  {top_label}_abslogFC: {top_df['abs_logFC'].median():.6f}",
        f"  background_abslogFC: {bg_df['abs_logFC'].median():.6f}",
        f"  {top_label}_neglog10FDR: {top_df['neglog10_FDR'].median():.6f}",
        f"  background_neglog10FDR: {bg_df['neglog10_FDR'].median():.6f}",
        f"  logFC_boxplot: {logfc_boxplot_png}",
        f"  abslogFC_boxplot: {abslogfc_boxplot_png}",
        f"  neglog10FDR_boxplot: {neglog10fdr_boxplot_png}",
        "",
        "cdf_plots:",
        f"  abslogFC_cdf: {cdf_abslogfc_png}",
        f"  neglog10FDR_cdf: {cdf_neglog10fdr_png}",
        "",
        "ranking_metrics:",
        f"  auprc: {pr_auc:.6f}",
        f"  pr_baseline: {pr_baseline:.6f}",
        f"  auroc: {roc_auc:.6f}",
        f"  enrichment_score: {enrichment_score:.6f}",
        f"  pr_curve: {pr_png}",
        f"  roc_curve: {roc_png}",
        f"  enrichment_curve: {enrichment_png}",
    ]
    report_txt.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"joined_tsv: {joined_tsv}")
    print(f"rows_used: {len(keep)}")
    print(f"score_col: {score_col}")
    print(f"dataset_id: {args.dataset_id or 'NA'}")
    print(f"mirna: {args.mirna or 'NA'}")
    print(f"cell_line: {args.cell_line or 'NA'}")
    print(f"positives: {positives}")
    print(f"negatives: {negatives}")
    print(f"enrichment_score: {enrichment_score:.6f}")
    print(f"report: {report_txt}")


if __name__ == "__main__":
    main()
