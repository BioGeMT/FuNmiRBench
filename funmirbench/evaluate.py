"""Evaluate joined GT/prediction tables: metrics, plots, reports."""

from funmirbench.evaluate_common import *
from funmirbench.evaluate_plots import *
from funmirbench.evaluate_reports import *


def evaluate_joined_dataframe(
    joined, *, plots_dir, reports_dir,
    fdr_threshold, abs_logfc_threshold, predictor_top_fraction,
    dataset_id=None, mirna=None, cell_line=None,
    perturbation=None, geo_accession=None,
    de_table_path=None, joined_tsv=None,
    predictor_output_paths=None,
    tool_labels=None,
    write_top_prediction_cdfs=False,
    logger=None,
):
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    _set_tool_labels(tool_labels)

    score_cols = sorted(c for c in joined.columns if c.startswith(SCORE_PREFIX))
    if not score_cols:
        raise ValueError("No score_<tool_id> columns found in joined dataframe.")

    dataset_id = dataset_id or (
        str(joined["dataset_id"].iloc[0]) if "dataset_id" in joined.columns else "NA"
    )
    mirna = mirna or (
        str(joined["mirna"].iloc[0]) if "mirna" in joined.columns else None
    )
    dataset_plots_dir = plots_dir
    predictor_plots_dir = dataset_plots_dir / "predictors"
    comparison_plots_dir = dataset_plots_dir / "comparisons"
    heatmap_plots_dir = dataset_plots_dir / "heatmaps"
    for path in (dataset_plots_dir, predictor_plots_dir, comparison_plots_dir, heatmap_plots_dir):
        path.mkdir(parents=True, exist_ok=True)
    tool_ids = [_tool_id_from_score_col(sc) for sc in score_cols]
    _set_tool_colors(tool_ids)
    global_rank_cols = []
    local_rank_cols = []
    for score_col, tool_id in zip(score_cols, tool_ids):
        global_rank_col = _rank_col_for_tool(tool_id)
        local_rank_col = _rank_col_for_tool(tool_id, prefix=LOCAL_RANK_PREFIX)
        if local_rank_col not in joined.columns:
            joined[local_rank_col] = _rank_scale_scores(joined[score_col])
        if global_rank_col not in joined.columns:
            joined[global_rank_col] = joined[local_rank_col]
        global_rank_cols.append(global_rank_col)
        local_rank_cols.append(local_rank_col)

    metric_rows = []
    dataset_plots = {}
    predictor_correlation_tsv = None
    comparisons = []
    coverage_by_tool = {}
    evaluated_score_cols = []
    evaluated_tool_ids = []
    evaluated_local_rank_cols = []

    _emit_log(logger, f"    Evaluation start: {dataset_id} | tools={tool_ids}")

    for score_col, tool_id, local_rank_col in zip(score_cols, tool_ids, local_rank_cols):
        _emit_log(logger, f"    Tool: {tool_id} | preparing scored pairs")
        tool_plots_dir = predictor_plots_dir / tool_id
        tool_plots_dir.mkdir(parents=True, exist_ok=True)
        try:
            scored, coverage_info = _prepare_scored_frame(
                joined, score_col=score_col,
                fdr_threshold=fdr_threshold, abs_logfc_threshold=abs_logfc_threshold,
                perturbation=perturbation,
            )
        except ValueError as exc:
            message = str(exc)
            valid_rows = joined["logFC"].notna() & joined["FDR"].notna()
            if "FDR" in joined.columns:
                valid_rows = valid_rows & (joined["FDR"].astype(float) > 0)
            scored_rows = int(joined.loc[valid_rows, score_col].notna().sum())
            total_valid_rows = int(valid_rows.sum())
            sparse_scoring = scored_rows < total_valid_rows
            skip_sparse_reason = (
                message.startswith("No scored rows remain")
                or message.startswith("No positives remain")
                or message.startswith("No negatives remain")
            )
            if not (skip_sparse_reason and sparse_scoring):
                raise
            _emit_log(logger, f"    Tool: {tool_id} | skipped: {exc}")
            continue
        scatter_png = tool_plots_dir / "score_vs_expected_effect.png"
        gsea_png = tool_plots_dir / "gsea_enrichment.png"
        pr_curve_png = tool_plots_dir / "precision_recall_curve.png"
        roc_curve_png = tool_plots_dir / "roc_curve.png"
        pearson, spearman = _plot_scatter_with_correlation(
            scored,
            score_col=score_col,
            dataset_id=dataset_id,
            tool_id=tool_id,
            positives_total=coverage_info["positives_total"],
            out_path=scatter_png,
        )
        enrichment_score = _plot_gsea_enrichment(
            scored,
            score_col=score_col,
            dataset_id=dataset_id,
            tool_id=tool_id,
            positives_total=coverage_info["positives_total"],
            out_path=gsea_png,
        )
        pr_auc, aps = _compute_pr_metrics(scored["is_positive"], scored[score_col])
        auroc = _compute_auroc(scored["is_positive"], scored[score_col])
        _plot_single_predictor_pr_curve(
            {
                "tool_id": tool_id,
                "y_true": scored["is_positive"],
                "y_score": scored[score_col],
                "coverage": coverage_info["coverage"],
                "positives_total": coverage_info["positives_total"],
            },
            dataset_id=dataset_id,
            out_path=pr_curve_png,
        )
        _plot_single_predictor_roc_curve(
            {
                "tool_id": tool_id,
                "y_true": scored["is_positive"],
                "y_score": scored[score_col],
                "coverage": coverage_info["coverage"],
                "positives_total": coverage_info["positives_total"],
            },
            dataset_id=dataset_id,
            out_path=roc_curve_png,
        )

        report_md = reports_dir / f"{dataset_id}__{tool_id}_evaluation_report.md"
        report_pdf = reports_dir / f"{dataset_id}__{tool_id}_evaluation_report.pdf"
        metrics = {
            "rows_total": float(coverage_info["rows_total"]),
            "rows_used": float(len(scored)),
            "positives": float(scored["is_positive"].sum()),
            "negatives": float(len(scored) - int(scored["is_positive"].sum())),
            "rows_missing_score": float(coverage_info["rows_missing_score"]),
            "coverage": float(coverage_info["coverage"]),
            "positives_total": float(coverage_info["positives_total"]),
            "positives_scored": float(coverage_info["positives_scored"]),
            "positive_coverage": float(coverage_info["positive_coverage"]),
            "pearson": pearson, "spearman": spearman,
            "aps": aps, "pr_auc": pr_auc, "auroc": auroc,
        }
        _write_tool_report(
            dataset_id=dataset_id, mirna=mirna, cell_line=cell_line,
            perturbation=perturbation, geo_accession=geo_accession,
            de_table_path=de_table_path, joined_tsv=joined_tsv,
            tool_id=tool_id,
            predictor_output_path=(predictor_output_paths or {}).get(tool_id),
            metrics=metrics, markdown_path=report_md, pdf_path=report_pdf,
            coverage_info=coverage_info,
            scatter_png=scatter_png,
            pr_curve_png=pr_curve_png,
            roc_curve_png=roc_curve_png,
            gsea_png=gsea_png,
            fdr_threshold=fdr_threshold,
            abs_logfc_threshold=abs_logfc_threshold,
        )
        _emit_log(
            logger,
            (
                f"    Tool: {tool_id} | coverage={coverage_info['coverage']:.1%} "
                f"| positive_cov={coverage_info['positive_coverage']:.1%} "
                f"| rows={coverage_info['rows_scored']}/{coverage_info['rows_total']} "
                f"| APS={aps:.3f} | AUROC={auroc:.3f} | ES={enrichment_score:.3f}"
            ),
        )
        _emit_log(logger, f"    Tool: {tool_id} | wrote scatter/report")

        metric_rows.append({
            "dataset_id": dataset_id, "mirna": mirna, "cell_line": cell_line,
            "perturbation": perturbation, "geo_accession": geo_accession,
            "tool_id": tool_id,
            "rows_total": coverage_info["rows_total"],
            "rows_scored": coverage_info["rows_scored"],
            "rows_missing_score": coverage_info["rows_missing_score"],
            "coverage": coverage_info["coverage"],
            "positive_coverage": coverage_info["positive_coverage"],
            "aps": aps, "spearman": spearman, "auroc": auroc, "pr_auc": pr_auc,
        })
        comparisons.append({
            "tool_id": tool_id,
            "y_true": scored["is_positive"],
            "y_score": scored[score_col],
            "coverage": coverage_info["coverage"],
        })
        coverage_by_tool[tool_id] = coverage_info["coverage"]
        evaluated_score_cols.append(score_col)
        evaluated_tool_ids.append(tool_id)
        evaluated_local_rank_cols.append(local_rank_col)
        dataset_plots[f"{tool_id}_scatter"] = str(scatter_png)
        dataset_plots[f"{tool_id}_gsea_enrichment"] = str(gsea_png)
        dataset_plots[f"{tool_id}_pr_curve"] = str(pr_curve_png)
        dataset_plots[f"{tool_id}_roc_curve"] = str(roc_curve_png)

    if not evaluated_score_cols:
        raise ValueError(f"No predictors had scored rows for {dataset_id}.")

    heatmap_png = heatmap_plots_dir / "algorithms_vs_genes.png"
    _plot_algorithms_vs_genes_heatmap(
        joined,
        score_cols=evaluated_score_cols,
        rank_cols=evaluated_local_rank_cols,
        tool_ids=evaluated_tool_ids,
        dataset_id=dataset_id, out_path=heatmap_png,
        fdr_threshold=fdr_threshold, abs_logfc_threshold=abs_logfc_threshold,
        perturbation=perturbation,
    )
    dataset_plots["algorithms_vs_genes_heatmap"] = str(heatmap_png)
    _emit_log(logger, f"    Dataset: {dataset_id} | wrote gene-level heatmap")

    top_positive_heatmap_png = heatmap_plots_dir / "top_10pct_positive_genes.png"
    wrote_top_positive_heatmap = _plot_top_positive_heatmap(
        joined,
        rank_cols=evaluated_local_rank_cols,
        tool_ids=evaluated_tool_ids,
        dataset_id=dataset_id,
        out_path=top_positive_heatmap_png,
        fdr_threshold=fdr_threshold,
        abs_logfc_threshold=abs_logfc_threshold,
        positive_fraction=0.10,
        perturbation=perturbation,
    )
    if wrote_top_positive_heatmap:
        dataset_plots["top_10pct_positive_heatmap"] = str(top_positive_heatmap_png)
        _emit_log(logger, f"    Dataset: {dataset_id} | wrote top-positive heatmap")

    if len(evaluated_score_cols) >= 2:
        comparison_pr_png = comparison_plots_dir / "precision_recall_common.png"
        comparison_pr_all_png = comparison_plots_dir / "precision_recall_all_scored.png"
        comparison_roc_png = comparison_plots_dir / "roc_common.png"
        comparison_roc_all_png = comparison_plots_dir / "roc_all_scored.png"
        comparison_gsea_png = comparison_plots_dir / "gsea_common.png"
        comparison_cdf_png = comparison_plots_dir / f"top_{TOP_PREDICTION_CDF_N}_effect_cdfs.png"
        try:
            common_pr = _prepare_common_scored_frame(
                joined,
                score_cols=evaluated_score_cols,
                fdr_threshold=fdr_threshold,
                abs_logfc_threshold=abs_logfc_threshold,
                perturbation=perturbation,
            )
        except ValueError as exc:
            common_comparisons = []
            _emit_log(logger, f"    Dataset: {dataset_id} | skipped common comparison plots: {exc}")
        else:
            common_comparisons = [
                {
                    "tool_id": tool_id,
                    "gene_id": common_pr["gene_id"],
                    "y_true": common_pr["is_positive"],
                    "y_score": common_pr[score_col],
                    "coverage": coverage_by_tool.get(tool_id, float("nan")),
                }
                for score_col, tool_id in zip(evaluated_score_cols, evaluated_tool_ids)
            ]
            _plot_predictor_pr_curves(
                common_comparisons,
                dataset_id=dataset_id,
                out_path=comparison_pr_png,
            )
        _plot_predictor_pr_curves_own_scored(
            comparisons,
            dataset_id=dataset_id,
            out_path=comparison_pr_all_png,
        )
        if common_comparisons:
            _plot_predictor_roc_curves(
                common_comparisons,
                dataset_id=dataset_id,
                out_path=comparison_roc_png,
            )
        _plot_predictor_roc_curves_own_scored(
            comparisons,
            dataset_id=dataset_id,
            out_path=comparison_roc_all_png,
        )
        if common_comparisons:
            _plot_predictor_gsea_curves(
                common_comparisons,
                dataset_id=dataset_id,
                out_path=comparison_gsea_png,
            )
        if write_top_prediction_cdfs:
            _plot_top_prediction_effect_cdfs(
                joined,
                score_cols=evaluated_score_cols,
                tool_ids=evaluated_tool_ids,
                dataset_id=dataset_id,
                out_path=comparison_cdf_png,
                perturbation=perturbation,
            )
        if common_comparisons:
            dataset_plots["predictor_pr_curves"] = str(comparison_pr_png)
            dataset_plots["predictor_roc_curves"] = str(comparison_roc_png)
            dataset_plots["predictor_gsea_curves"] = str(comparison_gsea_png)
        dataset_plots["predictor_pr_curves_all_scored"] = str(comparison_pr_all_png)
        dataset_plots["predictor_roc_curves_all_scored"] = str(comparison_roc_all_png)
        if write_top_prediction_cdfs:
            dataset_plots["predictor_top100_effect_cdfs"] = str(comparison_cdf_png)
        _emit_log(
            logger,
            f"    Dataset: {dataset_id} | wrote PR/ROC/GSEA comparison plots",
        )

        corr_tsv = reports_dir / f"{dataset_id}__predictor_correlation.tsv"
        corr_matrix = _build_predictor_correlation_matrix(
            joined,
            rank_cols=evaluated_local_rank_cols,
            tool_ids=evaluated_tool_ids,
            top_fraction=predictor_top_fraction,
        )
        corr_matrix.to_csv(corr_tsv, sep="\t")
        predictor_correlation_tsv = str(corr_tsv)
        _emit_log(logger, f"    Dataset: {dataset_id} | wrote predictor correlation table")

    _emit_log(logger, f"    Evaluation complete: {dataset_id}")

    return {
        "metric_rows": metric_rows,
        "plots": dataset_plots,
        "predictor_correlation_tsv": predictor_correlation_tsv,
        "tool_ids": tool_ids,
        "score_cols": score_cols,
    }
