"""Count unique-gene predictor overlaps for Venn/UpSet-style diagnostics."""

from __future__ import annotations

import argparse
import pathlib
from itertools import combinations

import pandas as pd

TOOL_IDS = ["targetscan", "mirdb_mirtarget", "microt_cnn", "mirbind2", "miraw"]
TOOL_LABELS = {
    "targetscan": "TargetScan",
    "mirdb_mirtarget": "miRDB",
    "microt_cnn": "microT-CNN",
    "mirbind2": "miRBind2-3UTR",
    "miraw": "miRAW",
}


def strip_ensembl_version(value) -> str:
    return str(value).strip().split(".", 1)[0]


def joined_paths(report_dir: pathlib.Path) -> list[pathlib.Path]:
    paths = sorted(report_dir.glob("**/joined.tsv"))
    if not paths:
        raise FileNotFoundError(f"No joined.tsv files found below {report_dir}")
    return paths


def score_columns(frame: pd.DataFrame) -> list[str]:
    return [f"score_{tool_id}" for tool_id in TOOL_IDS if f"score_{tool_id}" in frame.columns]


def load_unique_gene_score_flags(report_dir: pathlib.Path) -> pd.DataFrame:
    """Return one row per unique gene with one scored flag per predictor.

    A gene is considered scored by a predictor if that predictor has a non-missing
    score for the gene in at least one usable dataset-gene row.
    """
    rows = []
    for path in joined_paths(report_dir):
        frame = pd.read_csv(path, sep="\t")
        if "gene_id" not in frame.columns:
            continue
        frame = frame.copy()
        frame["gene_id"] = frame["gene_id"].map(strip_ensembl_version)
        frame["logFC_num"] = pd.to_numeric(frame.get("logFC"), errors="coerce")
        usable = frame.loc[frame["logFC_num"].notna()].dropna(subset=["gene_id"]).copy()
        cols = score_columns(usable)
        if not cols:
            continue
        scored = usable[cols].notna()
        out = pd.DataFrame({"gene_id": usable["gene_id"].to_numpy()})
        for tool_id in TOOL_IDS:
            col = f"score_{tool_id}"
            out[f"scored_{tool_id}"] = scored[col].to_numpy(bool) if col in scored.columns else False
        rows.append(out)
    if not rows:
        raise ValueError(f"No usable joined rows with predictor scores found below {report_dir}")
    stacked = pd.concat(rows, ignore_index=True).drop_duplicates()
    agg = {f"scored_{tool_id}": "any" for tool_id in TOOL_IDS}
    return stacked.groupby("gene_id", as_index=False).agg(agg)


def all_combination_counts(unique_flags: pd.DataFrame) -> pd.DataFrame:
    """Return inclusive intersections and exact Venn-region counts.

    intersection_gene_count: genes scored by all predictors in the combination,
    even if additional predictors also scored them.

    exact_region_gene_count: genes scored by exactly the predictors in the
    combination and by no other selected predictors. These are the numbers that
    go into a Venn/UpSet region.
    """
    rows = []
    flag = {tool_id: f"scored_{tool_id}" for tool_id in TOOL_IDS}
    all_flag_cols = [flag[tool_id] for tool_id in TOOL_IDS]
    total_unique = int(unique_flags["gene_id"].nunique())
    scored_by_any = unique_flags[all_flag_cols].any(axis=1)
    rows.append(
        {
            "combination_size": 0,
            "predictors": "none",
            "predictor_labels": "none",
            "intersection_gene_count": 0,
            "exact_region_gene_count": int((~scored_by_any).sum()),
            "total_unique_genes": total_unique,
        }
    )
    for size in range(1, len(TOOL_IDS) + 1):
        for combo in combinations(TOOL_IDS, size):
            combo_flags = [flag[tool_id] for tool_id in combo]
            other_flags = [flag[tool_id] for tool_id in TOOL_IDS if tool_id not in combo]
            intersection_mask = unique_flags[combo_flags].all(axis=1)
            exact_mask = intersection_mask.copy()
            if other_flags:
                exact_mask &= ~unique_flags[other_flags].any(axis=1)
            rows.append(
                {
                    "combination_size": size,
                    "predictors": "+".join(combo),
                    "predictor_labels": " + ".join(TOOL_LABELS[tool_id] for tool_id in combo),
                    "intersection_gene_count": int(intersection_mask.sum()),
                    "exact_region_gene_count": int(exact_mask.sum()),
                    "total_unique_genes": total_unique,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["combination_size", "exact_region_gene_count", "intersection_gene_count"],
        ascending=[True, False, False],
    )


def pairwise_matrix(unique_flags: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row_tool in TOOL_IDS:
        row_mask = unique_flags[f"scored_{row_tool}"].astype(bool)
        row = {"predictor": row_tool, "predictor_label": TOOL_LABELS[row_tool]}
        for col_tool in TOOL_IDS:
            col_mask = unique_flags[f"scored_{col_tool}"].astype(bool)
            row[col_tool] = int((row_mask & col_mask).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count unique-gene predictor overlaps.")
    parser.add_argument("--report-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("manuscript_assets"))
    args = parser.parse_args()

    tables_dir = args.out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    unique_flags = load_unique_gene_score_flags(args.report_dir)
    flags_path = tables_dir / "unique_gene_predictor_score_flags.tsv"
    unique_flags.to_csv(flags_path, sep="\t", index=False)

    combinations_path = tables_dir / "unique_gene_predictor_intersections.tsv"
    all_combination_counts(unique_flags).to_csv(combinations_path, sep="\t", index=False)

    pairwise_path = tables_dir / "unique_gene_predictor_pairwise_intersections.tsv"
    pairwise_matrix(unique_flags).to_csv(pairwise_path, sep="\t", index=False)

    print(f"unique_gene_predictor_score_flags: {flags_path}")
    print(f"unique_gene_predictor_intersections: {combinations_path}")
    print(f"unique_gene_predictor_pairwise_intersections: {pairwise_path}")


if __name__ == "__main__":
    main()
