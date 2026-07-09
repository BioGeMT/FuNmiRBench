"""Build diagnostic gene-property distributions for predictor-scored gene sets.

This script is intended as a QC/diagnostic companion to the manuscript asset
builder. It asks whether the genes scored by each predictor, and especially the
intersection gene set (IGS; genes scored by all selected predictors), differ in
3'UTR length or conservation.
"""

from __future__ import annotations

import argparse
import pathlib
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from funmirbench.gene_conservation import load_utr3_conservation
from funmirbench.gene_lengths import load_utr3_lengths

TOOL_IDS = ["targetscan", "mirdb_mirtarget", "microt_cnn", "mirbind2", "miraw"]
TOOL_LABELS = {
    "targetscan": "TargetScan scored",
    "mirdb_mirtarget": "miRDB scored",
    "microt_cnn": "microT-CNN scored",
    "mirbind2": "miRBind2-3UTR scored",
    "miraw": "miRAW scored",
}
BASE_SET_LABELS = OrderedDict(
    [
        ("fgs", "FGS usable"),
        ("igs", "IGS all predictors"),
        ("non_igs", "non-IGS"),
    ]
)


def strip_ensembl_version(value) -> str:
    return str(value).strip().split(".", 1)[0]


def find_repo_root(report_dir: pathlib.Path) -> pathlib.Path:
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


def joined_paths(report_dir: pathlib.Path) -> list[pathlib.Path]:
    paths = sorted(report_dir.glob("**/joined.tsv"))
    if not paths:
        raise FileNotFoundError(f"No joined.tsv files found below {report_dir}")
    return paths


def score_columns(frame: pd.DataFrame) -> list[str]:
    return [f"score_{tool_id}" for tool_id in TOOL_IDS if f"score_{tool_id}" in frame.columns]


def load_dataset_gene_membership(report_dir: pathlib.Path) -> pd.DataFrame:
    """Return one row per usable dataset-gene pair with score-membership flags."""
    rows = []
    for path in joined_paths(report_dir):
        frame = pd.read_csv(path, sep="\t")
        if "gene_id" not in frame.columns:
            continue
        dataset_id = str(frame["dataset_id"].iloc[0]) if "dataset_id" in frame.columns and not frame.empty else path.parent.name
        frame = frame.copy()
        frame["gene_id"] = frame["gene_id"].map(strip_ensembl_version)
        frame["logFC_num"] = pd.to_numeric(frame.get("logFC"), errors="coerce")
        usable = frame.loc[frame["logFC_num"].notna()].dropna(subset=["gene_id"]).copy()
        cols = score_columns(usable)
        if not cols:
            continue
        scored = usable[cols].notna()
        out = pd.DataFrame({"dataset_id": dataset_id, "gene_id": usable["gene_id"].to_numpy()})
        for tool_id in TOOL_IDS:
            col = f"score_{tool_id}"
            out[f"scored_{tool_id}"] = scored[col].to_numpy(bool) if col in scored.columns else False
        scored_cols = [f"scored_{tool_id}" for tool_id in TOOL_IDS]
        out["is_igs"] = out[scored_cols].all(axis=1)
        rows.append(out)
    if not rows:
        raise ValueError(f"No joined rows with predictor scores found below {report_dir}")
    return pd.concat(rows, ignore_index=True).drop_duplicates()


def load_gene_properties(report_dir: pathlib.Path) -> pd.DataFrame:
    repo_root = find_repo_root(report_dir)
    lengths = load_utr3_lengths(root=repo_root)
    conservation = load_utr3_conservation(root=repo_root)
    length_cols = ["gene_id", "utr3_length_bp"]
    conservation_cols = [col for col in ["gene_id", "mean_phyloP", "mean_phastCons"] if col in conservation.columns]
    props = lengths[length_cols].merge(conservation[conservation_cols], on="gene_id", how="outer")
    return props.drop_duplicates("gene_id")


def attach_properties(membership: pd.DataFrame, properties: pd.DataFrame) -> pd.DataFrame:
    return membership.merge(properties, on="gene_id", how="left")


def unique_gene_membership(dataset_gene: pd.DataFrame) -> pd.DataFrame:
    scored_cols = [f"scored_{tool_id}" for tool_id in TOOL_IDS]
    agg = {col: "any" for col in scored_cols}
    agg["is_igs"] = "any"
    unique = dataset_gene.groupby("gene_id", as_index=False).agg(agg)
    unique.insert(0, "dataset_id", "unique_gene")
    return unique


def group_masks(frame: pd.DataFrame) -> OrderedDict[str, pd.Series]:
    masks: OrderedDict[str, pd.Series] = OrderedDict()
    masks["FGS usable"] = pd.Series(True, index=frame.index)
    masks["IGS all predictors"] = frame["is_igs"].astype(bool)
    masks["non-IGS"] = ~frame["is_igs"].astype(bool)
    for tool_id in TOOL_IDS:
        masks[TOOL_LABELS[tool_id]] = frame[f"scored_{tool_id}"].astype(bool)
    return masks


def summarize_distribution(frame: pd.DataFrame, *, mode: str) -> pd.DataFrame:
    rows = []
    for gene_set, mask in group_masks(frame).items():
        sub = frame.loc[mask]
        for prop, label in [("utr3_length_bp", "3UTR length bp"), ("mean_phyloP", "mean phyloP"), ("mean_phastCons", "mean phastCons")]:
            if prop not in sub.columns:
                continue
            values = pd.to_numeric(sub[prop], errors="coerce")
            valid = values.dropna()
            rows.append(
                {
                    "mode": mode,
                    "gene_set": gene_set,
                    "property": label,
                    "n_rows": int(len(sub)),
                    "n_rows_with_property": int(valid.size),
                    "property_match_fraction": float(valid.size / len(sub)) if len(sub) else np.nan,
                    "n_unique_genes": int(sub["gene_id"].nunique()),
                    "n_unique_genes_with_property": int(sub.loc[values.notna(), "gene_id"].nunique()),
                    "mean": float(valid.mean()) if valid.size else np.nan,
                    "median": float(valid.median()) if valid.size else np.nan,
                    "q25": float(valid.quantile(0.25)) if valid.size else np.nan,
                    "q75": float(valid.quantile(0.75)) if valid.size else np.nan,
                    "q90": float(valid.quantile(0.90)) if valid.size else np.nan,
                    "q99": float(valid.quantile(0.99)) if valid.size else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_igs_source_tables(dataset_gene: pd.DataFrame, unique_gene: pd.DataFrame, tables_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    outputs = {}
    for mode, frame in [("dataset_gene", dataset_gene), ("unique_gene", unique_gene)]:
        cols = [col for col in ["dataset_id", "gene_id", "utr3_length_bp", "mean_phyloP", "mean_phastCons"] if col in frame.columns]
        out = frame.loc[frame["is_igs"].astype(bool), cols].copy()
        path = tables_dir / f"igs_gene_property_values_{mode}.tsv"
        out.to_csv(path, sep="\t", index=False)
        outputs[f"igs_gene_property_values_{mode}"] = path
    return outputs


def plot_property_boxplot(
    frame: pd.DataFrame,
    *,
    prop: str,
    ylabel: str,
    title: str,
    out_path: pathlib.Path,
    cap_quantile: float | None = 0.99,
) -> pathlib.Path:
    masks = group_masks(frame)
    labels = list(masks.keys())
    data = []
    for label in labels:
        values = pd.to_numeric(frame.loc[masks[label], prop], errors="coerce").dropna().to_numpy(float)
        data.append(values)
    finite = np.concatenate([values for values in data if values.size]) if any(values.size for values in data) else np.array([])
    if finite.size == 0:
        raise ValueError(f"No values available for {prop}")

    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    box = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch in box["boxes"]:
        patch.set_alpha(0.28)
    for median in box["medians"]:
        median.set_linewidth(1.4)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    if cap_quantile is not None:
        ymin = float(np.nanmin(finite))
        ymax = float(np.nanquantile(finite, cap_quantile))
        if ymax > ymin:
            ax.set_ylim(ymin, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def build_diagnostics(report_dir: pathlib.Path, out_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    membership = load_dataset_gene_membership(report_dir)
    properties = load_gene_properties(report_dir)
    dataset_gene = attach_properties(membership, properties)
    unique_gene = attach_properties(unique_gene_membership(membership), properties)

    outputs = {}
    summary = pd.concat(
        [
            summarize_distribution(dataset_gene, mode="dataset_gene"),
            summarize_distribution(unique_gene, mode="unique_gene"),
        ],
        ignore_index=True,
    )
    summary_path = tables_dir / "gene_property_distribution_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    outputs["gene_property_distribution_summary"] = summary_path
    outputs.update(write_igs_source_tables(dataset_gene, unique_gene, tables_dir))

    for mode, frame in [("dataset_gene", dataset_gene), ("unique_gene", unique_gene)]:
        outputs[f"length_distribution_{mode}"] = plot_property_boxplot(
            frame,
            prop="utr3_length_bp",
            ylabel="3'UTR length (bp)",
            title=f"3'UTR length distribution by predictor-scored gene set ({mode})",
            out_path=figures_dir / f"gene_property_length_distribution_{mode}.png",
            cap_quantile=0.99,
        )
        outputs[f"phylop_distribution_{mode}"] = plot_property_boxplot(
            frame,
            prop="mean_phyloP",
            ylabel="Mean phyloP conservation",
            title=f"Mean 3'UTR phyloP distribution by predictor-scored gene set ({mode})",
            out_path=figures_dir / f"gene_property_phylop_distribution_{mode}.png",
            cap_quantile=0.99,
        )
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Build IGS/per-predictor gene-property diagnostics.")
    parser.add_argument("--report-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("manuscript_assets"))
    args = parser.parse_args()
    outputs = build_diagnostics(args.report_dir, args.out_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
