"""
Plot correlation between prediction score and DE signal for one dataset/tool join.

Input: TSV produced by funmirbench.cli.join_experiment_predictions
Required columns:
- score
- logFC (or another DE metric via --y-col)

Outputs:
- PNG scatter plot
- TXT summary with Pearson/Spearman correlations

Note: This is intentionally minimal and not the full evaluation suite.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Optional

DEFAULT_ROOT = pathlib.Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make a correlation scatter plot from a joined TSV.")
    p.add_argument("--joined-tsv", type=pathlib.Path, required=True, help="Joined TSV path.")
    p.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    p.add_argument("--y-col", default="logFC", help="DE column to correlate against (default: logFC).")
    p.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("data/plots"), help="Output directory.")
    p.add_argument("--title", default=None, help="Optional plot title.")
    p.add_argument("--max-points", type=int, default=20000, help="Cap points for plotting (default: 20000).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    joined = args.joined_tsv
    if not joined.is_absolute():
        joined = root / joined
    if not joined.exists():
        raise FileNotFoundError(f"Joined TSV not found: {joined}")

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError("plot_correlation requires pandas.") from exc

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise ImportError("plot_correlation requires matplotlib.") from exc

    # Spearman without scipy: use ranks + Pearson
    import numpy as np  # type: ignore

    df = pd.read_csv(joined, sep="\t")
    if "score" not in df.columns:
        raise ValueError(f"{joined} missing required column 'score'")
    if args.y_col not in df.columns:
        raise ValueError(f"{joined} missing required column {args.y_col!r}")

    df = df.dropna(subset=["score", args.y_col]).copy()
    df["score"] = df["score"].astype(float)
    df[args.y_col] = df[args.y_col].astype(float)

    if len(df) == 0:
        raise ValueError("No rows after dropping NaNs; cannot plot.")

    # Subsample for plotting if huge (deterministic: head)
    if args.max_points and len(df) > args.max_points:
        df_plot = df.head(args.max_points)
    else:
        df_plot = df

    x = df["score"].to_numpy()
    y = df[args.y_col].to_numpy()

    # Pearson
    pearson = float(np.corrcoef(x, y)[0, 1])

    # Spearman: rank then Pearson
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(xr, yr)[0, 1])

    dataset_id = str(df["dataset_id"].iloc[0]) if "dataset_id" in df.columns else "unknown"
    tool = "unknown"
    if "mirna" in df.columns:
        mirna = str(df["mirna"].iloc[0])
    else:
        mirna = "unknown"
    # best-effort parse tool from filename
    stem = joined.stem
    if "_" in stem:
        tool = stem.split("_")[-1]

    title = args.title or f"{dataset_id} | {mirna} | {tool} : score vs {args.y_col}"

    # Plot
    plt.figure()
    plt.scatter(df_plot["score"], df_plot[args.y_col], s=6, alpha=0.4)
    plt.xlabel("prediction score")
    plt.ylabel(args.y_col)
    plt.title(title)
    plt.tight_layout()

    png_path = out_dir / f"{dataset_id}_{tool}_score_vs_{args.y_col}.png"
    plt.savefig(png_path, dpi=150)
    plt.close()

    summary_path = out_dir / f"{dataset_id}_{tool}_score_vs_{args.y_col}.txt"
    summary = (
        f"joined_tsv: {joined}\n"
        f"rows: {len(df)}\n"
        f"y_col: {args.y_col}\n"
        f"pearson: {pearson:.6f}\n"
        f"spearman: {spearman:.6f}\n"
        f"png: {png_path}\n"
    )
    summary_path.write_text(summary, encoding="utf-8")

    print(summary)


if __name__ == "__main__":
    main()
