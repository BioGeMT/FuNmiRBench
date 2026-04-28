from pathlib import Path

import pandas as pd


script_dir = Path(__file__).resolve().parent

for i in range(10):
    split_dir = script_dir / f"test_split_{i}"
    predict_path = split_dir / "predict.tsv"
    output_path = split_dir / "predict-gene-level.tsv"

    df = pd.read_csv(predict_path, sep="\t")
    idx = df.groupby(["query_ids", "target_ids"])["predictions"].idxmax()
    df = df.loc[idx, ["query_ids", "target_ids", "predictions"]]
    df.to_csv(output_path, sep="\t", index=False)
