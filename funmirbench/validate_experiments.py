"""Validate that experiment DE tables referenced by metadata exist and are readable."""

import argparse
import pathlib

import pandas as pd

from funmirbench.de_table import read_de_table


def main():
    parser = argparse.ArgumentParser(description="Validate experiment DE table files.")
    parser.add_argument("--experiments-tsv", type=pathlib.Path, required=True)
    parser.add_argument("--root", type=pathlib.Path, default=None)
    args = parser.parse_args()

    root = (args.root or args.experiments_tsv.parent).resolve()
    df = pd.read_csv(args.experiments_tsv, sep="\t")

    total = len(df)
    present = 0
    readable = 0
    missing = []
    unreadable = []

    for _, row in df.iterrows():
        path = root / row["de_table_path"]
        if not path.exists():
            missing.append((row["id"], str(path)))
            continue
        present += 1
        try:
            read_de_table(path)
            readable += 1
        except Exception as exc:
            unreadable.append((row["id"], str(exc)))

    print(f"Datasets in metadata: {total}")
    print(f"Files present:        {present}")
    print(f"Readable DE tables:   {readable}")
    if missing:
        print(f"Missing files:        {len(missing)}")
        for dataset_id, path in missing[:20]:
            print(f"  - {dataset_id}: {path}")
    if unreadable:
        print(f"Unreadable files:     {len(unreadable)}")
        for dataset_id, message in unreadable[:20]:
            print(f"  - {dataset_id}: {message}")


if __name__ == "__main__":
    main()
