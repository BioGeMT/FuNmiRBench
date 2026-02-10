# Data layout for FuNmiRBench

FuNmiRBench does **not** store large experiment tables or prediction outputs in git.

All local data lives under the `data/` folder, which is ignored by default.

---

## 📁 Expected folder structure

```
data/
├── experiments/
│   ├── processed/     # processed DE tables (TSV)
│   └── raw/           # optional raw / intermediate files
│
├── predictions/       # canonical tool outputs (TSV)
├── joined/            # joined experiment + prediction tables (TSV)
└── plots/             # output plots (PNG + TXT summaries)
```

---

## Processed experiment tables

Each file in:

```
data/experiments/processed/
```

- is a tab-separated `.tsv`
- contains differential expression results for one experiment
- is typically an edgeR output table

### Gene identifiers

edgeR outputs do not always include a gene column named `gene_name`.

FuNmiRBench supports both of the following formats:

#### Format A (explicit gene column)
```text
gene_name    logFC    logCPM    F    PValue    FDR
ENSG00000123456   ...
```

#### Format B (gene IDs stored as row names / first column)
```text
logFC    logCPM    F    PValue    FDR
ENSG00000123456   ...
```

FuNmiRBench detects gene IDs robustly (preferring Ensembl-like IDs such as `ENSG...`).

---

## Linking data to metadata

The file paths are referenced in `metadata/datasets.json` via the `data_path` field, e.g.:

```json
{
  "id": "001",
  "data_path": "data/experiments/processed/GSE210778_edger_out_oe_hsa_miR_375_3p_oe.tsv",
  "miRNA": "hsa-miR-375-3p",
  "perturbation": "overexpression"
}
```

---

## Downloading benchmark experiments

To download the published benchmark experiment corpus from Zenodo:

```bash
python -m funmirbench.cli.import_experiments --token "<TOKEN>"
```

This will populate:

```
data/experiments/processed/
```

---

## Notes

- `data/` is intended for **local reproducibility**, not version control.
- `metadata/` contains the versioned curated registry of what datasets exist.
