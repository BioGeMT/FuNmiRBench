# FuNmiRBench workflow

Legend:
✅ implemented
⏳ planned / not implemented

---

## High-level pipeline overview

```
                 ┌──────────────────────────────────────┐
                 │      Curated dataset metadata (repo)  │
                 │      metadata/mirna_experiment_info.tsv│
                 └──────────────────────┬───────────────┘
                                        │ ✅
                                        ▼
                 ┌──────────────────────────────────────┐
                 │ build_experiments_index              │
                 │  → metadata/datasets.json            │
                 └──────────────────────┬───────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────┐        ┌──────────────────────────────────────┐ 
│ Benchmark corpus (Zenodo)            │        │ User-provided processed experiments  │
│ processed DE tables (TSV)            │        │ (same TSV schema as benchmark)       │
└──────────────────────┬───────────────┘        └──────────────────────┬───────────────┘
                       │ ✅                                   │ ⏳
                       ▼                                      ▼
┌──────────────────────────────────────┐        ┌──────────────────────────────────────┐
│ import_experiments (Zenodo download) │        │ import_experiments --from-dir        │
│ → data/experiments/processed/        │        │ → data/experiments/processed/        │
└──────────────────────┬───────────────┘        └──────────────────────────────────────┘
                       │ ✅
                       ▼
┌──────────────────────────────────────┐
│ validate_experiments                 │
│ - path exists                         │
│ - readable TSV                         │
│ - gene identifier column detectable    │
└──────────────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────────┐
        │     Curated prediction tool metadata (repo)   │
        │     metadata/predictions_info.tsv             │
        └───────────────────────┬──────────────────────┘
                                │ ✅
                                ▼
                 ┌──────────────────────────────────────┐
                 │ build_predictions_index              │
                 │  → metadata/predictions.json         │
                 └──────────────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────────────┐
                 │ build_predictions --tool <tool>       │
                 │  → data/predictions/<tool>/*.tsv      │
                 └──────────────────────┬───────────────┘
                                │ ✅ (mock only today)
                                ▼
                 ┌──────────────────────────────────────┐
                 │ join_experiment_predictions           │
                 │  → data/joined/<dataset>_<tool>.tsv   │
                 └──────────────────────┬───────────────┘
                                │ ✅
                                ▼
                 ┌──────────────────────────────────────┐
                 │ plot_correlation                      │
                 │  → data/plots/*.png + *.txt           │
                 └──────────────────────────────────────┘
                                ✅
```

---

## Concrete artifacts and where they live

### Inputs (repo-tracked)
- `metadata/mirna_experiment_info.tsv` — dataset registry source of truth
- `metadata/predictions_info.tsv` — prediction tool registry source of truth

### Derived indexes (generated)
- `metadata/datasets.json` — built from `mirna_experiment_info.tsv`
- `metadata/predictions.json` — built from `predictions_info.tsv`

### Local data (not necessarily repo-tracked)
- `data/experiments/processed/*.tsv` — processed DE tables (edgeR outputs)
- `data/predictions/<tool>/*.tsv` — canonical miRNA–gene score tables
- `data/joined/*.tsv` — joined DE + prediction tables
- `data/plots/*` — plots + small text summaries

---

## CLI workflow (current)

### 1) Import benchmark experiments (Zenodo → local)
Download processed DE tables into `data/experiments/processed/`:

```bash
python -m funmirbench.cli.import_experiments --token "<TOKEN>"
# or: export ZENODO_TOKEN=... and omit --token
```

✅ Implemented: downloads only; does not change curated metadata.

### 2) Build experiments index
```bash
python -m funmirbench.cli.build_experiments_index
```
Produces: `metadata/datasets.json`.

### 3) Validate local experiments
```bash
python -m funmirbench.cli.validate_experiments
```
Recommended before running predictions.

### 4) Generate prediction scores
```bash
python -m funmirbench.cli.build_predictions --tool mock
```
Produces: `data/predictions/mock/*.tsv`.

### 5) Build predictions index
```bash
python -m funmirbench.cli.build_predictions_index
```
Produces: `metadata/predictions.json`.

### 6) Join one dataset with one tool
```bash
python -m funmirbench.cli.join_experiment_predictions \
  --dataset-id 001 \
  --tool mock \
  --out data/joined/001_mock.tsv
```

### 7) Plot correlation
```bash
python -m funmirbench.cli.plot_correlation \
  --joined-tsv data/joined/001_mock.tsv \
  --out-dir data/plots
```
Produces:
- a PNG scatter plot
- a TXT summary with Pearson/Spearman correlations

---

## Planned / not implemented yet
- ⏳ `import_experiments --from-dir` for user-provided processed TSVs
- ⏳ additional evaluation metrics (PR/ROC, enrichment, AUROC/AUPRC, etc.)
- ⏳ full RNA-seq processing pipeline (raw → counts → DE)
- ⏳ multi-dataset / multi-tool batch evaluation + dashboard-ready summaries
