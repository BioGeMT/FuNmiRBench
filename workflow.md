
# FuNmiRBench pipeline overview

Legend:
✅ implemented
⏳ planned / not implemented

```
                 ┌──────────────────────────────┐
                 │   Curated metadata (repo)     │
                 │  mirna_experiment_info.tsv    │
                 └──────────────┬───────────────┘
                                │  ✅
                                ▼
                 ┌──────────────────────────────┐
                 │ build_experiments_index       │
                 │  → metadata/datasets.json    │
                 └──────────────┬───────────────┘
                                │
                                ▼
┌──────────────────────────────┐        ┌──────────────────────────────┐
│   Benchmark corpus (Zenodo)   │        │   User-provided experiments   │
│   processed DE tables (TSV)   │        │   (processed tables)          │
└──────────────┬───────────────┘        └──────────────┬───────────────┘
               │  ✅                               │  ⏳
               ▼                                   ▼
┌──────────────────────────────┐        ┌──────────────────────────────┐
│ import_experiments (Zenodo)   │        │ import_experiments --from-dir │
│ → data/experiments/processed │        │ → data/experiments/processed │
└──────────────┬───────────────┘        └──────────────────────────────┘
               │
               │  ✅
               ▼
┌──────────────────────────────┐
│ validate_experiments          │
│ (existence + readable + gene) │
└──────────────┬───────────────┘
               │
               ▼
        ┌──────────────────────────────────────────────┐
        │     Prediction tool metadata (repo)           │
        │     predictions_info.tsv (source of truth)    │
        └───────────────────────┬──────────────────────┘
                                │  ✅
                                ▼
                 ┌──────────────────────────────┐
                 │ build_predictions_index       │
                 │ → metadata/predictions.json  │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ build_predictions --tool mock │
                 │ → data/predictions/mock/*.tsv│
                 └──────────────┬───────────────┘
                                │  ✅ (mock only)
                                ▼
                 ┌──────────────────────────────┐
                 │ join_experiment_predictions   │
                 │ → data/joined/<id>_<tool>.tsv │
                 └──────────────┬───────────────┘
                                │  ✅
                                ▼
                 ┌──────────────────────────────┐
                 │ plot_correlation              │
                 │ → data/plots/*.png + *.txt   │
                 └──────────────────────────────┘
                                ✅

Not implemented yet:
- User-provided experiment ingestion (--from-dir)
- Real evaluation metrics (AUC, PR, enrichment)
- Full RNA-seq processing pipeline
```
