                 ┌──────────────────────────────┐
                 │   Curated metadata (repo)     │
                 │  metadata/mirna_experiment    │
                 └──────────────┬───────────────┘
                                │  ✅ implemented
                                ▼
                 ┌──────────────────────────────┐
                 │ build_experiments_index       │
                 │  -> metadata/datasets.json    │
                 └──────────────┬───────────────┘
                                │
                                │ uses data_path pointers
                                ▼
┌──────────────────────────────┐        ┌──────────────────────────────┐
│   Benchmark corpus (Zenodo)   │        │   User-provided experiments   │
│   processed DE tables (TSV)   │        │   (already processed tables)  │
└──────────────┬───────────────┘        └──────────────┬───────────────┘
               │  ✅ implemented via token             │  ⏳ not implemented (planned)
               ▼                                       ▼
┌──────────────────────────────┐        ┌──────────────────────────────┐
│ import_experiments (Zenodo)   │        │ import_experiments --from-dir │
│ -> data/experiments/processed │        │ -> data/experiments/processed │
└──────────────┬───────────────┘        └──────────────────────────────┘
               │
               │  ✅ implemented
               ▼
┌──────────────────────────────┐
│ validate_experiments          │
│ (existence + readable + gene) │
└──────────────┬───────────────┘
               │
               │
               ▼
        ┌──────────────────────────────────────────────┐
        │          Predictions tool registry (repo)     │
        │     metadata/predictions_info.tsv (source)    │
        └───────────────────────┬──────────────────────┘
                                │ ✅ implemented
                                ▼
                 ┌──────────────────────────────┐
                 │ build_predictions_index       │
                 │ -> metadata/predictions.json  │
                 └──────────────┬───────────────┘
                                │
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ build_predictions --tool mock │
                 │ -> data/predictions/mock/...  │
                 └──────────────┬───────────────┘
                                │  ✅ implemented (mock only)
                                ▼
                 ┌──────────────────────────────┐
                 │ join_experiment_predictions   │
                 │ (DE genes ∩ predicted genes)  │
                 │ -> data/joined/<id>_<tool>.tsv│
                 └──────────────┬───────────────┘
                                │  ✅ implemented
                                ▼
                 ┌──────────────────────────────┐
                 │ plot_correlation              │
                 │ score vs logFC scatter + corr │
                 │ -> data/plots/*.png + *.txt   │
                 └──────────────────────────────┘
                                ✅ implemented
