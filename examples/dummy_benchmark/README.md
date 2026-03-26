Minimal downstream benchmark fixture.

Files:
- `metadata/datasets.json`
- `metadata/predictions.json`
- `data/experiments/processed/DEMO001.tsv`
- `data/predictions/alpha/alpha_canonical.tsv`
- `data/predictions/beta/beta_canonical.tsv`
- `benchmark.yaml`

The config directory is treated as the benchmark root, so the runner
automatically looks for `metadata/` and `data/` next to `benchmark.yaml`.

Run:

```bash
PYTHONPATH=src python -m funmirbench.cli.run_benchmark --config examples/dummy_benchmark/benchmark.yaml
```

Outputs land in:

```text
examples/dummy_benchmark/results/run_demo/
```
