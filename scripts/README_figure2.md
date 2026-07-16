# Figure 2 Scripts

This folder contains the scripts used to generate the manuscript Figure 2
assets.

## Generate Figure 2

Run from the repository root:

```bash
uv run python scripts/figure2_coverage.py \
  --run-dir results/20260709_122904 \
  --panel all
```

This writes the individual panels and the combined six-panel figure to:

```text
manuscript_assets/figure2/
```

The Figure 2 script reads the ground-truth FDR and effect thresholds from the
completed benchmark run's config snapshot:

```text
<run-dir>/benchmark_config.yaml
```

Regenerate the benchmark run with the current pipeline if that file is missing.

The combined figure is:

```text
manuscript_assets/figure2/figure2_combined.png
manuscript_assets/figure2/figure2_combined.svg
```

Panels:

- A: experiment coverage by predictor
- B: FGS/IGS gene-set coverage
- C: positive miRNA-gene pair coverage
- D: background miRNA-gene pair coverage
- E: longest protein-coding transcript 3'UTR length for IGS vs non-IGS genes
- F: mean 3'UTR conservation for IGS vs non-IGS genes

Panel E derives longest 3'UTR length from:

```text
data/resources/ensembl/Homo_sapiens.GRCh38.115.gtf.gz
```

For each gene, the script keeps one value: the longest 3'UTR among
protein-coding transcripts, after merging transcript-level 3'UTR intervals.

Panel F reads the precomputed gene-level conservation table:

```text
results/manuscript_supplement_utr_conservation/20260709_122904/utr3_conservation_raw.tsv
```

The table must contain:

```text
gene_id
utr3_mean_conservation
```

## Download phastCons100way

If the conservation BigWig is not already available locally, download it with:

```bash
uv run python scripts/download_phastcons100way.py
```

This downloads UCSC `hg38.phastCons100way.bw` to:

```text
data/resources/conservation/hg38.phastCons100way.bw
```

The BigWig is ignored by git because it is large.

## Recompute Conservation Scores

If the gene-level conservation table needs to be regenerated, use the
phastCons100way BigWig with the conservation extraction script from the working
branch history, or restore/adapt it from the implementation that produced:

```text
results/manuscript_supplement_utr_conservation/20260709_122904/utr3_conservation_raw.tsv
```

The Figure 2 script intentionally consumes the precomputed table so that normal
figure regeneration does not require re-reading a multi-GB BigWig.
