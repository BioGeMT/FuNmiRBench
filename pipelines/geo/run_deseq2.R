args <- commandArgs(trailingOnly = TRUE)

read_arg <- function(flag) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) {
    stop(sprintf("Missing required argument: %s", flag), call. = FALSE)
  }
  args[[idx + 1]]
}

sample_sheet_path <- read_arg("--sample-sheet")
tx2gene_path <- read_arg("--tx2gene")
output_path <- read_arg("--output")

suppressPackageStartupMessages(library(tximport))
suppressPackageStartupMessages(library(DESeq2))

samples <- read.delim(sample_sheet_path, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
if (!all(c("sample_id", "condition", "quant_sf") %in% colnames(samples))) {
  stop("Sample sheet must contain sample_id, condition, and quant_sf columns.", call. = FALSE)
}

samples$sample_id <- trimws(samples$sample_id)
samples$condition <- trimws(samples$condition)
samples$quant_sf <- trimws(samples$quant_sf)

if (length(unique(samples$condition)) != 2) {
  stop("Exactly two conditions are required for DESeq2.", call. = FALSE)
}
if (!all(c("control", "treated") %in% unique(samples$condition))) {
  stop("Conditions must include control and treated.", call. = FALSE)
}

files <- samples$quant_sf
names(files) <- samples$sample_id

tx2gene <- read.delim(tx2gene_path, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
if (ncol(tx2gene) < 2) {
  stop("tx2gene table must have at least two columns.", call. = FALSE)
}
tx2gene <- tx2gene[, 1:2]
colnames(tx2gene) <- c("transcript_id", "gene_id")

txi <- tximport(files, type = "salmon", tx2gene = tx2gene, ignoreTxVersion = TRUE)

col_data <- data.frame(
  row.names = samples$sample_id,
  condition = factor(samples$condition, levels = c("control", "treated"))
)

dds <- DESeqDataSetFromTximport(txi, colData = col_data, design = ~ condition)
dds <- dds[rowSums(counts(dds)) > 0, ]
dds <- DESeq(dds)
res <- results(dds, contrast = c("condition", "treated", "control"))

out <- data.frame(
  gene_id = rownames(res),
  logFC = res$log2FoldChange,
  PValue = res$pvalue,
  FDR = res$padj,
  stringsAsFactors = FALSE
)
out <- out[order(out$PValue, na.last = TRUE), ]

write.table(out, file = output_path, sep = "\t", quote = FALSE, row.names = FALSE)
