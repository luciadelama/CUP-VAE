# Load data.table while hiding startup messages.
suppressPackageStartupMessages({
  library(data.table)
})

# ============================================================
# Configuration
# ============================================================

# Input count matrix and gene annotation file used for biotype filtering.
input_counts <- "counts_matrix_filtered.csv"

# This file must contain at least:
#   gene_id
#   gene_type or gene_biotype
input_gene_annotation <- "gene_annotation.csv"

# Output files: filtered counts, retained genes, and a biotype summary table.
output_counts <- "counts_matrix_biotype_filtered.csv"
output_kept_genes <- "genes_kept_by_biotype.csv"
  # Count how many genes were removed or kept for each biotype.
output_biotype_summary <- "biotype_filtering_summary.csv"

# Column names used to match genes between the count matrix and annotation file.
gene_id_col_counts <- "gene_id"
gene_id_col_annot  <- "gene_id"

# Change this if your annotation column is called "gene_biotype"
# Annotation column containing the gene biotype.
biotype_col <- "gene_type"

# Set TRUE if Ensembl IDs contain version numbers, e.g. ENSG00000141510.17
# Remove Ensembl version suffixes so IDs can match across files.
strip_ensembl_version <- TRUE

# Remove genes with zero expression across all samples before/after biotype filtering
# Optionally remove genes that have zero expression across all samples.
remove_zero_expression <- TRUE

# Biotypes to retain
# Note: GENCODE/Ensembl often uses "misc_RNA", not "miscRNA"
# Gene biotypes that are kept for downstream analysis.
retained_biotypes <- c(
  "protein_coding",
  "lncRNA",
  "miRNA",
  "misc_RNA",
  "miscRNA",
  "scRNA",
  "scaRNA",
  "snoRNA",
  "snRNA",
  "sRNA"
)

# ============================================================
# Helper functions
# ============================================================

# Convert gene IDs to character format and optionally remove Ensembl version numbers.
normalise_gene_id <- function(x, strip_version = TRUE) {
  x <- as.character(x)

  if (strip_version) {
    x <- sub("\\..*$", "", x)
  }

  x
}

# Read the count matrix and check that the gene ID column exists.
read_counts <- function(counts_file, gene_id_col) {
  counts_df <- fread(counts_file)

  if (!gene_id_col %in% colnames(counts_df)) {
    stop("Counts file must contain a gene ID column called: ", gene_id_col)
  }

  if (anyDuplicated(counts_df[[gene_id_col]]) > 0) {
    warning("Duplicated gene IDs found in counts matrix. They will be kept as separate rows.")
  }

  counts_df
}

# Read the annotation table and check for gene ID and biotype columns.
read_gene_annotation <- function(annotation_file, gene_id_col, biotype_col) {
  annot_df <- fread(annotation_file)

  if (!gene_id_col %in% colnames(annot_df)) {
    stop("Annotation file must contain a gene ID column called: ", gene_id_col)
  }

  if (!biotype_col %in% colnames(annot_df)) {
    stop("Annotation file must contain a biotype column called: ", biotype_col)
  }

  annot_df
}

# Prepare a clean annotation table with one biotype per gene ID key.
prepare_annotation <- function(annot_df, gene_id_col, biotype_col,
                               strip_ensembl_version = TRUE) {
  annot <- annot_df[, .(
    gene_id_original_annot = get(gene_id_col),
    gene_biotype = get(biotype_col)
  )]

  annot[, gene_id_key := normalise_gene_id(
    gene_id_original_annot,
    strip_version = strip_ensembl_version
  )]

  annot <- annot[!is.na(gene_id_key) & !is.na(gene_biotype)]

  annot <- unique(annot[, .(gene_id_key, gene_biotype)])

  # Identify genes that appear with more than one biotype annotation.
  duplicated_annotation <- annot[, .N, by = gene_id_key][N > 1]

  if (nrow(duplicated_annotation) > 0) {
    warning(
      "Some genes have multiple biotype annotations. Keeping the first annotation for each gene."
    )

    annot <- annot[, .SD[1], by = gene_id_key]
  }

  annot
}

# Merge counts with annotations and keep only selected biotypes with nonzero expression.
filter_counts_by_biotype <- function(counts_df, annot_df,
                                     gene_id_col_counts,
                                     retained_biotypes,
                                     strip_ensembl_version = TRUE,
                                     remove_zero_expression = TRUE) {
  counts_df <- copy(counts_df)

  sample_cols <- setdiff(colnames(counts_df), gene_id_col_counts)

  # Store the original row order so it can be restored after merging.
  counts_df[, row_id := .I]
  counts_df[, gene_id_key := normalise_gene_id(
    get(gene_id_col_counts),
    strip_version = strip_ensembl_version
  )]

  # Add biotype information to each gene in the count matrix.
  counts_annotated <- merge(
    counts_df,
    annot_df,
    by = "gene_id_key",
    all.x = TRUE,
    sort = FALSE
  )

  setorder(counts_annotated, row_id)

  counts_mat <- as.matrix(counts_annotated[, ..sample_cols])
  storage.mode(counts_mat) <- "numeric"

  if (remove_zero_expression) {
    # A gene is considered expressed if at least one sample has a count above zero.
    nonzero_expression <- rowSums(counts_mat > 0, na.rm = TRUE) > 0
  } else {
    nonzero_expression <- rep(TRUE, nrow(counts_annotated))
  }

  # Check whether each gene belongs to one of the retained biotypes.
  has_retained_biotype <- counts_annotated$gene_biotype %in% retained_biotypes

  keep_genes <- has_retained_biotype & nonzero_expression

  counts_filtered <- counts_annotated[
    keep_genes,
    c(gene_id_col_counts, sample_cols),
    with = FALSE
  ]

  kept_genes <- counts_annotated[
    keep_genes,
    .(
      gene_id = get(gene_id_col_counts),
      gene_biotype
    )
  ]

  # Add diagnostic columns used to create the filtering summary.
  counts_annotated[, zero_expression := !nonzero_expression]
  counts_annotated[, retained_biotype := has_retained_biotype]
  counts_annotated[, kept := keep_genes]

  counts_annotated[is.na(gene_biotype), gene_biotype := "unannotated"]

  biotype_summary <- counts_annotated[
    ,
    .(
      n_input_genes = .N,
      n_zero_expression = sum(zero_expression),
      n_retained_biotype = sum(retained_biotype),
      n_kept_final = sum(kept)
    ),
    by = gene_biotype
  ][order(-n_kept_final, -n_input_genes)]

  list(
    counts_filtered = counts_filtered,
    kept_genes = kept_genes,
    biotype_summary = biotype_summary,
    n_input_genes = nrow(counts_df),
    n_output_genes = nrow(counts_filtered),
    n_unannotated = sum(counts_annotated$gene_biotype == "unannotated")
  )
}

# Print the main filtering results and output file locations.
print_summary <- function(result, output_counts, output_kept_genes,
                          output_biotype_summary) {
  message("Gene biotype filtering completed.")
  message("Input genes:       ", result$n_input_genes)
  message("Output genes:      ", result$n_output_genes)
  message("Unannotated genes: ", result$n_unannotated)
  message("Output counts:     ", normalizePath(output_counts))
  message("Kept genes file:   ", normalizePath(output_kept_genes))
  message("Summary file:      ", normalizePath(output_biotype_summary))
}

# ============================================================
# Main
# ============================================================

# Run the full biotype-filtering workflow.
main <- function() {
  counts_df <- read_counts(
    counts_file = input_counts,
    gene_id_col = gene_id_col_counts
  )

  annot_raw <- read_gene_annotation(
    annotation_file = input_gene_annotation,
    gene_id_col = gene_id_col_annot,
    biotype_col = biotype_col
  )

  annot_df <- prepare_annotation(
    annot_df = annot_raw,
    gene_id_col = gene_id_col_annot,
    biotype_col = biotype_col,
    strip_ensembl_version = strip_ensembl_version
  )

  result <- filter_counts_by_biotype(
    counts_df = counts_df,
    annot_df = annot_df,
    gene_id_col_counts = gene_id_col_counts,
    retained_biotypes = retained_biotypes,
    strip_ensembl_version = strip_ensembl_version,
    remove_zero_expression = remove_zero_expression
  )

  fwrite(result$counts_filtered, output_counts)
  fwrite(result$kept_genes, output_kept_genes)
  fwrite(result$biotype_summary, output_biotype_summary)

  print_summary(
    result = result,
    output_counts = output_counts,
    output_kept_genes = output_kept_genes,
    output_biotype_summary = output_biotype_summary
  )
}

main()