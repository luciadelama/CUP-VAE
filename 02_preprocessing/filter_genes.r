# Load required packages while hiding startup messages to keep the console output clean.
suppressPackageStartupMessages({
  library(Matrix)
  library(data.table)
})

# ============================================================
# Configuration
# ============================================================
# Input files produced by the previous sample-filtering step.
input_counts <- "counts_matrix_filtered.csv"
input_meta   <- "metadata_filtered.csv"

output_counts <- "counts_matrix_length_filtered_hand.csv"

# A gene is kept if it reaches this minimum count threshold in enough samples.
min_count <- 10
min_prop  <- 0.70
# Filtering is applied within this metadata group, here one project at a time.
group_var <- "project"

# ============================================================
# Helper functions
# ============================================================

# Read the count matrix and metadata, then check that the required columns exist.
read_input_data <- function(counts_file, meta_file) {
  counts_df <- fread(counts_file)
  meta_df   <- fread(meta_file)

  if (!"gene_id" %in% colnames(counts_df)) {
    stop("Counts file must contain a 'gene_id' column.")
  }

  if (!"sample_id" %in% colnames(meta_df)) {
    stop("Metadata file must contain a 'sample_id' column.")
  }

  if (!group_var %in% colnames(meta_df)) {
    stop("Metadata file must contain the grouping column: '", group_var, "'.")
  }

  list(counts = counts_df, meta = meta_df)
}

# Make sure the metadata samples match the count matrix columns and are in the same order.
align_counts_and_metadata <- function(counts_df, meta_df) {
  count_sample_ids <- setdiff(colnames(counts_df), "gene_id")
  meta_sample_ids  <- meta_df$sample_id

  missing_in_meta <- setdiff(count_sample_ids, meta_sample_ids)
  if (length(missing_in_meta) > 0) {
    stop(
      "The following count matrix samples are missing in metadata: ",
      paste(head(missing_in_meta, 10), collapse = ", "),
      if (length(missing_in_meta) > 10) " ..." else ""
    )
  }

  missing_in_counts <- setdiff(meta_sample_ids, count_sample_ids)
  if (length(missing_in_counts) > 0) {
    warning(
      "The following metadata samples are missing in the counts matrix and will be ignored: ",
      paste(head(missing_in_counts, 10), collapse = ", "),
      if (length(missing_in_counts) > 10) " ..." else ""
    )
    meta_df <- meta_df[sample_id %in% count_sample_ids]
  }

  meta_df <- meta_df[match(count_sample_ids, sample_id)]

  if (any(is.na(meta_df$sample_id))) {
    stop("Failed to align metadata to the counts matrix columns.")
  }

  list(counts = counts_df, meta = meta_df)
}

# Convert the count table into a numeric matrix with gene IDs as row names.
build_counts_matrix <- function(counts_df) {
  gene_ids <- counts_df[["gene_id"]]
  count_dt <- counts_df[, !"gene_id"]

  counts_mat <- as.matrix(count_dt)

  storage.mode(counts_mat) <- "numeric"
  rownames(counts_mat) <- gene_ids

  counts_mat
}

# Keep genes that are sufficiently expressed in at least one project/group.
filter_genes_by_group <- function(counts_mat, meta_df, group_column,
                                  min_count = 10, min_prop = 0.70) {
  group <- factor(meta_df[[group_column]])

  if (any(is.na(group))) {
    stop("Grouping column contains missing values.")
  }

  # For each group, check the proportion of samples where each gene passes the count threshold.
  keep_by_group <- sapply(levels(group), function(g) {
    idx <- which(group == g)

    if (length(idx) == 0) {
      return(rep(FALSE, nrow(counts_mat)))
    }

    rowMeans(counts_mat[, idx, drop = FALSE] >= min_count) >= min_prop
  })

  if (is.vector(keep_by_group)) {
    keep_by_group <- matrix(keep_by_group, ncol = 1)
  }

  # Keep a gene if it passes the expression rule in at least one group.
  keep_genes <- rowSums(keep_by_group) > 0

  list(
    keep_genes    = keep_genes,
    counts_mat    = counts_mat[keep_genes, , drop = FALSE],
    gene_ids_kept = rownames(counts_mat)[keep_genes]
  )
}

# Save the filtered matrix back to CSV with gene IDs restored as a column.
write_filtered_counts <- function(counts_mat, output_file) {
  counts_out <- data.frame(
    gene_id = rownames(counts_mat),
    counts_mat,
    check.names = FALSE
  )

  fwrite(counts_out, output_file)
  counts_out
}

# Print a short summary so the user can verify how many genes were kept.
print_summary <- function(n_genes_before, n_genes_after, n_samples,
                          min_count, min_prop, group_var, output_file) {
  message("Gene filtering completed.")
  message("Input genes:  ", n_genes_before)
  message("Output genes: ", n_genes_after)
  message("Samples:      ", n_samples)
  message("Rule: keep genes with at least ", min_count,
          " counts in at least ", round(min_prop * 100),
          "% of samples within at least one ", group_var, ".")
  message("Output file:  ", normalizePath(output_file))
}

# ============================================================
# Main
# ============================================================

# Run the full gene-filtering workflow.
main <- function() {
  input_data <- read_input_data(input_counts, input_meta)

  aligned <- align_counts_and_metadata(
    counts_df = input_data$counts,
    meta_df   = input_data$meta
  )

  counts_mat <- build_counts_matrix(aligned$counts)

  n_genes_before <- nrow(counts_mat)
  n_samples      <- ncol(counts_mat)

  filtered <- filter_genes_by_group(
    counts_mat   = counts_mat,
    meta_df      = aligned$meta,
    group_column = group_var,
    min_count    = min_count,
    min_prop     = min_prop
  )

  counts_out <- write_filtered_counts(
    counts_mat  = filtered$counts_mat,
    output_file = output_counts
  )

  print_summary(
    n_genes_before = n_genes_before,
    n_genes_after  = nrow(counts_out),
    n_samples      = n_samples,
    min_count      = min_count,
    min_prop       = min_prop,
    group_var      = group_var,
    output_file    = output_counts
  )
}

main()