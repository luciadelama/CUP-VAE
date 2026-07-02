# Load data.table while suppressing package startup messages.
suppressPackageStartupMessages({
  library(data.table)
})

# ============================================================
# Configuration
# ============================================================
# Input files containing all samples before biological sample filtering.
input_counts <- "counts_matrix_all.csv"
input_meta   <- "metadata_all.csv"

# Output files containing the filtered count matrix and metadata.
out_counts <- "counts_matrix_filtered.csv"
out_meta   <- "metadata_filtered.csv"

# ============================================================
# Helper functions
# ============================================================

# Read the merged count matrix and metadata, then check required columns.
read_merged_data <- function(counts_file, meta_file) {
  counts_df <- fread(counts_file)
  meta_df   <- fread(meta_file)

  if (!"gene_id" %in% colnames(counts_df)) {
    stop("The counts file must contain a 'gene_id' column.")
  }

  required_meta_cols <- c("sample_id", "project", "source")
  missing_meta_cols <- setdiff(required_meta_cols, colnames(meta_df))

  if (length(missing_meta_cols) > 0) {
    stop(
      "The metadata file is missing required columns: ",
      paste(missing_meta_cols, collapse = ", ")
    )
  }

  list(
    counts = counts_df,
    meta   = meta_df
  )
}

# Map less common TCGA sample type labels to the main categories used here.
standardize_tcga_sample_type <- function(sample_type) {
  sample_type <- as.character(sample_type)

  sample_type[sample_type == "Additional - New Primary"] <- "Primary Tumor"
  sample_type[sample_type == "Primary Blood Derived Cancer - Peripheral Blood"] <- "Primary Tumor"
  sample_type[sample_type == "Additional Metastatic"] <- "Metastatic"

  sample_type
}

# Harmonize tissue names so related TCGA and GTEx tissues use the same label.
harmonize_tissue_name <- function(tissue, source) {
  tissue <- as.character(tissue)
  source <- as.character(source)

  tissue[source == "TCGA" & tissue == "Colorectal"] <- "Colon"
  tissue[source == "GTEx" & tissue == "Cervix Uteri"] <- "Cervix"
  tissue[source == "GTEx" & tissue == "Fallopian Tube"] <- "Ovary"

  tissue
}

# Create cleaned metadata columns used for filtering and downstream labels.
build_curated_metadata <- function(meta_df) {
  meta <- copy(meta_df)

  # ----------------------------------------------------------
  # Exclude duplicated GTEx study
  # ----------------------------------------------------------
  # Remove a duplicated GTEx study entry before filtering.
  meta <- meta[!(source == "GTEx" & project == "STUDY_NA")]

  # ----------------------------------------------------------
  # Standardize TCGA sample types
  # ----------------------------------------------------------
  if (!"sample_type_raw" %in% colnames(meta)) {
    meta[, sample_type_raw := NA_character_]
  }

  meta[, sample_type := sample_type_raw]
  meta[source == "TCGA", sample_type := standardize_tcga_sample_type(sample_type_raw)]
  meta[source == "GTEx", sample_type := "Normal"]

  # ----------------------------------------------------------
  # Harmonize tissue names
  # ----------------------------------------------------------
  if (!"tissue_raw" %in% colnames(meta)) {
    meta[, tissue_raw := NA_character_]
  }

  meta[, tissue := harmonize_tissue_name(tissue_raw, source)]

  meta
}

# Keep TCGA tumor/normal samples and all GTEx normal samples.
filter_metadata <- function(meta_df) {
  keep_tcga <- meta_df$source == "TCGA" &
    meta_df$sample_type %in% c("Primary Tumor", "Solid Tissue Normal")

  keep_gtex <- meta_df$source == "GTEx"

  keep <- keep_tcga | keep_gtex

  meta_filt <- meta_df[keep, , drop = FALSE]

  if (nrow(meta_filt) == 0) {
    stop("No samples passed the filtering criteria.")
  }

  meta_filt
}

# Subset the count matrix so it contains only the samples kept in metadata.
filter_counts_by_metadata <- function(counts_df, meta_df) {
  sample_cols <- meta_df$sample_id

  missing_samples <- setdiff(sample_cols, colnames(counts_df))
  if (length(missing_samples) > 0) {
    stop(
      "Some metadata sample IDs were not found in the counts matrix: ",
      paste(head(missing_samples, 10), collapse = ", "),
      if (length(missing_samples) > 10) " ..." else ""
    )
  }

  counts_df[, c("gene_id", sample_cols), with = FALSE]
}

# Save the filtered count matrix and metadata to CSV files.
write_filtered_outputs <- function(counts_df, meta_df, counts_file, meta_file) {
  fwrite(counts_df, counts_file)
  fwrite(meta_df, meta_file)

  message("Written filtered counts:   ", normalizePath(counts_file))
  message("Written filtered metadata: ", normalizePath(meta_file))
}

# Summarize the filtered samples by source, sample type, and tissue.
summarize_filtered_samples <- function(meta_df) {
  as.data.table(meta_df)[, .N, by = .(source, sample_type, tissue)][
    order(source, sample_type, tissue)
  ]
}

# Summarize how many TCGA samples remain per project and sample type.
summarize_tcga_projects <- function(meta_df) {
  tcga_meta <- meta_df[source == "TCGA"]

  if (nrow(tcga_meta) == 0) {
    return(data.table())
  }

  tcga_meta[, .N, by = .(project, sample_type)][order(project, sample_type)]
}

# ============================================================
# Main
# ============================================================

# Run the full sample-filtering workflow.
main <- function() {
  merged <- read_merged_data(input_counts, input_meta)

  meta_curated <- build_curated_metadata(merged$meta)
  meta_filtered <- filter_metadata(meta_curated)
  counts_filtered <- filter_counts_by_metadata(merged$counts, meta_filtered)

  write_filtered_outputs(
    counts_df   = counts_filtered,
    meta_df     = meta_filtered,
    counts_file = out_counts,
    meta_file   = out_meta
  )

  message(
    "Filtered dataset dimensions: ",
    nrow(counts_filtered), " genes x ", ncol(counts_filtered) - 1, " samples"
  )

  message("\nFiltered sample summary:")
  print(summarize_filtered_samples(meta_filtered))

  message("\nTCGA project summary:")
  print(summarize_tcga_projects(meta_filtered))

  message("\nDone.")
}

main()