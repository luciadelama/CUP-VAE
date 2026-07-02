# Load packages needed to read RSE objects, handle matrices, and write CSV files.
suppressPackageStartupMessages({
  library(SummarizedExperiment)
  library(Matrix)
  library(data.table)
})

# ============================================================
# Configuration
# ============================================================
# Directories containing processed TCGA and GTEx RSE files.
tcga_dir   <- "rse_tcga"
gtex_dir   <- "rse_gtex"
assay_name <- "counts_length"

# Output CSV files for the merged count matrix and metadata.
out_counts <- "counts_matrix_all.csv"
out_meta   <- "metadata_all.csv"

# ============================================================
# Helper functions
# ============================================================

# Find RDS files in a directory, optionally only keeping length-normalized files.
get_rds_files <- function(dir_path, cleaned_only = TRUE) {
  if (!dir.exists(dir_path)) {
    warning("Directory does not exist: ", dir_path)
    return(character(0))
  }
  
  pattern <- if (cleaned_only) "\\_length\\.rds$" else "\\.rds$"
  list.files(dir_path, pattern = pattern, full.names = TRUE)
}

# Return the first available metadata column from a list of possible names.
safe_get_col <- function(df, candidates, default = NA_character_) {
  for (nm in candidates) {
    if (nm %in% colnames(df)) {
      return(as.character(df[[nm]]))
    }
  }
  rep(default, nrow(df))
}

# Read one RSE file, extract the selected assay, and create unified metadata columns.
read_rse_project <- function(fp, assay_name, source_label) {
  rse <- readRDS(fp)
  
  if (!assay_name %in% assayNames(rse)) {
    stop(
      "Assay '", assay_name, "' not found in: ", fp,
      "\nAvailable assays: ", paste(assayNames(rse), collapse = ", ")
    )
  }
  
  # Extract the count assay used for the final matrix.
  cnt <- assay(rse, assay_name)
  
  if (!inherits(cnt, "dgCMatrix")) {
    cnt <- as.matrix(cnt)
  }
  
  genes <- rownames(cnt)
  if (is.null(genes)) {
    stop("Counts matrix has no gene rownames in: ", fp)
  }
  
  # Convert sample metadata to a data frame and add consistent identifiers.
  meta <- as.data.frame(colData(rse))
  meta$sample_id <- colnames(cnt)
  meta$project   <- tools::file_path_sans_ext(basename(fp))
  meta$source    <- source_label
  
  # Keep raw metadata columns, but also create light unified columns
  meta$sample_type_raw <- if (source_label == "TCGA") {
    safe_get_col(meta, c("tcga.cgc_sample_sample_type", "sample_type"))
  } else {
    safe_get_col(meta, c("sample_type"))
  }
  
  meta$tissue_raw <- if (source_label == "TCGA") {
    safe_get_col(meta, c("tissue", "tcga.cgc_case_project_primary_site"))
  } else {
    safe_get_col(meta, c("tissue", "gtex.smts", "gtex.smtsd"))
  }
  
  list(
    counts = cnt,
    meta   = meta,
    file   = fp
  )
}

# Merge all RSE projects by keeping genes that are common to every project.
merge_rse_list <- function(rse_list) {
  if (length(rse_list) == 0) {
    stop("No RSE objects were loaded.")
  }
  
  gene_lists <- lapply(rse_list, function(x) rownames(x$counts))
  # Use only genes present in all datasets so the matrices can be combined safely.
  common_genes <- Reduce(intersect, gene_lists)
  
  if (length(common_genes) == 0) {
    stop("No common genes found across all RSE objects.")
  }
  
  message("Common genes across all projects: ", length(common_genes))
  
  counts_list <- lapply(rse_list, function(x) {
    x$counts[common_genes, , drop = FALSE]
  })
  
  # Combine sample columns from all projects into one matrix.
  counts_all <- do.call(cbind, counts_list)
  
  if (anyDuplicated(colnames(counts_all)) > 0) {
    stop("Duplicate sample IDs remain after project-level renaming.")
  }
  
  # Safe row-bind with missing columns filled by NA
  meta_all <- data.table::rbindlist(
    lapply(rse_list, function(x) data.table::as.data.table(x$meta)),
    fill = TRUE,
    use.names = TRUE
  )
  
  meta_all <- as.data.frame(meta_all)
  
  # Reorder metadata to match counts exactly
  meta_all <- meta_all[match(colnames(counts_all), meta_all$sample_id), , drop = FALSE]
  
  if (any(is.na(meta_all$sample_id))) {
    stop("Metadata alignment failed: some count columns could not be matched.")
  }
  
  list(
    counts = counts_all,
    meta   = meta_all
  )
}

# Write the merged matrix and metadata to CSV files.
write_outputs <- function(counts_mat, meta_df, counts_file, meta_file) {
  counts_df <- data.frame(
    gene_id = rownames(counts_mat),
    counts_mat,
    check.names = FALSE
  )
  
  fwrite(counts_df, counts_file)
  fwrite(meta_df, meta_file)
  
  message("Written counts:   ", normalizePath(counts_file))
  message("Written metadata: ", normalizePath(meta_file))
}

# ============================================================
# Main
# ============================================================

# Run the full RSE-to-CSV merging workflow.
main <- function() {
  tcga_files <- get_rds_files(tcga_dir, cleaned_only = TRUE)
  gtex_files <- get_rds_files(gtex_dir, cleaned_only = TRUE)
  
  message("TCGA files found: ", length(tcga_files))
  message("GTEx files found: ", length(gtex_files))
  
  if (length(tcga_files) == 0 && length(gtex_files) == 0) {
    stop("No RDS files found in either directory.")
  }
  
  tcga_list <- lapply(
    tcga_files,
    read_rse_project,
    assay_name   = assay_name,
    source_label = "TCGA"
  )
  
  gtex_list <- lapply(
    gtex_files,
    read_rse_project,
    assay_name   = assay_name,
    source_label = "GTEx"
  )
  
  merged <- merge_rse_list(c(tcga_list, gtex_list))
  
  message(
    "Final matrix dimensions: ",
    nrow(merged$counts), " genes x ", ncol(merged$counts), " samples"
  )
  
  write_outputs(
    counts_mat  = merged$counts,
    meta_df     = merged$meta,
    counts_file = out_counts,
    meta_file   = out_meta
  )
  
  message("Done.")
}

main()
