# =========================================================
# Download recount3 TCGA and GTEx projects and save them as RSE objects.
# Download recount3 projects and save RSE objects
# =========================================================

# -----------------------------
# Packages
# -----------------------------
# Install required packages only if they are not already available.
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager")
}

pkgs <- c("recount3", "SummarizedExperiment")

for (pkg in pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    BiocManager::install(pkg, ask = FALSE, update = FALSE)
  }
}

library(recount3)
library(SummarizedExperiment)


# -----------------------------
# Config
# -----------------------------
# Output directories and assay name used for saved RSE objects.
OUTDIR_TCGA <- "RSE_TCGA"
OUTDIR_GTEX <- "RSE_GTEx"
ASSAY_NAME <- "counts"

# Create output folders if they do not already exist.
dir.create(OUTDIR_TCGA, showWarnings = FALSE, recursive = TRUE)
dir.create(OUTDIR_GTEX, showWarnings = FALSE, recursive = TRUE)

# -----------------------------
# Helper: process one project
# -----------------------------
# Download and save one recount3 project as an RSE object.
process_project <- function(proj_info, source = c("tcga", "gtex"),
                            assay_name = ASSAY_NAME,
                            overwrite = FALSE) {
  
  source <- match.arg(source)
  project_name <- as.character(proj_info$project)
  
  outdir <- if (source == "tcga") OUTDIR_TCGA else OUTDIR_GTEX
  outfile <- file.path(outdir, paste0(source, "_", project_name, ".rds"))
  
  # Skip projects that were already downloaded unless overwrite is enabled.
  if (file.exists(outfile) && !overwrite) {
    message("Skipping ", project_name, " (already exists)")
    return(invisible(NULL))
  }
  
  message("Downloading ", source, " project: ", project_name)
  
  # Download project as RSE
  # Create the RSE object from recount3 project information.
  rse <- create_rse(proj_info)
  
  # Add scaled counts as a new assay
  # Store transformed counts in the selected assay name.
  assay(rse, assay_name) <- transform_counts(rse)
  
  # Save RSE
  saveRDS(rse, outfile)
  
  message("Saved: ", outfile)
}


# -----------------------------
# Helper: process all projects from one source
# -----------------------------
# Loop over all projects from one data source and process them one by one.
process_all_projects <- function(source = c("tcga", "gtex"),
                                 overwrite = FALSE) {
  
  source <- match.arg(source)
  
# Retrieve the recount3 project table.
  human_projects <- available_projects()
  projects <- subset(human_projects, file_source == source)
  
  for (i in seq_len(nrow(projects))) {
    proj_info <- projects[i, , drop = FALSE]
    
    # Continue with the next project if one download fails.
    tryCatch(
      {
        process_project(
          proj_info   = proj_info,
          source      = source,
          overwrite   = overwrite
        )
      },
      error = function(e) {
        project_name <- as.character(proj_info$project)
        message("Error in ", project_name, ": ", e$message)
      }
    )
  }
}

# -----------------------------
# Create and save project_info table
# -----------------------------
message("Creating project_info table...")

human_projects <- available_projects()

# Keep only TCGA and GTEx project metadata for reference.
project_info <- subset(
  human_projects,
  file_source %in% c("tcga", "gtex")
)

project_info <- project_info[, c(
  "project",
  "project_home",
  "file_source",
  "project_type",
  "n_samples"
)]

# Add status column
# Add a status column that can be updated later if needed.
project_info$download_status <- "pending"

# Sort (very useful)
# Sort projects by source and sample size to make the table easier to inspect.
project_info <- project_info[order(
  project_info$file_source,
  -project_info$n_samples
), ]

# Save
# Save the project information in CSV and RDS formats.
write.csv(
  project_info,
  file = "project_info.csv",
  row.names = FALSE
)

saveRDS(
  project_info,
  file = "project_info.rds"
)

message("Saved project_info with n_samples")


# -----------------------------
# Run
# -----------------------------
# Process all TCGA projects
# Download all TCGA projects.
process_all_projects(source = "tcga", overwrite = FALSE)

# Process all GTEx projects
# Download all GTEx projects.
process_all_projects(source = "gtex", overwrite = FALSE)
