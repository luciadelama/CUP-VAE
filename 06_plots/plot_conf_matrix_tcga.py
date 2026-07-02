#!/usr/bin/env python3

import os
from matplotlib.colors import PowerNorm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =========================================================
# Input and output paths
# =========================================================
RESULTS_DIR = "."
METADATA_PATH = "../../data/metadata/final_sample_metadata_collapsed_replicates.csv"

PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR,
    "mlr_elastic_net_test_predictions.csv"
)

OUTPUT_DIR = os.path.join(RESULTS_DIR, "tcga_project_vs_predicted_tissue")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# Expected mapping from TCGA project code to tissue label
# =========================================================
expected_tissue_map = {
    "ACC": "Adrenal Gland",
    "BLCA": "Bladder",
    "BRCA": "Breast",
    "CESC": "Cervix",
    "CHOL": "Bile Duct",
    "COAD": "Colorectal",
    "DLBC": "Lymph Nodes",
    "ESCA": "Esophagus",
    "GBM": "Brain",
    "HNSC": "Head and Neck",
    "KICH": "Kidney",
    "KIRC": "Kidney",
    "KIRP": "Kidney",
    "LAML": "Bone Marrow",
    "LGG": "Brain",
    "LIHC": "Liver",
    "LUAD": "Lung",
    "LUSC": "Lung",
    "MESO": "Pleura",
    "OV": "Ovary",
    "PAAD": "Pancreas",
    "PCPG": "Adrenal Gland",
    "PRAD": "Prostate",
    "READ": "Colorectal",
    "SARC": "Soft Tissue",
    "SKCM": "Skin",
    "STAD": "Stomach",
    "TGCT": "Testis",
    "THCA": "Thyroid",
    "THYM": "Thymus",
    "UCEC": "Uterus",
    "UCS": "Uterus",
    "UVM": "Eye"
}

# =========================================================
# Load prediction and metadata tables
# =========================================================
pred_df = pd.read_csv(PREDICTIONS_PATH)
meta_df = pd.read_csv(METADATA_PATH)

required_pred_cols = {"sample_id", "predicted_label"}
required_meta_cols = {"sample_id", "dataset", "project"}

missing_pred = required_pred_cols - set(pred_df.columns)
missing_meta = required_meta_cols - set(meta_df.columns)

if missing_pred:
    raise ValueError(f"Missing columns in predictions file: {missing_pred}")

if missing_meta:
    raise ValueError(f"Missing columns in metadata file: {missing_meta}")

# =========================================================
# Merge predictions with metadata to recover dataset and project information
# =========================================================
df = pred_df.merge(
    meta_df[["sample_id", "dataset", "project"]],
    on="sample_id",
    how="left"
)

# Keep only TCGA samples from the test predictions.
tcga_df = df[df["dataset"] == "TCGA"].copy()

if tcga_df.empty:
    raise ValueError("No TCGA samples found in test predictions.")

print("Total test samples:", len(df))
print("TCGA test samples:", len(tcga_df))
print("Number of TCGA projects:", tcga_df["project"].nunique())
print("Number of predicted tissues:", tcga_df["predicted_label"].nunique())

# =========================================================
# Build correspondence matrix: TCGA project x predicted tissue
# =========================================================
# Normalize by TCGA project so each row sums to 100%.
cm_percent = pd.crosstab(
    tcga_df["project"],
    tcga_df["predicted_label"],
    normalize="index"
) * 100

# =========================================================
# Reorder predicted tissue columns alphabetically
# =========================================================
cm_percent = cm_percent.reindex(sorted(cm_percent.columns), axis=1)

# =========================================================
# Reorder rows according to the expected tissue for each TCGA project
# =========================================================
row_info = pd.DataFrame({
    "project": cm_percent.index,
    "expected_tissue": [expected_tissue_map.get(project, "Unknown") for project in cm_percent.index]
})

row_info["expected_tissue_rank"] = row_info["expected_tissue"].map(
    {tissue: i for i, tissue in enumerate(cm_percent.columns)}
)

# Projects with an expected tissue not present in columns go at the end
row_info["expected_tissue_rank"] = row_info["expected_tissue_rank"].fillna(999)

row_order = (
    row_info
    .sort_values(["expected_tissue_rank", "expected_tissue", "project"])
    ["project"]
    .tolist()
)

cm_percent = cm_percent.loc[row_order]

# Save the reordered percentage matrix
matrix_path = os.path.join(
    OUTPUT_DIR,
    "tcga_project_vs_predicted_tissue_percent_ordered.csv"
)
cm_percent.to_csv(matrix_path)

# =========================================================
# Create selective annotations for non-zero values only
# =========================================================
annot = cm_percent.copy().astype(str)

for i in range(cm_percent.shape[0]):
    for j in range(cm_percent.shape[1]):
        value = cm_percent.iloc[i, j]

        # Show only non-zero values.
        # No artificial diagonal is used because rows and columns are different variables.
        if value > 0:
            if np.isclose(value, round(value)):
                annot.iloc[i, j] = f"{int(round(value))}"
            else:
                annot.iloc[i, j] = f"{value:.1f}"
        else:
            annot.iloc[i, j] = ""

# =========================================================
# Plot the TCGA project versus predicted tissue heatmap
# =========================================================
n_rows = cm_percent.shape[0]
n_cols = cm_percent.shape[1]

plt.figure(figsize=(max(16, n_cols * 0.55), max(8, n_rows * 0.32)))

# Use PowerNorm to make lower percentages easier to see.
ax = sns.heatmap(
    cm_percent,
    annot=annot,
    fmt="",
    cmap="Blues",
    norm=PowerNorm(gamma=0.5, vmin=0, vmax=100),
    linewidths=0.2,
    linecolor="lightgrey",
    cbar_kws={"label": "% of samples within TCGA project"}
)

plt.xlabel("Predicted tissue", fontsize=14)
plt.ylabel("TCGA project", fontsize=14)
plt.title(
    "Elastic Net MLR predicted tissue distribution by TCGA project",
    fontsize=16,
    pad=12
)

plt.xticks(rotation=45, ha="right", fontsize=11)
plt.yticks(rotation=0, fontsize=11)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=9)
cbar.set_label("% of samples within TCGA project", fontsize=12)

plt.tight_layout()

output_path = os.path.join(
    OUTPUT_DIR,
    "tcga_project_vs_predicted_tissue_ordered.png"
)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {output_path}")
print(f"Saved matrix to: {matrix_path}")