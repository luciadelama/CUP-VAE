import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# PCA of all input expression data colored by sample group
# This script runs PCA on the full expression matrix, creates broad
# sample groups, saves PCA coordinates, and plots PC1 vs PC2.
# ============================================================

# -----------------------------
# Input files
# -----------------------------
# The expression matrix has sample IDs as rows and gene features as columns.
RAW_COUNTS_PATH = "../data/matrices/final_counts_filtered_transpose_collapsed_replicates.csv" # sample_id + gene features
METADATA_PATH = "../data/metadata/final_sample_metadata_collapsed_replicates.csv"

# Load expression data and metadata using sample IDs as the dataframe index.
raw_counts = pd.read_csv(RAW_COUNTS_PATH, index_col=0)
metadata = pd.read_csv(METADATA_PATH, index_col=0)

# -----------------------------
# 1. Preprocess expression matrix
# -----------------------------

# Apply log1p transformation to reduce skew from very large count values.
# log1p is safe for count data because it handles zero counts.
X = np.log1p(raw_counts.values.astype("float32"))

# Standardize each gene/feature before PCA so features are comparable.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype("float32")

# -----------------------------
# 2. Run PCA
# -----------------------------

# Keep only the first two principal components for visualization.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(X_pca[:2])

# Print explained variance for interpretation of PC1 and PC2.
print("Explained variance:", pca.explained_variance_ratio_)
print("Cumulative:", np.cumsum(pca.explained_variance_ratio_))

# -----------------------------
# 3. Build PCA dataframe and sample groups
# -----------------------------

# Store PCA coordinates in a dataframe indexed by sample ID.
pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"],
    index=raw_counts.index
)

# Reorder metadata so that it matches the PCA dataframe exactly.
metadata = metadata.loc[pca_df.index]

# Add dataset and sample type information for grouping.
pca_df["dataset"] = metadata["dataset"]
pca_df["sample_type"] = metadata["sample_type"]

# Create three broad sample groups for the plot:
# GTEx normal, TCGA normal, and TCGA tumor.
pca_df["group"] = np.select(
    [
        (pca_df["dataset"] == "GTEx"),
        (pca_df["dataset"] == "TCGA") & (pca_df["sample_type"] == "Solid Tissue Normal"),
        (pca_df["dataset"] == "TCGA") & (pca_df["sample_type"] == "Primary Tumor"),
    ],
    [
        "GTEx normal",
        "TCGA normal",
        "TCGA tumor",
    ],
    default="Other"
)

print(pca_df["group"].value_counts())

# Save PCA dataframe so that other scripts can reuse it without recomputing PCA.
pca_df.to_csv("pca_input_expression_df.csv", index=True)

# Save explained variance separately for later use in plot labels.
explained_variance_df = pd.DataFrame({
    "PC": ["PC1", "PC2"],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "explained_variance_percent": pca.explained_variance_ratio_ * 100
})

explained_variance_df.to_csv("pca_input_expression_explained_variance.csv", index=False)

# -----------------------------
# 4. Plot PCA colored by sample group
# -----------------------------

plt.figure(figsize=(11, 8))

# Fixed colors are used so the same sample groups always have the same colors.
palette = {
    "GTEx normal": "#1f77b4",
    "TCGA normal": "#ff7f0e",
    "TCGA tumor": "#d62728",
}

# Plot larger background groups first so that TCGA normal samples can be drawn on top.
for group in ["GTEx normal", "TCGA tumor"]:
    subset = pca_df[pca_df["group"] == group]
    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=group,
        color=palette[group],
        s=35,
        alpha=0.6,
        linewidths=0
    )

# Plot TCGA normal last so these points remain visible if they overlap other groups.
subset = pca_df[pca_df["group"] == "TCGA normal"]
plt.scatter(
    subset["PC1"],
    subset["PC2"],
    label="TCGA normal",
    color=palette["TCGA normal"],
    s=45,
    alpha=0.95,
    linewidths=0
)

# Add explained variance percentages to the axis labels.
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)")
plt.title("PCA of input expression data colored by sample group")

# Place the legend outside the plot to avoid covering samples.
plt.legend(
    title="Sample group",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=9,
    title_fontsize=10
)

plt.tight_layout()
plt.savefig("pca_input_expression_by_sample_group.png", dpi=300, bbox_inches="tight")
plt.show()
