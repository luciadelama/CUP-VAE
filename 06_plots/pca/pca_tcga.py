import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================
# PCA of TCGA expression data
# This script keeps only TCGA samples, preprocesses the expression
# matrix, runs PCA, and plots samples colored by tissue and shaped
# by tumour/normal sample type.
# ============================================================

# -----------------------------
# Input files
# -----------------------------
# The expression matrix has samples as rows and genes/features as columns.
# The metadata file contains dataset, tissue, and sample type information.
RAW_COUNTS_PATH = "../data/matrices/final_counts_filtered_transpose_collapsed_replicates.csv"
METADATA_PATH = "../data/metadata/final_sample_metadata_collapsed_replicates.csv"

# Load expression data and metadata using sample IDs as the dataframe index.
raw_counts = pd.read_csv(RAW_COUNTS_PATH, index_col=0)
metadata = pd.read_csv(METADATA_PATH, index_col=0)

# -----------------------------
# 1. Keep only TCGA samples
# -----------------------------

# Print metadata columns to verify that the expected columns are available.
print(metadata.columns)

# Select samples where the dataset label is TCGA.
# str.upper() makes the comparison robust to differences in capitalization.
tcga_metadata = metadata[metadata["dataset"].str.upper() == "TCGA"].copy()

# Keep only sample IDs that are present in both the counts matrix and metadata.
common_ids = raw_counts.index.intersection(tcga_metadata.index)

raw_counts_tcga = raw_counts.loc[common_ids]
metadata_tcga = tcga_metadata.loc[common_ids]

# Print basic information about the subset used for PCA.
print(f"TCGA samples used for PCA: {raw_counts_tcga.shape[0]}")
print(f"Number of genes/features: {raw_counts_tcga.shape[1]}")
print(f"Number of tissues: {metadata_tcga['tissue'].nunique()}")
print(metadata_tcga["sample_type"].value_counts())

# -----------------------------
# 2. Preprocess expression matrix
# -----------------------------

# Apply log1p transformation to reduce the effect of very large count values.
# log1p(x) is safe for count data because it can handle zero values.
X = np.log1p(raw_counts_tcga.values.astype("float32"))

# Standardize each gene/feature before PCA so that all genes are on a comparable scale.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype("float32")

# -----------------------------
# 3. Run PCA
# -----------------------------

# Keep the first two principal components for visualization.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Print PCA coordinates and explained variance for checking the result.
print(X_pca[:2])
print("Explained variance:", pca.explained_variance_ratio_)
print("Cumulative:", np.cumsum(pca.explained_variance_ratio_))

# -----------------------------
# 4. Build PCA dataframe
# -----------------------------

# Store PCA coordinates in a dataframe indexed by sample ID.
pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"],
    index=raw_counts_tcga.index
)

# Add metadata columns needed for plotting.
pca_df["tissue"] = metadata_tcga["tissue"]
pca_df["sample_type"] = metadata_tcga["sample_type"]

# Recode sample type names to make the plot legend cleaner.
pca_df["sample_group"] = pca_df["sample_type"].replace({
    "Primary Tumor": "Tumour",
    "Solid Tissue Normal": "Normal",
    "Tumor": "Tumour",
    "Normal": "Normal"
})

# Keep original sample type values for any samples that were not mapped above.
pca_df["sample_group"] = pca_df["sample_group"].fillna(pca_df["sample_type"])

print(pca_df["sample_group"].value_counts())

# -----------------------------
# 5. Plot PCA colored by tissue and shaped by tumour/normal status
# -----------------------------

n_tissues = pca_df["tissue"].nunique()
tissue_order = sorted(pca_df["tissue"].dropna().unique())

plt.figure(figsize=(12, 8))

# Color samples by tissue and use marker shape to show tumour/normal status.
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="tissue",
    hue_order=tissue_order,
    style="sample_group",
    markers={
        "Tumour": "o",
        "Normal": "^"
    },
    palette=sns.color_palette("tab20", n_colors=n_tissues),
    s=12,
    alpha=0.8,
    linewidth=0
)

# Include the percentage of variance explained by each PC in the axis labels.
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)")
plt.title("PCA of TCGA expression data colored by tissue and shaped by sample type")

# Place the legend outside the plot to avoid covering the points.
plt.legend(
    title="Tissue / Sample type",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=8,
    title_fontsize=10,
    markerscale=1.5
)

plt.tight_layout()
plt.savefig("pca_tcga_expression_by_tissue_and_sample_type.png", dpi=300, bbox_inches="tight")
plt.show()
