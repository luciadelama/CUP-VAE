import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================
# PCA of GTEx expression data
# This script keeps only GTEx samples, applies log transformation
# and standard scaling, then plots the first two principal components
# colored by tissue type.
# ============================================================

# -----------------------------
# Input files
# -----------------------------
# The expression matrix has samples as rows and genes/features as columns.
# The metadata file contains sample information such as dataset and tissue.
RAW_COUNTS_PATH = "../data/matrices/final_counts_filtered_transpose_collapsed_replicates.csv"
METADATA_PATH = "../data/metadata/final_sample_metadata_collapsed_replicates.csv"

# Load expression data and metadata using sample IDs as the dataframe index.
raw_counts = pd.read_csv(RAW_COUNTS_PATH, index_col=0)
metadata = pd.read_csv(METADATA_PATH, index_col=0)

# -----------------------------
# 1. Keep only GTEx samples
# -----------------------------

# Print metadata columns to verify that the expected columns are available.
print(metadata.columns)

# Select samples where the dataset label is GTEx.
# str.upper() makes the comparison robust to differences in capitalization.
gtex_metadata = metadata[metadata["dataset"].str.upper() == "GTEX"].copy()

# Keep only sample IDs that are present in both the counts matrix and metadata.
# This prevents alignment errors when plotting or adding metadata columns.
common_ids = raw_counts.index.intersection(gtex_metadata.index)

raw_counts_gtex = raw_counts.loc[common_ids]
metadata_gtex = gtex_metadata.loc[common_ids]

# Print basic information about the subset used for PCA.
print(f"GTEx samples used for PCA: {raw_counts_gtex.shape[0]}")
print(f"Number of genes/features: {raw_counts_gtex.shape[1]}")
print(f"Number of tissues: {metadata_gtex['tissue'].nunique()}")
print(metadata_gtex["tissue"].value_counts())

# -----------------------------
# 2. Preprocess expression matrix
# -----------------------------

# Apply log1p transformation to reduce the effect of very large count values.
# log1p(x) is used instead of log(x) because gene counts can contain zeros.
X = np.log1p(raw_counts_gtex.values.astype("float32"))

# Standardize each gene/feature to mean 0 and variance 1 before PCA.
# This gives each gene a comparable contribution to the PCA.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype("float32")

# -----------------------------
# 3. Run PCA
# -----------------------------

# Keep only the first two principal components for visualization.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Print the first two PCA coordinates and the explained variance.
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
    index=raw_counts_gtex.index
)

# Add tissue labels from the metadata so they can be used for coloring.
pca_df["tissue"] = metadata_gtex["tissue"]

# Remove samples without tissue annotation, just in case.
pca_df = pca_df.dropna(subset=["tissue"])

# Define a stable tissue order for the legend and colors.
n_tissues = pca_df["tissue"].nunique()
tissue_order = sorted(pca_df["tissue"].unique())

# -----------------------------
# 5. Plot PCA colored by tissue
# -----------------------------

plt.figure(figsize=(11, 8))

# Draw one point per sample and color points according to tissue type.
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="tissue",
    hue_order=tissue_order,
    palette=sns.color_palette("tab20", n_colors=n_tissues),
    s=8,
    alpha=0.8,
    linewidth=0
)

# Include the percentage of variance explained by each PC in the axis labels.
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)")
plt.title("PCA of GTEx expression data colored by tissue")

# Place the legend outside the plot to avoid covering the points.
plt.legend(
    title="Tissue",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=8,
    title_fontsize=10,
    markerscale=1.5
)

plt.tight_layout()
plt.savefig("pca_gtex_expression_by_tissue.png", dpi=300, bbox_inches="tight")
plt.show()
