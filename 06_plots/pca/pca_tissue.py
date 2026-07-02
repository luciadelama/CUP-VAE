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
# PCA of all input expression data colored by tissue
# This script preprocesses the full expression matrix, runs PCA,
# and visualizes the first two principal components by tissue type.
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
# log1p is suitable for count data because it handles zero counts.
X = np.log1p(raw_counts.values.astype("float32"))

# Standardize each gene/feature before PCA.
# This prevents genes with larger numeric ranges from dominating the PCA.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype("float32")

# -----------------------------
# 2. Run PCA
# -----------------------------

# Keep only the first two principal components for a 2D visualization.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(X_pca[:2])

# Print the variance explained by each component and the cumulative variance.
print("Explained variance:", pca.explained_variance_ratio_)
print("Cumulative:", np.cumsum(pca.explained_variance_ratio_))

# -----------------------------
# 3. Build PCA dataframe
# -----------------------------

# Store PCA coordinates in a dataframe indexed by sample ID.
pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"],
    index=raw_counts.index
)

# Add tissue labels from metadata for coloring the PCA plot.
pca_df["tissue"] = metadata["tissue"]

# Define the number and order of tissue categories for consistent plotting.
n_tissues = pca_df["tissue"].nunique()
tissue_order = sorted(pca_df["tissue"].dropna().unique())

# -----------------------------
# 4. Plot PCA colored by tissue
# -----------------------------

plt.figure(figsize=(11, 8))

# Each point is one sample, colored according to tissue type.
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="tissue",
    hue_order=tissue_order,
    palette=sns.color_palette("tab20", n_colors=n_tissues),
    s=6,
    alpha=0.8,
    linewidth=0
)

# Add explained variance percentages to the axis labels.
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)")
plt.title("PCA of input expression data colored by tissue")

# Place the legend outside the plot because there are many tissue classes.
plt.legend(
    title="Tissue",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=8,
    title_fontsize=10
)

plt.tight_layout()
plt.savefig("pca_input_expression_by_tissue.png", dpi=300, bbox_inches="tight")
plt.show()
