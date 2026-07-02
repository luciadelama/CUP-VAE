import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# ============================================================
# PCA embedding generation
# This script applies log transformation to the expression matrix,
# computes the first 50 principal components, and saves the PCA
# embeddings and explained variance values.
# ============================================================

# -----------------------------
# Load data
# -----------------------------
print("Loading data...")

# In PCA, samples should be rows and features should be columns.
# This file is already the transposed count matrix, with one row per sample.
raw_df = pd.read_csv("../data/matrices/final_counts_filtered_transpose.csv")

# Store sample IDs separately so they can be added back to the PCA output.
sample_ids = raw_df["sample_id"]

# Use only gene expression columns as PCA input features.
X = raw_df.drop(columns=["sample_id"]).values

# Apply log1p transformation to reduce skew from large count values.
# log1p is used because count data can include zeros.
X = np.log1p(X)
print("Shape of X:", X.shape)

# -----------------------------
# Run PCA
# -----------------------------
print("Running PCA...")

# Compute the first 50 principal components as a lower-dimensional embedding.
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

# Store explained variance ratio for each principal component.
explained = pca.explained_variance_ratio_
print("Explained variance per component:")
print(explained)

# Print the total variance captured by the first 50 components.
print("Total explained variance (50 PCs):", explained.sum())

# -----------------------------
# Save PCA embeddings
# -----------------------------

# Create column names PC1, PC2, ..., PC50.
pca_columns = [f"PC{i+1}" for i in range(50)]
pca_df = pd.DataFrame(X_pca, columns=pca_columns)

# Add sample IDs as the first column so rows can be matched back to samples.
pca_df.insert(0, "sample_id", sample_ids)

print("Saving PCA embeddings...")
pca_df.to_csv("pca_embeddings_50d.csv", index=False)

# Save explained variance and cumulative explained variance for interpretation.
var_df = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(50)],
    "explained_variance_ratio": explained,
    "cumulative_explained_variance": np.cumsum(explained)
})

var_df.to_csv("pca_explained_variance_50d.csv", index=False)
