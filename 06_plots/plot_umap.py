import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import matplotlib as mpl

print("Loading data...")
df = pd.read_csv("latent_embeddings.csv")

sample_ids = df["sample_id"]
datasets = df["dataset"]
sample_type = df["sample_type"]   # Use sample type to separate TCGA tumor and TCGA normal samples.

# Use only latent embedding columns as UMAP input features.
X = df.iloc[:, df.columns.str.startswith("z")].values


# Create a new group variable for plotting.
group = []

for d, s in zip(datasets, sample_type):
    if d == "TCGA" and s == "Primary Tumor":
        group.append("TCGA Tumor")
    elif d == "TCGA" and s == "Solid Tissue Normal":
        group.append("TCGA Normal")
    else:
        group.append("GTEx")

group = pd.Series(group)


print("Running UMAP...")
# Configure UMAP for a two-dimensional projection of the latent space.
reducer = umap.UMAP(
    n_neighbors=50,
    min_dist=0.05,
    metric="cosine",
    random_state=42,
    n_jobs=-1
)

# Fit UMAP and transform the latent embeddings.
embedding = reducer.fit_transform(X)

print("Saving results...")
# Save the UMAP coordinates together with sample metadata.
umap_df = pd.DataFrame({
    "sample_id": sample_ids,
    "dataset": datasets,
    "sample_type": sample_type,
    "group": group,
    "UMAP1": embedding[:,0],
    "UMAP2": embedding[:,1]
})

umap_df.to_csv("umap_embeddings.csv", index=False)


print("Plotting...")
plt.figure(figsize=(8,6))

# Colors used for each group.
color_map = {
    "GTEx": "#1f77b4",        # azul
    "TCGA Tumor": "#2ca02c",  # verde
    "TCGA Normal": "#d62728"  # rojo
}

# 1. Plot GTEx samples in the background.
mask = group == "GTEx"
plt.scatter(
    embedding[mask,0],
    embedding[mask,1],
    c=color_map["GTEx"],
    s=2,
    alpha=0.2,
    label="GTEx"
)

# 2. Plot TCGA tumor samples with transparency.
mask = group == "TCGA Tumor"
plt.scatter(
    embedding[mask,0],
    embedding[mask,1],
    c=color_map["TCGA Tumor"],
    s=2,
    alpha=0.25,
    label="TCGA Tumor"
)

# 3. Plot TCGA normal samples last so they remain visible on top.
mask = group == "TCGA Normal"
plt.scatter(
    embedding[mask,0],
    embedding[mask,1],
    c=color_map["TCGA Normal"],
    s=4,
    alpha=0.95,
    label="TCGA Normal"
)

plt.legend(title="Dataset / Sample type")

plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("UMAP of VAE latent embeddings")

plt.tight_layout()
plt.savefig("umap_plot_dataset3.png", dpi=300, bbox_inches="tight")

print("Done!")