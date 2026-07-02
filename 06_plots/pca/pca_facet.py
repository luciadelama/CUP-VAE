import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math

# ============================================================
# Faceted PCA plot by tissue
# This script uses previously saved PCA coordinates and creates one
# small PCA panel per tissue. In each panel, all samples are shown in
# grey as background, while the selected tissue is highlighted by group.
# ============================================================

# Load saved PCA dataframe produced by the sample-group PCA script.
pca_df = pd.read_csv("pca_input_expression_df.csv", index_col=0)

# Load explained variance so the axis labels can show PC1/PC2 percentages.
explained_variance_df = pd.read_csv("pca_input_expression_explained_variance.csv")

# Extract the explained variance percentage for PC1.
pc1_var = explained_variance_df.loc[
    explained_variance_df["PC"] == "PC1",
    "explained_variance_percent"
].iloc[0]

# Extract the explained variance percentage for PC2.
pc2_var = explained_variance_df.loc[
    explained_variance_df["PC"] == "PC2",
    "explained_variance_percent"
].iloc[0]

# Fixed colors for the three sample groups.
palette = {
    "GTEx normal": "#1f77b4",
    "TCGA normal": "#ff7f0e",
    "TCGA tumor": "#d62728",
}

# Define a fixed group order so the legend and plotting order are consistent.
group_order = ["GTEx normal", "TCGA normal", "TCGA tumor"]

# List all tissues in alphabetical order and ignore missing tissue labels.
tissues = sorted(pca_df["tissue"].dropna().unique())

# -----------------------------
# Facet layout
# -----------------------------

# Use seven columns and calculate the number of rows needed for all tissues.
n_cols = 7
n_rows = math.ceil(len(tissues) / n_cols)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(n_cols * 2.2, n_rows * 2.0),
    sharex=True,
    sharey=True
)

# Flatten axes so they can be looped over easily.
axes = axes.flatten()

# Use global axis limits so all tissue panels are directly comparable.
x_min, x_max = pca_df["PC1"].min(), pca_df["PC1"].max()
y_min, y_max = pca_df["PC2"].min(), pca_df["PC2"].max()

# Add a small margin around the global limits.
x_pad = (x_max - x_min) * 0.03
y_pad = (y_max - y_min) * 0.03

# -----------------------------
# Plot one panel per tissue
# -----------------------------

for ax, tissue in zip(axes, tissues):
    tissue_df = pca_df[pca_df["tissue"] == tissue]

    # Plot all samples in grey as background context.
    ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        color="lightgrey",
        s=2,
        alpha=0.25,
        linewidths=0
    )

    # Highlight only the samples from the current tissue, colored by sample group.
    for group in group_order:
        subset = tissue_df[tissue_df["group"] == group]

        if subset.empty:
            continue

        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            color=palette[group],
            s=8,
            alpha=0.85,
            linewidths=0
        )

    # Show the tissue name and the number of samples in that tissue.
    ax.set_title(f"{tissue} (n={len(tissue_df)})", fontsize=7)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.tick_params(axis="both", labelsize=6)
    ax.grid(True, linewidth=0.3, alpha=0.3)

# Remove unused empty panels if the grid has more axes than tissues.
for ax in axes[len(tissues):]:
    ax.axis("off")

# Shared axis labels for the whole figure.
fig.supxlabel(f"PC1 ({pc1_var:.2f}% variance)", fontsize=11)
fig.supylabel(f"PC2 ({pc2_var:.2f}% variance)", fontsize=11)

# Create one shared legend instead of repeating legends in every panel.
legend_handles = [
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        label=group,
        markerfacecolor=palette[group],
        markersize=6
    )
    for group in group_order
]

fig.legend(
    handles=legend_handles,
    title="Sample group",
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    fontsize=9,
    title_fontsize=10,
    frameon=False
)

plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig(
    "pca_faceted_by_tissue_colored_by_sample_group.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
