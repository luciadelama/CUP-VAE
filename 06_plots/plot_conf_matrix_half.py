import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import PowerNorm

# -----------------------------
# Input file
# -----------------------------
csv_path = "mlr_elastic_net_test_confusion_matrix.csv"

cm_counts = pd.read_csv(csv_path, index_col=0)

# -----------------------------
# Normalize each row by the true class total
# -----------------------------
# Each row sums to 100%, so values show the distribution of predictions within each true class.
cm_percent = cm_counts.div(cm_counts.sum(axis=1), axis=0) * 100

# -----------------------------
# Create selective annotations for the heatmap
# -----------------------------
annot = cm_percent.copy().astype(str)

for i in range(cm_percent.shape[0]):
    for j in range(cm_percent.shape[1]):
        value = cm_percent.iloc[i, j]

        # Show diagonal and non-zero errors only
        if i == j or value > 0:
            if np.isclose(value, round(value)):
                annot.iloc[i, j] = f"{int(round(value))}"
            else:
                annot.iloc[i, j] = f"{value:.1f}"
        else:
            annot.iloc[i, j] = ""

# -----------------------------
# Plot normalized confusion matrix
# -----------------------------
# Use a dynamic figure size so the plot remains readable for many labels.
n_labels = cm_percent.shape[0]

plt.figure(figsize=(max(16, n_labels * 0.55), max(8, n_labels * 0.32)))

# Use PowerNorm to make both small and large percentages easier to see.
ax = sns.heatmap(
    cm_percent,
    annot=annot,
    fmt="",
    cmap="Blues",
    norm=PowerNorm(gamma=0.5, vmin=0, vmax=100),
    linewidths=0.2,
    linecolor="lightgrey",
    cbar_kws={"label": "% of samples in true class"}
)

plt.xlabel("Predicted label", fontsize=14)
plt.ylabel("True label", fontsize=14)
plt.title("Elastic Net MLR Confusion Matrix", fontsize=16, pad=12)

plt.xticks(rotation=45, ha="right",fontsize=11)
plt.yticks(rotation=0, fontsize=11)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=9)
cbar.set_label("% of samples in true class", fontsize=12)

plt.tight_layout()
# Save the figure as a high-resolution PNG file.
plt.savefig("confusion_matrix_percent.png", dpi=300, bbox_inches="tight")