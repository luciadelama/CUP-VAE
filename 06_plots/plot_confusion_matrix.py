import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# =========================
# Input and output paths
# =========================
input_dir = "classifier_results"
output_dir = os.path.join(input_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

# =========================
# Load class labels used for all confusion matrices
# =========================
class_labels = pd.read_csv(
    os.path.join(input_dir, "class_labels.csv")
)["class_label"].values

# =========================
# Plot settings
# These settings make the matrix readable when many classes are shown.
# =========================
figsize = (18, 16)   # Try a larger size, such as (20, 18), if there are many classes.
dpi = 300
cmap = "Blues"

# Font sizes
title_size = 18
label_size = 14
tick_size = 8

models = ["raw", "vae", "pca"]

for name in models:
    print(f"Plotting confusion matrix for {name}...")

    # Load the confusion matrix for the current model.
    cm_path = os.path.join(input_dir, f"confusion_matrix_{name}.csv")
    cm_df = pd.read_csv(cm_path, index_col=0)
    cm = cm_df.values

    # Create one large figure per model.
    fig, ax = plt.subplots(figsize=figsize)

    # Use sklearn display helper to plot counts with the original class labels.
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(
        ax=ax,
        cmap=cmap,
        xticks_rotation=90,
        colorbar=True,
        values_format="d"
    )

    ax.set_title(f"Confusion Matrix - {name.upper()}", fontsize=title_size, pad=20)
    ax.set_xlabel("Predicted label", fontsize=label_size)
    ax.set_ylabel("True label", fontsize=label_size)

    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)

    plt.tight_layout()
    # Save the plot as a high-resolution PNG file.
    plt.savefig(
        os.path.join(output_dir, f"confusion_matrix_{name}_large_blue.png"),
        dpi=dpi,
        bbox_inches="tight"
    )
    plt.close()

print("Done.")