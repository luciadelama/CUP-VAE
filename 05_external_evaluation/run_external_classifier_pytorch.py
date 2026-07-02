#!/usr/bin/env python3

# Purpose: classify external samples directly from count features using a saved PyTorch elastic-net MLR model.

"""
Test external samples directly with a classifier trained on counts.

This script does NOT load or use any VAE.

Expected external counts format:
    - genes in rows
    - samples in columns
    - first gene column named either: geneid, GeneID, ensembl_gene_id, or Ensembl_ID

The script:
    1. Loads the external counts matrix.
    2. Maps external ENSEMBL IDs without version to the exact training IDs with version.
    3. Collapses duplicated genes if needed.
    4. Transposes to samples x genes.
    5. Aligns columns to the training feature order from feature_cols.json.
    6. Applies the same preprocessing used for training.
    7. Loads the trained counts classifier and predicts external samples.
"""

from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# =========================================================
# Configuration
# Paths and preprocessing settings for testing external samples
# with a classifier trained directly on count features.
# =========================================================

EXTERNAL_COUNTS_PATH = Path("/ngc/projects/gm_ext/lucdel/counts_UPT_Ensembl.tsv")
FEATURE_COLS_PATH = Path("feature_cols.json")

# Classifier trained directly on counts, NOT on VAE latents.
MODEL_PATH = Path("mlr_elastic_net_model.pt")

# Set to None if the counts classifier was trained without a scaler.
# Otherwise, use the scaler fitted during counts-classifier training.
SCALER_PATH = Path("mlr_elastic_net_scaler.joblib")
# SCALER_PATH = None

OUTPUT_DIR = Path("external_mlr_elastic_net_predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXTERNAL_PREDICTIONS_PATH = OUTPUT_DIR / "external_predictions_counts_classifier_only.csv"
ALIGNED_COUNTS_PATH = OUTPUT_DIR / "external_counts_aligned_for_classifier.csv"
EXTERNAL_TOP2_PREDICTIONS_PATH = OUTPUT_DIR / "top2_predictions.csv"

FILL_MISSING_GENES_WITH_ZERO = True

# IMPORTANT: these must match exactly what you used when training the counts classifier.
NORMALIZE_TO_40M = True
APPLY_LOG1P = True

# =========================================================
# Model definition
# This must match the architecture used when the classifier checkpoint was saved.
# =========================================================

class MultinomialLogisticRegression(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.linear(x)


# =========================================================
# Utility functions
# These helpers find the gene ID column, align external genes to the
# training feature list, load the saved classifier, and run prediction.
# =========================================================

def find_gene_column(counts_df: pd.DataFrame) -> str:
    """Find the gene ID column in the external counts file."""

    possible_gene_cols = [
        "geneid",
        "GeneID",
        "gene_id",
        "ensembl_gene_id",
        "Ensembl_ID",
        "ENSEMBL",
    ]

    for col in possible_gene_cols:
        if col in counts_df.columns:
            return col

    raise ValueError(
        "Could not find a gene ID column. Expected one of: "
        f"{possible_gene_cols}. Found columns: {counts_df.columns[:10].tolist()}"
    )


def add_training_versions_to_external_ensembl_ids(
    counts_df: pd.DataFrame,
    feature_cols: list[str],
    gene_col: str,
) -> pd.DataFrame:
    """
    Map external ENSEMBL IDs without version to the exact ENSEMBL IDs with version
    used during training.

    Example:
        ENSG00000141510 -> ENSG00000141510.18
    """

    training_id_map = {
        str(gene).split(".")[0]: str(gene)
        for gene in feature_cols
    }

    counts_df = counts_df.copy()
    counts_df[gene_col] = counts_df[gene_col].astype(str)
    counts_df["ensembl_gene_id_base"] = counts_df[gene_col].str.split(".").str[0]
    counts_df["ensembl_gene_id"] = counts_df["ensembl_gene_id_base"].map(training_id_map)

    unmapped = counts_df["ensembl_gene_id"].isna().sum()
    print("Number of external ENSEMBL IDs not found in training feature list:", unmapped)

    if unmapped > 0:
        print("First unmapped external ENSEMBL IDs:")
        print(
            counts_df.loc[
                counts_df["ensembl_gene_id"].isna(),
                "ensembl_gene_id_base",
            ].head(10).tolist()
        )

    counts_df = counts_df.dropna(subset=["ensembl_gene_id"])
    counts_df = counts_df.drop(columns=["ensembl_gene_id_base"])

    # Remove the original gene column if it is different from the standardized one
    if gene_col != "ensembl_gene_id" and gene_col in counts_df.columns:
        counts_df = counts_df.drop(columns=[gene_col])

    return counts_df


def load_and_align_external_counts(
    counts_path: Path,
    feature_cols: list[str],
    fill_missing_genes_with_zero: bool = True,
    normalize_to_40m: bool = True,
    apply_log1p: bool = True,
) -> pd.DataFrame:
    """
    Load external counts with genes as rows and samples as columns,
    align genes to the exact training feature order, and return samples x genes.
    """

    print("\nLoading external counts...")
    # Use whitespace separation because the external count file is stored as a TSV-like text file.
    # counts_df = pd.read_csv(counts_path, sep="\t")
    counts_df = pd.read_csv(counts_path, sep=r"\s+")

    print("Original external counts shape:", counts_df.shape)

    gene_col = find_gene_column(counts_df)
    print("Using gene column:", gene_col)

    counts_df = add_training_versions_to_external_ensembl_ids(
        counts_df=counts_df,
        feature_cols=feature_cols,
        gene_col=gene_col,
    )

    gene_ids = counts_df["ensembl_gene_id"]
    sample_counts = counts_df.drop(columns=["ensembl_gene_id"])

    # Ensure all sample count columns are numeric before normalization.
    sample_counts = sample_counts.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if normalize_to_40m:
        print("Normalizing each sample to library size 40M...")
        library_sizes = sample_counts.sum(axis=0)

        zero_library_samples = library_sizes[library_sizes == 0].index.tolist()
        if zero_library_samples:
            raise ValueError(
                "Some samples have library size 0, cannot normalize: "
                f"{zero_library_samples[:10]}"
            )

        sample_counts = sample_counts.div(library_sizes, axis=1) * 40_000_000

    counts_df = pd.concat([gene_ids, sample_counts], axis=1)

    # Collapse duplicated ENSEMBL IDs by summing their counts.
    counts_df = counts_df.groupby("ensembl_gene_id", as_index=False).sum(numeric_only=True)
    counts_df = counts_df.set_index("ensembl_gene_id")

    # Transpose the matrix so rows are samples and columns are genes.
    external_df = counts_df.T.reset_index()
    external_df = external_df.rename(columns={"index": "sample_id"})

    print("External counts after transpose:", external_df.shape)

    missing_features = [gene for gene in feature_cols if gene not in external_df.columns]

    if missing_features:
        if fill_missing_genes_with_zero:
            print(
                f"WARNING: {len(missing_features)} training genes are missing "
                "in external matrix. Filling missing genes with 0."
            )
            missing_df = pd.DataFrame(
                0.0,
                index=external_df.index,
                columns=missing_features,
            )
            external_df = pd.concat([external_df, missing_df], axis=1)
        else:
            raise ValueError(
                f"Missing {len(missing_features)} genes in external matrix. "
                f"First missing genes: {missing_features[:10]}"
            )

    extra_features = [
        col for col in external_df.columns
        if col not in feature_cols and col != "sample_id"
    ]

    if extra_features:
        print(f"WARNING: Found {len(extra_features)} extra genes. Ignoring them.")

    external_df = external_df[["sample_id"] + feature_cols]

    if apply_log1p:
        print("Applying log1p transformation...")
        external_df[feature_cols] = np.log1p(external_df[feature_cols].astype(np.float32))

    print("Final aligned external matrix shape:", external_df.shape)

    return external_df

def load_pytorch_mlr_model(model_path: Path, device: torch.device):
    print("\nLoading PyTorch MLR checkpoint...")
    print(model_path)

    checkpoint = torch.load(model_path, map_location=device)

    feature_cols = checkpoint["feature_cols"]
    label_classes = np.array(checkpoint["label_classes"])

    n_features = checkpoint["n_features"]
    n_classes = checkpoint["n_classes"]

    model = MultinomialLogisticRegression(
        n_features=n_features,
        n_classes=n_classes,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print("Loaded model successfully.")
    print("Number of features:", n_features)
    print("Number of classes:", n_classes)

    return model, feature_cols, label_classes, checkpoint


def predict_external(
    model,
    X_external: np.ndarray,
    label_classes: np.ndarray,
    device: torch.device,
):
    X_tensor = torch.tensor(X_external, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1)

    probs = probs.cpu().numpy()
    pred_idx = pred_idx.cpu().numpy()

    predicted_labels = label_classes[pred_idx]

    return predicted_labels, probs


# =========================================================
# Main
# =========================================================

def main() -> None:
    print(">> Testing external samples with saved PyTorch MLR elastic net model")
    print(">> No VAE will be loaded or used")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------------------------------------
    # Load model checkpoint
    # -----------------------------------------------------

    model, feature_cols, label_classes, checkpoint = load_pytorch_mlr_model(
        model_path=MODEL_PATH,
        device=device,
    )

    # -----------------------------------------------------
    # Load scaler
    # -----------------------------------------------------

    print("\nLoading scaler...")
    scaler = joblib.load(SCALER_PATH)
    print("Loaded scaler:", SCALER_PATH)

    # -----------------------------------------------------
    # Load, preprocess and align external counts
    # -----------------------------------------------------

    external_df = load_and_align_external_counts(
        counts_path=EXTERNAL_COUNTS_PATH,
        feature_cols=feature_cols,
        fill_missing_genes_with_zero=FILL_MISSING_GENES_WITH_ZERO,
        normalize_to_40m=NORMALIZE_TO_40M,
        apply_log1p=APPLY_LOG1P,
    )

    external_df.to_csv(ALIGNED_COUNTS_PATH, index=False)
    print("\nSaved aligned external counts:")
    print(ALIGNED_COUNTS_PATH)

    sample_ids = external_df["sample_id"].values
    X_external = external_df[feature_cols].values.astype(np.float32)

    print("X_external before scaling:", X_external.shape)

    # -----------------------------------------------------
    # Apply the StandardScaler fitted during classifier training.
    # The scaler must not be refitted on the external cohort.
    # -----------------------------------------------------

    print("\nApplying training StandardScaler...")
    X_external_scaled = scaler.transform(X_external).astype(np.float32)

    print("X_external after scaling:", X_external_scaled.shape)

    # -----------------------------------------------------
    # Predict external samples
    # -----------------------------------------------------

    print("\nPredicting external samples...")

    predicted_labels, probs = predict_external(
        model=model,
        X_external=X_external_scaled,
        label_classes=label_classes,
        device=device,
    )

    predictions_df = pd.DataFrame({
        "sample_id": sample_ids,
        "predicted_tissue": predicted_labels,
        "predicted_probability": probs.max(axis=1),
    })

    prob_df = pd.DataFrame(
        probs,
        columns=[f"prob_{label}" for label in label_classes],
    )

    predictions_df = pd.concat(
        [
            predictions_df.reset_index(drop=True),
            prob_df.reset_index(drop=True),
        ],
        axis=1,
    )

    predictions_df.to_csv(EXTERNAL_PREDICTIONS_PATH, index=False)

    print("\nExternal predicted tissue distribution:")
    print(pd.Series(predicted_labels).value_counts())

    print("\nSaved external predictions:")
    print(EXTERNAL_PREDICTIONS_PATH)

    # -----------------------------------------------------
    # Save compact top 1 / top 2 predictions for easier manual inspection.
    # -----------------------------------------------------

    if probs.shape[1] < 2:
        raise ValueError(
            "Model returned fewer than 2 classes, so top 2 predictions cannot be computed."
        )

    sorted_class_indices = np.argsort(probs, axis=1)[:, ::-1]

    top1_idx = sorted_class_indices[:, 0]
    top2_idx = sorted_class_indices[:, 1]

    top2_predictions_df = pd.DataFrame({
        "sample_id": sample_ids,
        "predicted_tissue_top1": label_classes[top1_idx],
        "probability_top1": probs[np.arange(probs.shape[0]), top1_idx],
        "predicted_tissue_top2": label_classes[top2_idx],
        "probability_top2": probs[np.arange(probs.shape[0]), top2_idx],
    })

    top2_predictions_df.to_csv(EXTERNAL_TOP2_PREDICTIONS_PATH, index=False)

    print("\nSaved compact top 1 / top 2 predictions:")
    print(EXTERNAL_TOP2_PREDICTIONS_PATH)

    print("\nDone.")


if __name__ == "__main__":
    main()