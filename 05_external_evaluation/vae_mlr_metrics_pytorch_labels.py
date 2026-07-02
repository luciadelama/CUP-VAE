#!/usr/bin/env python3

# Purpose: generate VAE latent embeddings for external samples and classify them using a PyTorch linear classifier.

from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from vae_model import VariationalAutoencoder


# =========================================================
# Paths
# These paths point to the trained VAE, the PyTorch classifier,
# label names, external cohort input, and output files.
# =========================================================

# ----- VAE files -----
VAE_MODEL_PATH = Path("vae_model_best.pt")
VAE_CONFIG_PATH = Path("run_config.json")
VAE_FEATURE_COLS_PATH = Path("feature_cols.json")

# If your VAE was trained with a scaler before the encoder, set this to True.
# If not, keep it False.
VAE_SCALER_PATH = Path("vae_scaler.joblib")
USE_VAE_SCALER = False

# ----- PyTorch classifier trained on VAE latent embeddings -----
CLASSIFIER_MODEL_PATH = Path("vae_mlr_elastic_net_model.pt")

# IMPORTANT:
# We do NOT use the classifier scaler here because your scaler expects 35350 genes,
# while the classifier input here is 64 latent features.
USE_CLASSIFIER_SCALER = False
CLASSIFIER_SCALER_PATH = Path("vae_mlr_elastic_net_scaler.joblib")

# Label file
LABEL_CLASSES_PATH = Path("mlr_elastic_net_labels.json")

# ----- External cohort count matrix -----
EXTERNAL_COUNTS_PATH = Path("/ngc/projects/gm_ext/lucdel/counts_UPT_Ensembl.tsv")

# ----- Output files created by this script -----
OUTPUT_DIR = Path("external_test_vae_mlr_pytorch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXTERNAL_ALIGNED_COUNTS_PATH = OUTPUT_DIR / "external_aligned_counts.csv"
EXTERNAL_LATENT_PATH = OUTPUT_DIR / "external_latent_embeddings.csv"
EXTERNAL_PREDICTIONS_PATH = OUTPUT_DIR / "external_predictions_probabilities.csv"

FILL_MISSING_GENES_WITH_ZERO = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =========================================================
# Utility functions
# These helpers align external genes to the training genes, load the
# classifier checkpoint, and extract the linear classifier weights.
# =========================================================

def add_training_versions_to_external_ensembl_ids(
    counts_df: pd.DataFrame,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Maps external ENSEMBL IDs without version to the exact ENSEMBL IDs
    used during training.

    Example:
        ENSG00000141510 -> ENSG00000141510.18
    """

    training_id_map = {
        gene.split(".")[0]: gene
        for gene in feature_cols
    }

    counts_df["ensembl_gene_id_base"] = (
        counts_df["ensembl_gene_id"]
        .astype(str)
        .str.split(".")
        .str[0]
    )

    counts_df["ensembl_gene_id"] = counts_df["ensembl_gene_id_base"].map(training_id_map)

    unmapped = counts_df["ensembl_gene_id"].isna().sum()
    print("External ENSEMBL IDs not found in training features:", unmapped)

    if unmapped > 0:
        print("First unmapped IDs:")
        print(
            counts_df.loc[
                counts_df["ensembl_gene_id"].isna(),
                "ensembl_gene_id_base",
            ].head(10).tolist()
        )

    counts_df = counts_df.dropna(subset=["ensembl_gene_id"])
    counts_df = counts_df.drop(columns=["ensembl_gene_id_base"])

    return counts_df


def load_and_align_external_counts(
    counts_path: Path,
    feature_cols: list,
    fill_missing_genes_with_zero: bool = True,
) -> pd.DataFrame:
    """
    Loads external count matrix where genes are rows and samples are columns.
    Then:
    1. Maps ENSEMBL IDs to training versions.
    2. Normalizes by library size to 40M.
    3. Transposes to samples x genes.
    4. Aligns columns to training gene order.
    """

    print("\nLoading external counts...")
    counts_df = pd.read_csv(counts_path, sep=r"\s+")

    if "geneid" not in counts_df.columns:
        raise ValueError("External counts file must contain a 'geneid' column.")

    print("Original external counts shape:", counts_df.shape)

    counts_df = counts_df.rename(columns={"geneid": "ensembl_gene_id"})

    counts_df = add_training_versions_to_external_ensembl_ids(
        counts_df=counts_df,
        feature_cols=feature_cols,
    )

    gene_ids = counts_df["ensembl_gene_id"]
    sample_counts = counts_df.drop(columns=["ensembl_gene_id"])

    library_sizes = sample_counts.sum(axis=0)

    if (library_sizes == 0).any():
        bad_samples = library_sizes[library_sizes == 0].index.tolist()
        raise ValueError(f"These samples have library size 0: {bad_samples}")

    sample_counts = sample_counts.div(library_sizes, axis=1) * 40_000_000

    counts_df = pd.concat([gene_ids, sample_counts], axis=1)

    counts_df = (
        counts_df
        .groupby("ensembl_gene_id", as_index=False)
        .sum(numeric_only=True)
    )

    counts_df = counts_df.set_index("ensembl_gene_id")

    external_df = counts_df.T.reset_index()
    external_df = external_df.rename(columns={"index": "sample_id"})

    print("External counts after transpose:", external_df.shape)

    missing_features = [
        gene for gene in feature_cols
        if gene not in external_df.columns
    ]

    if missing_features:
        if fill_missing_genes_with_zero:
            print(f"WARNING: {len(missing_features)} training genes missing in external matrix.")
            print("Filling missing genes with 0.")

            missing_df = pd.DataFrame(
                0.0,
                index=external_df.index,
                columns=missing_features,
            )

            external_df = pd.concat([external_df, missing_df], axis=1)
        else:
            raise ValueError(
                f"Missing {len(missing_features)} genes. "
                f"First missing genes: {missing_features[:10]}"
            )

    extra_features = [
        c for c in external_df.columns
        if c not in feature_cols and c != "sample_id"
    ]

    if extra_features:
        print(f"WARNING: Found {len(extra_features)} extra genes. Ignoring them.")

    external_df = external_df[["sample_id"] + feature_cols]

    print("Final aligned external matrix shape:", external_df.shape)

    return external_df


class LinearMLRClassifier(nn.Module):
    """
    Simple PyTorch multinomial logistic regression:
    z1...zN -> class logits
    """

    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.linear(x)


def extract_state_dict(checkpoint):
    """
    Handles several common .pt formats.
    """

    if isinstance(checkpoint, dict):
        for key in [
            "model_state_dict",
            "state_dict",
            "classifier_state_dict",
            "clf_state_dict",
        ]:
            if key in checkpoint:
                return checkpoint[key]

        if all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint

    raise ValueError(
        "Could not find a valid state_dict inside classifier checkpoint. "
        "Run torch.load(...) and inspect checkpoint.keys()."
    )


def find_linear_weight_and_bias(state_dict):
    """
    Finds the main weight and bias in a linear classifier state_dict.
    """

    weight_key = None
    bias_key = None

    for k, v in state_dict.items():
        if torch.is_tensor(v) and v.ndim == 2 and k.endswith("weight"):
            weight_key = k
            break

    if weight_key is None:
        raise ValueError("Could not find a 2D weight tensor in classifier state_dict.")

    out_features, in_features = state_dict[weight_key].shape

    possible_bias_key = weight_key.replace("weight", "bias")

    if possible_bias_key in state_dict:
        bias_key = possible_bias_key
    else:
        for k, v in state_dict.items():
            if torch.is_tensor(v) and v.ndim == 1 and v.shape[0] == out_features:
                bias_key = k
                break

    if bias_key is None:
        raise ValueError("Could not find matching bias tensor in classifier state_dict.")

    return weight_key, bias_key, in_features, out_features


def load_pytorch_linear_classifier(model_path: Path, device):
    """
    Loads a PyTorch linear classifier even if the keys do not exactly match
    the class defined above.
    """

    print("\nLoading PyTorch classifier...")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)

    print("Classifier state_dict keys:")
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)}")

    weight_key, bias_key, n_features, n_classes = find_linear_weight_and_bias(state_dict)

    print("\nDetected classifier architecture:")
    print("Input latent features:", n_features)
    print("Output classes:", n_classes)
    print("Weight key:", weight_key)
    print("Bias key:", bias_key)

    clf = LinearMLRClassifier(
        n_features=n_features,
        n_classes=n_classes,
    ).to(device)

    remapped_state_dict = {
        "linear.weight": state_dict[weight_key],
        "linear.bias": state_dict[bias_key],
    }

    clf.load_state_dict(remapped_state_dict)
    clf.eval()

    return clf, n_features, n_classes


# =========================================================
# Main
# =========================================================

def main() -> None:
    print(f">> Using device: {DEVICE}")

    # -----------------------------------------------------
    # Load VAE configuration, gene feature columns, and label names.
    # These must match the training run used to create the VAE and classifier.
    # -----------------------------------------------------

    print("\nLoading VAE config, feature columns and labels...")

    with open(VAE_CONFIG_PATH, "r") as f:
        config = json.load(f)

    with open(VAE_FEATURE_COLS_PATH, "r") as f:
        feature_cols = json.load(f)

    with open(LABEL_CLASSES_PATH, "r") as f:
        class_labels = json.load(f)

    class_labels = [str(label) for label in class_labels]

    if "input_shape" in config:
        input_shape = torch.Size(config["input_shape"])
    elif "n_features" in config:
        input_shape = torch.Size([int(config["n_features"])])
    else:
        input_shape = torch.Size([len(feature_cols)])

    latent_features = int(config["latent_features"])

    fixed_log_sigma_x = float(config.get("fixed_log_sigma_x", 0.0))
    fixed_log_sigma_z = float(config.get("fixed_log_sigma_z", 0.0))

    print("Number of VAE input genes:", len(feature_cols))
    print("VAE latent features:", latent_features)
    print("Number of classifier labels:", len(class_labels))

    # -----------------------------------------------------
    # Load and align external counts to the same gene order used during VAE training.
    # -----------------------------------------------------

    external_df = load_and_align_external_counts(
        counts_path=EXTERNAL_COUNTS_PATH,
        feature_cols=feature_cols,
        fill_missing_genes_with_zero=FILL_MISSING_GENES_WITH_ZERO,
    )

    external_df.to_csv(EXTERNAL_ALIGNED_COUNTS_PATH, index=False)
    print("Saved aligned external counts:", EXTERNAL_ALIGNED_COUNTS_PATH)

    print("Number of external samples:", external_df.shape[0])

    X = external_df[feature_cols].values.astype(np.float32)

    X = np.log1p(X)

    if USE_VAE_SCALER:
        if not VAE_SCALER_PATH.exists():
            raise FileNotFoundError(
                f"USE_VAE_SCALER=True but file does not exist: {VAE_SCALER_PATH}"
            )

        print("\nLoading VAE scaler:", VAE_SCALER_PATH)
        vae_scaler = joblib.load(VAE_SCALER_PATH)

        X = vae_scaler.transform(X).astype(np.float32)
        print("Applied VAE scaler.")

    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    print("X external shape:", X.shape)

    # -----------------------------------------------------
    # Load VAE model
    # -----------------------------------------------------

    print("\nLoading VAE model...")

    vae = VariationalAutoencoder(
        input_shape=input_shape,
        latent_features=latent_features,
        fixed_log_sigma_x=fixed_log_sigma_x,
        fixed_log_sigma_z=fixed_log_sigma_z,
    ).to(DEVICE)

    vae_checkpoint = torch.load(VAE_MODEL_PATH, map_location=DEVICE)

    if isinstance(vae_checkpoint, dict) and "model_state_dict" in vae_checkpoint:
        vae.load_state_dict(vae_checkpoint["model_state_dict"])
        print("Loaded VAE from checkpoint['model_state_dict']")
        print("VAE epoch:", vae_checkpoint.get("epoch"))
        print("VAE best val ELBO:", vae_checkpoint.get("best_val_elbo"))
    else:
        vae.load_state_dict(vae_checkpoint)
        print("Loaded VAE directly from state_dict")

    vae.eval()

    # -----------------------------------------------------
    # Generate latent embeddings
    # -----------------------------------------------------

    print("\nGenerating external latent embeddings...")

    with torch.no_grad():
        qz = vae.posterior(X_tensor)
        mu = qz.mu.cpu().numpy()

    latent_df = external_df[["sample_id"]].copy()
    latent_df["split"] = "external"

    for i in range(mu.shape[1]):
        latent_df[f"z{i + 1}"] = mu[:, i]

    latent_df.to_csv(EXTERNAL_LATENT_PATH, index=False)

    print("Saved latent embeddings:", EXTERNAL_LATENT_PATH)
    print("Latent dataframe shape:", latent_df.shape)

    # -----------------------------------------------------
    # Prepare latent embeddings as classifier input.
    # Each row is one external sample and each column is one latent feature.
    # -----------------------------------------------------

    print("\nPreparing latent embeddings for classifier...")

    latent_feature_cols = [f"z{i + 1}" for i in range(mu.shape[1])]
    X_latent_external = latent_df[latent_feature_cols].values.astype(np.float32)

    if USE_CLASSIFIER_SCALER:
        if not CLASSIFIER_SCALER_PATH.exists():
            raise FileNotFoundError(
                f"USE_CLASSIFIER_SCALER=True but file does not exist: {CLASSIFIER_SCALER_PATH}"
            )

        print("Loading classifier scaler:", CLASSIFIER_SCALER_PATH)
        classifier_scaler = joblib.load(CLASSIFIER_SCALER_PATH)
        X_latent_external = classifier_scaler.transform(X_latent_external).astype(np.float32)
        print("Applied classifier scaler.")
    else:
        print("No classifier scaler applied.")

    print("Latent external shape used for classifier:", X_latent_external.shape)

    # -----------------------------------------------------
    # Load PyTorch classifier
    # -----------------------------------------------------

    clf, clf_n_features, clf_n_classes = load_pytorch_linear_classifier(
        model_path=CLASSIFIER_MODEL_PATH,
        device=DEVICE,
    )

    if X_latent_external.shape[1] != clf_n_features:
        raise ValueError(
            f"Classifier expects {clf_n_features} features, "
            f"but external latent matrix has {X_latent_external.shape[1]}."
        )

    if len(class_labels) != clf_n_classes:
        raise ValueError(
            f"Label file has {len(class_labels)} labels, "
            f"but classifier outputs {clf_n_classes} classes."
        )

    # -----------------------------------------------------
    # Predict external samples
    # -----------------------------------------------------

    print("\nPredicting external samples...")

    X_latent_tensor = torch.tensor(
        X_latent_external,
        dtype=torch.float32,
    ).to(DEVICE)

    with torch.no_grad():
        logits = clf(X_latent_tensor)
        y_prob_external = torch.softmax(logits, dim=1).cpu().numpy()

    y_pred_idx = np.argmax(y_prob_external, axis=1)

    y_pred_external = np.array([
        class_labels[i] for i in y_pred_idx
    ])

    print("\nPredicted tissue distribution:")
    print(pd.Series(y_pred_external).value_counts())

    # -----------------------------------------------------
    # Save predictions and probabilities
    # -----------------------------------------------------

    prob_df = pd.DataFrame(
        y_prob_external,
        columns=[f"prob_{label}" for label in class_labels],
    )

    predictions_df = pd.DataFrame({
        "sample_id": latent_df["sample_id"].values,
        "predicted_class_idx": y_pred_idx,
        "predicted_tissue": y_pred_external,
    })

    predictions_df = pd.concat(
        [
            predictions_df.reset_index(drop=True),
            prob_df.reset_index(drop=True),
        ],
        axis=1,
    )

    top_indices = np.argsort(y_prob_external, axis=1)[:, ::-1]

    predictions_df["top1_class_idx"] = top_indices[:, 0]
    predictions_df["top1_tissue"] = [
        class_labels[i] for i in top_indices[:, 0]
    ]
    predictions_df["top1_probability"] = [
        y_prob_external[row_idx, class_idx]
        for row_idx, class_idx in enumerate(top_indices[:, 0])
    ]

    predictions_df["top2_class_idx"] = top_indices[:, 1]
    predictions_df["top2_tissue"] = [
        class_labels[i] for i in top_indices[:, 1]
    ]
    predictions_df["top2_probability"] = [
        y_prob_external[row_idx, class_idx]
        for row_idx, class_idx in enumerate(top_indices[:, 1])
    ]

    predictions_df.to_csv(EXTERNAL_PREDICTIONS_PATH, index=False)

    print("\nSaved external predictions:")
    print(EXTERNAL_PREDICTIONS_PATH)

    print("\nDone.")


if __name__ == "__main__":
    main()