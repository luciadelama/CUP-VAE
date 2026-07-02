#!/usr/bin/env python3

"""
Generate latent embeddings from a saved best VAE checkpoint.

The script reloads the model configuration, rebuilds the VAE architecture,
loads the trained weights, and exports the posterior mean q(z|x) as the latent
embedding for each sample.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from vae_model import VariationalAutoencoder

# =========================================================
# Configuration
# =========================================================
VAE_OUTPUT_DIR = Path("../results/vae_outputs_755995")
DATA_DIR = Path("../data")

MODEL_PATH = VAE_OUTPUT_DIR / "vae_model_best.pt"
CONFIG_PATH = VAE_OUTPUT_DIR / "run_config.json"
SPLITS_PATH = DATA_DIR / "splits/data_splits.csv"

OUTPUT_LATENT_PATH = VAE_OUTPUT_DIR / "latent_embeddings_best.csv"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Main execution function. Keeping the workflow inside main avoids running it on import.
def main() -> None:
    print(f">> Using device: {DEVICE}")

    # -----------------------------------------------------
    # Load the saved training configuration
    # -----------------------------------------------------
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    counts_path = config["counts_path"]
    metadata_path = config["metadata_path"]

    feature_cols = config.get("feature_cols", None)
    input_shape_from_config = config.get("input_shape", None)
    latent_features = int(config["latent_features"])

    fixed_log_sigma_x = float(config.get("fixed_log_sigma_x", 0.0))
    fixed_log_sigma_z = float(config.get("fixed_log_sigma_z", 0.0))

    print("Latent features:", latent_features)
    # print("Input shape:", input_shape)
    # print("Number of feature cols:", len(feature_cols))

    # -----------------------------------------------------
    # Load input count matrix and metadata
    # -----------------------------------------------------
    counts_df = pd.read_csv(counts_path)
    metadata_df = pd.read_csv(metadata_path)
    splits_df = pd.read_csv(SPLITS_PATH)

    if "sample_id" not in counts_df.columns:
        raise ValueError("counts file must contain sample_id")

    if "sample_id" not in metadata_df.columns:
        raise ValueError("metadata file must contain sample_id")

    if "tissue" not in metadata_df.columns:
        raise ValueError("metadata file must contain tissue")

    if "sample_id" not in splits_df.columns or "split" not in splits_df.columns:
        raise ValueError("splits file must contain sample_id and split")

    # Merge metadata with the count matrix using sample IDs
    final_df = metadata_df.merge(counts_df, on="sample_id", how="inner")

    # Add predefined train/validation/test split labels
    final_df = final_df.merge(
        splits_df[["sample_id", "split"]],
        on="sample_id",
        how="inner",
    )

    if feature_cols is None:
        feature_cols = [
            c for c in final_df.columns
            if c not in metadata_df.columns and c != "split"
        ]

        print("WARNING: feature_cols not found in config.")
        print("Reconstructed feature_cols from counts columns.")
        print("Only safe if counts file is exactly the same as during training.")

    print("Merged shape:", final_df.shape)
    print(final_df["split"].value_counts())
    print("Number of feature cols:", len(feature_cols))


    # -----------------------------------------------------
    # Check that all training features are present in the current data
    # -----------------------------------------------------
    missing_features = [c for c in feature_cols if c not in final_df.columns]

    if missing_features:
        raise ValueError(
            f"Missing {len(missing_features)} features in counts data. "
            f"First missing features: {missing_features[:10]}"
        )

    # Keep the same feature order that was used during training
    X = final_df[feature_cols].values.astype(np.float32)
    X = np.log1p(X)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    if input_shape_from_config is None:
        input_shape = X_tensor[0].shape
        print("WARNING: input_shape not found in config.")
        print("Reconstructed input_shape from X_tensor:", input_shape)
    else:
        input_shape = torch.Size(input_shape_from_config)

    print("X shape:", X.shape)

    # -----------------------------------------------------
    # Rebuild the VAE and load the saved best checkpoint
    # -----------------------------------------------------
    vae = VariationalAutoencoder(
        input_shape=input_shape,
        latent_features=latent_features,
        fixed_log_sigma_x=fixed_log_sigma_x,
        fixed_log_sigma_z=fixed_log_sigma_z,
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()

    print("Loaded best model from:", MODEL_PATH)
    print("Best model epoch:", checkpoint.get("epoch"))
    print("Best val ELBO:", checkpoint.get("best_val_elbo"))

    # -----------------------------------------------------
    # Generate latent embeddings using the posterior mean
    # -----------------------------------------------------
    with torch.no_grad():
        qz = vae.posterior(X_tensor)
        mu = qz.mu.cpu().numpy()

    latent_df = final_df[["sample_id", "tissue", "split"]].copy()

    for i in range(mu.shape[1]):
        latent_df[f"z{i + 1}"] = mu[:, i]

    latent_df.to_csv(OUTPUT_LATENT_PATH, index=False)

    print("\nSaved latent embeddings from BEST model:")
    print(OUTPUT_LATENT_PATH)
    print("Latent dataframe shape:", latent_df.shape)


if __name__ == "__main__":
    main()