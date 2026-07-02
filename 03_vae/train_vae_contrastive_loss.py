#!/usr/bin/env python3

"""
Train a Variational Autoencoder (VAE) with supervised contrastive loss.

The script loads gene count data and metadata, builds train/validation/test
splits, trains a VAE, saves checkpoints and training curves, and exports latent
embeddings from the latest and best checkpoints.
"""

import os
from pathlib import Path
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, Subset

from vae_model_contrastive_loss import VariationalAutoencoder, VariationalInference, make_vae_plots


# =========================================================
# Configuration
# =========================================================
COUNTS_PATH = "../data/matrices/final_counts_filtered_transpose_collapsed_replicates.csv" # Input matrix is expected to have samples as rows and genes as columns
METADATA_PATH = "../data/metadata/final_sample_metadata_collapsed_replicates_with_contrastive_labels.csv"

LABEL_COL = "contrastive_label"  # Metadata column used as the class label for contrastive loss

JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
OUTPUT_DIR = f"vae_outputs_{JOB_ID}"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

BATCH_SIZE = 256
LATENT_FEATURES = 64
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0
NUM_EPOCHS = 1000

BETA = 1.0
LAMBDA_CONTRASTIVE = 1000  # Weight applied to the supervised contrastive loss
CONTRASTIVE_TEMPERATURE = 0.1  # Controls how strongly similarities are scaled in contrastive loss

FIXED_LOG_SIGMA_X = 0.0
FIXED_LOG_SIGMA_Z = 0.0
RANDOM_STATE = 42

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =========================================================
# Main
# =========================================================
# Main execution function. Keeping the workflow inside main avoids running it on import.
def main() -> None:
    print(f">> Using device: {DEVICE}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "vae_model.pt"
    best_model_path = output_dir / "vae_model_best.pt"
    best_total_loss_path = output_dir / "vae_model_best_total_loss.pt"
    history_path = output_dir / "training_history.csv"
    split_path = output_dir / "data_splits.csv"
    latent_path = output_dir / "latent_embeddings_last.csv"
    config_path = output_dir / "run_config.json"
    plot_path = output_dir / "training_curves.png"

    latent_epochs_dir = output_dir / "latent_per_epoch"
    latent_epochs_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Load input count matrix and metadata
    # -----------------------------------------------------
    counts_df = pd.read_csv(COUNTS_PATH)
    metadata_df = pd.read_csv(METADATA_PATH)

    #if "gene_id" not in counts_df.columns:
    #   raise ValueError("counts.csv must contain a 'gene_id' column")
    if "sample_id" not in metadata_df.columns:
        raise ValueError("metadata.csv must contain a 'sample_id' column")
    
    required_metadata_cols = {
        "sample_id",
        "tissue",
        "sample_type",
        LABEL_COL,
    }

    missing_cols = required_metadata_cols - set(metadata_df.columns)

    if missing_cols:
        raise ValueError(
            f"Missing required metadata columns: {sorted(missing_cols)}"
        )

    # counts_only = counts_df.drop(columns=["gene_id"])

    # Merge directly, since counts already has samples as rows
    final_df = pd.merge(metadata_df, counts_df, on="sample_id")

    print("Merged shape:", final_df.shape)
    print("Number of samples:", final_df.shape[0])
    print("Number of genes:", final_df.shape[1] - metadata_df.shape[1])

    feature_cols = [c for c in final_df.columns if c not in metadata_df.columns]

    X = final_df[feature_cols].values.astype(np.float32)
    # X = np.log1p(X) # Log-transform counts for better numerical stability (optional, depending on the likelihood used in the VAE)

    y = final_df[LABEL_COL].values

    label_names = sorted(pd.unique(y))
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    y_encoded = np.array([label_to_idx[label] for label in y], dtype=np.int64)

    print("Number of tissue classes:", len(label_names))
    print("Label mapping:")
    for label, idx in label_to_idx.items():
        print(f"  {idx}: {label}")

    print("Final X shape:", X.shape)

    # -----------------------------------------------------
    # Train / Val / Test split (Load predefined splits if they exist)
    # -----------------------------------------------------
    SPLIT_PATH = Path("data/splits/data_splits.csv")

    # Reuse predefined splits when available; otherwise create stratified splits.
    if SPLIT_PATH.exists():
        split_df = pd.read_csv(SPLIT_PATH)
        train_idx = split_df[split_df["split"] == "train"].index.values
        val_idx = split_df[split_df["split"] == "val"].index.values
        test_idx = split_df[split_df["split"] == "test"].index.values

    else:
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - TRAIN_FRAC), random_state=RANDOM_STATE)
        train_idx, temp_idx = next(sss1.split(X, y))

        X_temp = X[temp_idx]
        y_temp = y[temp_idx]

        relative_test_size = TEST_FRAC / (VAL_FRAC + TEST_FRAC)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_test_size, random_state=RANDOM_STATE)
        val_idx_rel, test_idx_rel = next(sss2.split(X_temp, y_temp))

        val_idx = temp_idx[val_idx_rel]
        test_idx = temp_idx[test_idx_rel]

    print("Train:", len(train_idx))
    print("Val:", len(val_idx))
    print("Test:", len(test_idx))

    print("\nTrain tissue proportions:")
    print(pd.Series(y[train_idx]).value_counts(normalize=True))
    print("\nVal tissue proportions:")
    print(pd.Series(y[val_idx]).value_counts(normalize=True))
    print("\nTest tissue proportions:")
    print(pd.Series(y[test_idx]).value_counts(normalize=True))

    final_df["split"] = "unknown"
    final_df.loc[train_idx, "split"] = "train"
    final_df.loc[val_idx, "split"] = "val"
    final_df.loc[test_idx, "split"] = "test"

    embedding_metadata_df = final_df[["sample_id", "tissue", "sample_type", LABEL_COL, "split"]].copy()

    # -----------------------------------------------------
    # Tensors and loaders
    # -----------------------------------------------------
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    full_dataset = TensorDataset(X_tensor, y_tensor)

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    vae = VariationalAutoencoder(
        input_shape=X_tensor[0].shape, 
        latent_features=LATENT_FEATURES,
        fixed_log_sigma_x=FIXED_LOG_SIGMA_X,
        fixed_log_sigma_z=FIXED_LOG_SIGMA_Z,
    ).to(DEVICE)
    
    vi = VariationalInference(
        beta=BETA, 
        lambda_contrastive=LAMBDA_CONTRASTIVE, 
        contrastive_temperature=CONTRASTIVE_TEMPERATURE, 
        use_posterior_mean=True,
    )
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    training_data = defaultdict(list)
    validation_data = defaultdict(list)

    run_config = {
        "counts_path": COUNTS_PATH,
        "metadata_path": METADATA_PATH,
        "label_col": LABEL_COL,
        "latent_features": LATENT_FEATURES,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "beta": BETA,
        "lambda_contrastive": LAMBDA_CONTRASTIVE,
        "contrastive_temperature": CONTRASTIVE_TEMPERATURE,
        "fixed_log_sigma_x": FIXED_LOG_SIGMA_X,
        "fixed_log_sigma_z": FIXED_LOG_SIGMA_Z,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "input_shape": list(X_tensor[0].shape),
        "feature_cols": feature_cols,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "device": str(DEVICE),
    }
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)

    # -----------------------------------------------------
    # Train the model and save progress after each epoch
    # -----------------------------------------------------
    best_val_elbo = -np.inf
    best_val_total_loss = np.inf

    for epoch in range(1, NUM_EPOCHS + 1):
        vae.train()
        training_epoch_data = defaultdict(list)

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            loss, diagnostics, _ = vi(vae, x_batch, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k, v in diagnostics.items():
                training_epoch_data[k].append(v.mean().item())

        for k, v in training_epoch_data.items():
            training_data[k].append(float(np.mean(v)))

        # Evaluate the model on the validation set without updating weights
        vae.eval()
        validation_epoch_data = defaultdict(list)

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                loss, diagnostics, _ = vi(vae, x_batch, y_batch)

                for k, v in diagnostics.items():
                    validation_epoch_data[k].append(v.mean().item())

        for k, v in validation_epoch_data.items():
            validation_data[k].append(float(np.mean(v)))

        # Save training history
        history_df = pd.DataFrame({
            "epoch": np.arange(1, len(training_data["elbo"]) + 1),

            "train_elbo": training_data["elbo"],
            "train_kl": training_data["kl"],
            "train_log_px": training_data["log_px"],
            "train_vae_loss": training_data["vae_loss"],
            "train_contrastive_loss": training_data["contrastive_loss"],
            "train_total_loss": training_data["total_loss"],

            "val_elbo": validation_data["elbo"],
            "val_kl": validation_data["kl"],
            "val_log_px": validation_data["log_px"],
            "val_vae_loss": validation_data["vae_loss"],
            "val_contrastive_loss": validation_data["contrastive_loss"],
            "val_total_loss": validation_data["total_loss"],
        })
        history_df.to_csv(history_path, index=False)

        # Save the most recent model checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": vae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_data": dict(training_data),
            "validation_data": dict(validation_data),
            "config": run_config,
        }, model_path)

        # Save the checkpoint with the best validation ELBO
        current_val_elbo = validation_data["elbo"][-1]
        if current_val_elbo > best_val_elbo:
            best_val_elbo = current_val_elbo
            torch.save({
                "epoch": epoch,
                "model_state_dict": vae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_elbo": best_val_elbo,
                "config": run_config,
            }, best_model_path)

        # Save the checkpoint with the best validation ELBO based on total loss as well
        current_val_total_loss = validation_data["total_loss"][-1]
        if current_val_total_loss < best_val_total_loss:
            best_val_total_loss = current_val_total_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": vae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_total_loss": best_val_total_loss,
                "config": run_config,
            }, best_total_loss_path)

        # Update the training-curve plot
        make_vae_plots(training_data, validation_data, save_path=plot_path)

        # Periodically save latent embeddings so training progress can be inspected
        if epoch % 25 == 0 or epoch == 1: 
            vae.eval()
            with torch.no_grad():
                X_all_device = X_tensor.to(DEVICE)
                X_all_enc = torch.log1p(X_all_device)  
                qz_all = vae.posterior(X_all_enc)   # Compute q(z|x) for all samples
                mu_all = qz_all.mu.cpu().numpy()       # Use the posterior mean as the latent embedding

            epoch_latent_df = embedding_metadata_df.copy()
            for i in range(mu_all.shape[1]):
                epoch_latent_df[f"z{i+1}"] = mu_all[:, i]

            epoch_latent_path = latent_epochs_dir / f"latent_embeddings_epoch_{epoch:03d}.csv"
            epoch_latent_df.to_csv(epoch_latent_path, index=False)
    
    def save_latents_from_checkpoint(checkpoint_path: Path, output_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        vae.load_state_dict(checkpoint["model_state_dict"])
        vae.eval()

        with torch.no_grad():
            X_all_device = X_tensor.to(DEVICE)
            X_all_enc = torch.log1p(X_all_device)
            qz_all = vae.posterior(X_all_enc)
            mu_all = qz_all.mu.cpu().numpy()

        latent_df = embedding_metadata_df.copy()
        for i in range(mu_all.shape[1]):
            latent_df[f"z{i+1}"] = mu_all[:, i]

        latent_df.to_csv(output_path, index=False)

    # -----------------------------------------------------
    # Save latent embeddings for all samples used in this run
    # -----------------------------------------------------
    vae.eval()
    with torch.no_grad():
        X_all_device = X_tensor.to(DEVICE)
        X_all_enc = torch.log1p(X_all_device)
        qz_all = vae.posterior(X_all_enc) # Compute the latent posterior q(z|x) for all samples
        mu_all = qz_all.mu.cpu().numpy()     # Use the posterior mean as the deterministic latent embedding

    latent_df = embedding_metadata_df.copy()
    for i in range(mu_all.shape[1]):
        latent_df[f"z{i+1}"] = mu_all[:, i]
    latent_df.to_csv(latent_path, index=False)

    save_latents_from_checkpoint(
        best_model_path,
        output_dir / "latent_embeddings_best_elbo.csv"
    )

    save_latents_from_checkpoint(
        best_total_loss_path,
        output_dir / "latent_embeddings_best_total_loss.csv"
    )

    print("\nSaved outputs:")
    print(f"  Config:           {config_path}")
    print(f"  Splits:           {split_path}")
    print(f"  History:          {history_path}")
    print(f"  Curves:           {plot_path}")
    print(f"  Latest model:     {model_path}")
    print(f"  Best model:       {best_model_path}")
    print(f"  Best total loss:  {best_total_loss_path}")
    print(f"  Latent embeddings:{latent_path}")


if __name__ == "__main__":
    main()