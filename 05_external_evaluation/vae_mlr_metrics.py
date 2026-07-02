#!/usr/bin/env python3

# Purpose: evaluate an external cohort using a trained VAE, latent-space classifier, and diagnosis-to-tissue mapping.

from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from vae_model import VariationalAutoencoder


# =========================================================
# Configuration
# These paths point to the trained VAE, the saved classifier,
# the external cohort files, and the output files created by this script.
# =========================================================

VAE_MODEL_PATH = Path("vae_model_best.pt")
VAE_CONFIG_PATH = Path("run_config.json")

SCALER_PATH = Path("vae_latent_scaler.joblib")
CLASSIFIER_PATH = Path("vae_logistic_classifier.joblib")

FEATURE_COLS_PATH = Path("feature_cols.json")

EXTERNAL_COUNTS_PATH = Path("/ngc/projects/gm_ext/lucdel/counts_UPT_Ensembl.tsv")
EXTERNAL_METADATA_PATH = Path("/ngc/projects/gm_ext/lucdel/labels_UPT.tsv")

OUTPUT_DIR = Path("external_test_samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXTERNAL_LATENT_PATH = OUTPUT_DIR / "external_latent_embeddings_best.csv"
EXTERNAL_PREDICTIONS_PATH = OUTPUT_DIR / "external_predictions_probabilities.csv"
EVALUATED_PREDICTIONS_PATH = OUTPUT_DIR / "evaluated_predictions_vae.csv"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix_vae.csv"
CLASSIFICATION_REPORT_PATH = OUTPUT_DIR / "classification_report_vae.txt"
METRICS_PATH = OUTPUT_DIR / "metrics_vae.csv"

FILL_MISSING_GENES_WITH_ZERO = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =========================================================
# Diagnosis to valid tissue labels
# Some diagnoses can correspond to more than one acceptable tissue label.
# This dictionary is used when evaluating external predictions.
# =========================================================

diagnosis_to_tissues = {
    "Biliary tract cancer": ["bile duct", "liver"],
    "Breast cancer": ["breast"],
    "Cervical cancer": ["cervix", "uterus"],
    "Colorectal cancer": ["colorectal"],
    "Endometrial cancer": ["uterus", "endometrium"],
    "Esophageal cancer": ["esophagus"],
    "Gastric/GEJ cancer": ["stomach", "esophagus"],
    "Head and neck cancer": ["head and neck"],
    "Melanoma": ["skin"],
    "Mesothelioma": ["pleura", "lung"],
    "Neuroendocrine carcinoma (NEC)": ["colorectal", "lung", "pancreas"],
    "Non-small cell lung cancer (NSCLC)": ["lung"],
    "Ovarian cancer": ["ovary"],
    "Pancreatic cancer": ["pancreas"],
    "Prostate cancer": ["prostate"],
    "Sarcoma": ["soft tissue"],
    "Small cell lung cancer (SCLC)": ["lung"],
    "Urothelial cancer": ["bladder", "urinary tract"],
}


# =========================================================
# Utility functions
# These helper functions prepare external counts, align gene IDs,
# generate latent embeddings, and evaluate predictions.
# =========================================================

def add_training_versions_to_external_ensembl_ids(
    counts_df: pd.DataFrame,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Map external ENSEMBL IDs without version to the exact ENSEMBL IDs
    with version used during training.

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

    return counts_df


def load_and_align_external_counts(
    counts_path: Path,
    feature_cols: list,
    fill_missing_genes_with_zero: bool = True,
) -> pd.DataFrame:
    """
    Load external counts where genes are rows and samples are columns,
    normalize by 40M library size, transpose to samples x genes,
    and align to the training feature order.
    """

    print("\nLoading external counts...")
    counts_df = pd.read_csv(counts_path, sep="\s+")

    if "geneid" not in counts_df.columns:
        raise ValueError("External counts file must contain a 'geneid' column.")

    print("Original external counts shape:", counts_df.shape)

    counts_df = counts_df.rename(columns={"geneid": "ensembl_gene_id"})

    counts_df = add_training_versions_to_external_ensembl_ids(
        counts_df=counts_df,
        feature_cols=feature_cols,
    )

    # Normalize each external sample to a fixed library size of 40 million counts.
    # This makes the external cohort more comparable before log transformation.
    gene_ids = counts_df["ensembl_gene_id"]
    sample_counts = counts_df.drop(columns=["ensembl_gene_id"])

    library_sizes = sample_counts.sum(axis=0)

    if (library_sizes == 0).any():
        bad_samples = library_sizes[library_sizes == 0].index.tolist()
        raise ValueError(f"These samples have library size 0: {bad_samples}")

    sample_counts = sample_counts.div(library_sizes, axis=1) * 40_000_000

    counts_df = pd.concat([gene_ids, sample_counts], axis=1)

    # Collapse duplicate gene IDs after version mapping by summing their counts.
    counts_df = (
        counts_df
        .groupby("ensembl_gene_id", as_index=False)
        .sum(numeric_only=True)
    )

    counts_df = counts_df.set_index("ensembl_gene_id")

    # Transpose the matrix so rows are samples and columns are genes, matching training input.
    external_df = counts_df.T.reset_index()
    external_df = external_df.rename(columns={"index": "sample_id"})

    print("External counts after transpose:", external_df.shape)

    missing_features = [
        gene for gene in feature_cols
        if gene not in external_df.columns
    ]

    if missing_features:
        if fill_missing_genes_with_zero:
            print(
                f"WARNING: {len(missing_features)} training genes are missing "
                "in external matrix."
            )
            print("Filling missing genes with 0.")

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
        c for c in external_df.columns
        if c not in feature_cols and c != "sample_id"
    ]

    if extra_features:
        print(f"WARNING: Found {len(extra_features)} extra genes. Ignoring.")

    external_df = external_df[["sample_id"] + feature_cols]

    print("Final aligned external matrix shape:", external_df.shape)

    return external_df


def evaluate_external_predictions(
    predictions_df: pd.DataFrame,
    metadata_path: Path,
    output_dir: Path,
) -> None:
    """
    Merge predictions with metadata diagnosis, map diagnosis to valid tissues,
    calculate metrics, and save evaluation outputs.
    """

    print("\nEvaluating external predictions...")

    meta_df = pd.read_csv(metadata_path, sep="\t")

    meta_df = meta_df.rename(columns={
        "SID": "sample_id",
        "Diagnosis": "diagnosis",
    })

    if "sample_id" not in meta_df.columns:
        raise ValueError("Metadata file must contain column 'SID' or 'sample_id'.")

    if "diagnosis" not in meta_df.columns:
        raise ValueError("Metadata file must contain column 'Diagnosis' or 'diagnosis'.")

    predictions_df = predictions_df.copy()
    meta_df = meta_df.copy()

    predictions_df["sample_id"] = predictions_df["sample_id"].astype(str).str.strip()
    meta_df["sample_id"] = meta_df["sample_id"].astype(str).str.strip()

    # External prediction sample IDs may be like SAMPLE-xxx.
    # The metadata SID usually corresponds to the first part.
    predictions_df["sample_id_short"] = predictions_df["sample_id"].str.split("-").str[0]
    meta_df["sample_id_short"] = meta_df["sample_id"]

    predictions_df["predicted_tissue"] = (
        predictions_df["predicted_tissue"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    meta_df["diagnosis"] = (
        meta_df["diagnosis"]
        .astype(str)
        .str.strip()
    )

    meta_df_unique = (
        meta_df
        .drop_duplicates(subset=["sample_id_short"], keep="first")
        .copy()
    )

    df = predictions_df.merge(
        meta_df_unique[["sample_id_short", "diagnosis"]],
        on="sample_id_short",
        how="left",
    )

    df["true_tissue"] = df["diagnosis"].map(diagnosis_to_tissues)

    missing_diagnosis = df[df["diagnosis"].isna()]
    missing_tissue = df[df["true_tissue"].isna()]

    print("Total predictions:", len(predictions_df))
    print("Samples after merge:", len(df))
    print("Samples without diagnosis:", len(missing_diagnosis))
    print("Samples without true tissue mapping:", len(missing_tissue))

    if len(missing_diagnosis) > 0:
        print("\nSamples without diagnosis:")
        print(missing_diagnosis["sample_id"].unique())

    if len(missing_tissue) > 0:
        print("\nDiagnoses without tissue mapping:")
        print(missing_tissue["diagnosis"].dropna().unique())

    df_eval = df.dropna(
        subset=["diagnosis", "true_tissue", "predicted_tissue"]
    ).copy()

    if df_eval.empty:
        raise ValueError(
            "No evaluable samples left after merging predictions with metadata."
        )

    # Mark a prediction as correct if the predicted tissue is one of the valid tissues
    # for that diagnosis.
    df_eval["correct"] = df_eval.apply(
        lambda row: row["predicted_tissue"] in row["true_tissue"],
        axis=1,
    )

    # Use the first valid tissue as the canonical label for single-label metrics.
    df_eval["true_tissue_canonical"] = df_eval["true_tissue"].apply(lambda x: x[0])

    # For multi-valid-label cases, convert correct predictions to canonical label
    # so MCC/confusion matrix can be computed as single-label classification.
    df_eval["predicted_tissue_adjusted"] = df_eval.apply(
        lambda row: row["true_tissue_canonical"]
        if row["predicted_tissue"] in row["true_tissue"]
        else row["predicted_tissue"],
        axis=1,
    )

    accuracy = df_eval["correct"].mean()

    mcc = matthews_corrcoef(
        df_eval["true_tissue_canonical"],
        df_eval["predicted_tissue_adjusted"],
    )

    precision_macro = precision_score(
        df_eval["true_tissue_canonical"],
        df_eval["predicted_tissue_adjusted"],
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        df_eval["true_tissue_canonical"],
        df_eval["predicted_tissue_adjusted"],
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        df_eval["true_tissue_canonical"],
        df_eval["predicted_tissue_adjusted"],
        average="macro",
        zero_division=0,
    )

    print("\nExternal evaluation metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Precision macro: {precision_macro:.4f}")
    print(f"Recall macro: {recall_macro:.4f}")
    print(f"F1 macro: {f1_macro:.4f}")

    report = classification_report(
        df_eval["true_tissue_canonical"],
        df_eval["predicted_tissue_adjusted"],
        zero_division=0,
    )

    print("\nClassification report:")
    print(report)

    # Original tissue-vs-tissue confusion matrix for metrics

    labels = sorted(
        set(df_eval["true_tissue_canonical"]) |
        set(df_eval["predicted_tissue_adjusted"])
    )

    cm = confusion_matrix(
        df_eval["true_tissue_canonical"],
        df_eval["predicted_tissue_adjusted"],
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{x}" for x in labels],
        columns=[f"pred_{x}" for x in labels],
    )

    cm_df.to_csv(output_dir / "confusion_matrix_tissue_vs_tissue_vae.csv")

    # New diagnosis-vs-predicted tissue confusion matrix for interpretability
    cm_diag_tissue = pd.crosstab(
        df_eval["diagnosis"],
        df_eval["predicted_tissue"],
        rownames=["true_diagnosis"],
        colnames=["predicted_tissue"],
        dropna=False
    )

    cm_diag_tissue = cm_diag_tissue.loc[
        cm_diag_tissue.sum(axis=1) > 0,
        cm_diag_tissue.sum(axis=0) > 0
    ]

    cm_diag_tissue.to_csv(
        output_dir / "confusion_matrix_diagnosis_vs_tissue_vae.csv"
    )

    metrics_df = pd.DataFrame([{
        "n_predictions": len(predictions_df),
        "n_evaluable": len(df_eval),
        "n_missing_diagnosis": len(missing_diagnosis),
        "n_missing_tissue_mapping": len(missing_tissue),
        "accuracy": accuracy,
        "mcc": mcc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }])

    df_eval.to_csv(output_dir / "evaluated_predictions_vae.csv", index=False)
    
    metrics_df.to_csv(output_dir / "metrics_vae.csv", index=False)

    with open(output_dir / "classification_report_vae.txt", "w") as f:
        f.write(report)

    print("\nSaved evaluation files:")
    print(output_dir / "evaluated_predictions_vae.csv")
    print(output_dir / "confusion_matrix_vae.csv")
    print(output_dir / "metrics_vae.csv")
    print(output_dir / "classification_report_vae.txt")


def main() -> None:
    print(f">> Using device: {DEVICE}")

    # -----------------------------------------------------
    # Load config and feature columns
    # -----------------------------------------------------

    print("\nLoading VAE config and feature columns...")

    with open(VAE_CONFIG_PATH, "r") as f:
        config = json.load(f)

    with open(FEATURE_COLS_PATH, "r") as f:
        feature_cols = json.load(f)

    if "input_shape" in config:
        input_shape = torch.Size(config["input_shape"])
    elif "n_features" in config:
        input_shape = torch.Size([int(config["n_features"])])
    else:
        input_shape = torch.Size([len(feature_cols)])

    latent_features = int(config["latent_features"])

    fixed_log_sigma_x = float(config.get("fixed_log_sigma_x", 0.0))
    fixed_log_sigma_z = float(config.get("fixed_log_sigma_z", 0.0))

    print("Number of training genes:", len(feature_cols))
    print("Latent features:", latent_features)

    # -----------------------------------------------------
    # Load and align external counts
    # -----------------------------------------------------

    external_df = load_and_align_external_counts(
        counts_path=EXTERNAL_COUNTS_PATH,
        feature_cols=feature_cols,
        fill_missing_genes_with_zero=FILL_MISSING_GENES_WITH_ZERO,
    )

    print("Number of external samples:", external_df.shape[0])

    X = external_df[feature_cols].values.astype(np.float32)
    X = np.log1p(X)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    print("X shape:", X.shape)

    # -----------------------------------------------------
    # Load best VAE model
    # -----------------------------------------------------

    print("\nLoading best VAE model...")

    vae = VariationalAutoencoder(
        input_shape=input_shape,
        latent_features=latent_features,
        fixed_log_sigma_x=fixed_log_sigma_x,
        fixed_log_sigma_z=fixed_log_sigma_z,
    ).to(DEVICE)

    checkpoint = torch.load(VAE_MODEL_PATH, map_location=DEVICE)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()

    print("Loaded best model from:", VAE_MODEL_PATH)
    print("Best model epoch:", checkpoint.get("epoch"))
    print("Best val ELBO:", checkpoint.get("best_val_elbo"))

    # -----------------------------------------------------
    # Generate latent embeddings
    # -----------------------------------------------------

    print("\nGenerating latent embeddings for external samples...")

    with torch.no_grad():
        qz = vae.posterior(X_tensor)
        mu = qz.mu.cpu().numpy()

    latent_df = external_df[["sample_id"]].copy()
    latent_df["split"] = "external"

    for i in range(mu.shape[1]):
        latent_df[f"z{i + 1}"] = mu[:, i]

    latent_df.to_csv(EXTERNAL_LATENT_PATH, index=False)

    print("\nSaved latent embeddings:")
    print(EXTERNAL_LATENT_PATH)
    print("Latent dataframe shape:", latent_df.shape)

    # -----------------------------------------------------
    # Load trained classifier and scaler
    # -----------------------------------------------------

    print("\nLoading trained classifier and scaler...")

    scaler = joblib.load(SCALER_PATH)
    clf = joblib.load(CLASSIFIER_PATH)

    print("Loaded scaler:", SCALER_PATH)
    print("Loaded classifier:", CLASSIFIER_PATH)

    latent_feature_cols = [c for c in latent_df.columns if c.startswith("z")]
    X_latent_external = latent_df[latent_feature_cols].values.astype(np.float32)

    # Important: do NOT fit scaler again
    X_latent_external_scaled = scaler.transform(X_latent_external)

    # -----------------------------------------------------
    # Predict external samples
    # -----------------------------------------------------

    print("\nPredicting external samples...")

    y_pred_external = clf.predict(X_latent_external_scaled)
    y_prob_external = clf.predict_proba(X_latent_external_scaled)

    class_labels = clf.classes_

    print("\nExternal predicted tissue distribution:")
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
        "predicted_tissue": y_pred_external,
    })

    predictions_df = pd.concat(
        [
            predictions_df.reset_index(drop=True),
            prob_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Add the top 1 and top 2 predicted tissues with their probabilities for easier review.
    top_indices = np.argsort(y_prob_external, axis=1)[:, ::-1]

    predictions_df["top1_tissue"] = [
        class_labels[i] for i in top_indices[:, 0]
    ]
    predictions_df["top1_probability"] = [
        y_prob_external[row_idx, class_idx]
        for row_idx, class_idx in enumerate(top_indices[:, 0])
    ]

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

    # -----------------------------------------------------
    # Evaluate predictions using diagnosis dictionary
    # -----------------------------------------------------

    evaluate_external_predictions(
        predictions_df=predictions_df,
        metadata_path=EXTERNAL_METADATA_PATH,
        output_dir=OUTPUT_DIR,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()