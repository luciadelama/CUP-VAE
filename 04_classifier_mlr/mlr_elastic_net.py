"""
Train and evaluate a multinomial logistic regression classifier with elastic net
regularization using raw gene expression counts as input features.

The script:
1. Loads raw count data, sample metadata, and predefined train/validation/test splits.
2. Merges the files by sample_id.
3. Applies log1p transformation and standardization using only the training set.
4. Trains a PyTorch linear classifier with manual L1 and L2 penalties.
5. Saves metrics, plots, predictions, model parameters, and preprocessing objects.
"""
# =========================
# Multinomial logistic regression with elastic net regularization
# Input features are raw gene expression counts
# =========================

import os
import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)


# =========================
# Input and output paths
# =========================

RAW_COUNTS_PATH = "../data/matrices/final_counts_filtered_transpose_collapsed_replicates.csv"
METADATA_PATH = "../data/metadata/final_sample_metadata_collapsed_replicates.csv"
SPLITS_PATH = "../data/splits/data_splits.csv"

JOB_ID = os.environ.get("SLURM_JOB_ID", "local")
OUTPUT_DIR = Path(f"mlr_outputs_{JOB_ID}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Outputs will be saved in: {OUTPUT_DIR.resolve()}")


# =========================
# Reproducibility settings
# =========================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================
# Training hyperparameters
# =========================

BATCH_SIZE = 256
NUM_EPOCHS = 300
LEARNING_RATE = 1e-3

# Keep AdamW weight_decay at 0 because the L2 penalty is added manually in elastic_net_penalty
WEIGHT_DECAY = 0.0

# Strength of the manual L1 and L2 regularization terms
LAMBDA_L1 = 1e-5
LAMBDA_L2 = 1e-5

# Maximum gradient norm used to stabilize training
GRAD_CLIP_MAX_NORM = 5.0


# =========================
# Load and merge raw counts, metadata, and split information
# =========================

raw_df = pd.read_csv(RAW_COUNTS_PATH)
meta_df = pd.read_csv(METADATA_PATH)
split_df = pd.read_csv(SPLITS_PATH)

# Add tissue labels to the count matrix using sample_id.
df = raw_df.merge(
    meta_df[["sample_id", "tissue"]],
    on="sample_id",
    how="inner"
)

# Add the predefined train/validation/test split for each sample.
df = df.merge(
    split_df[["sample_id", "split"]],
    on="sample_id",
    how="inner"
)

feature_cols = [c for c in df.columns if c not in ["sample_id", "tissue", "split"]]

X = df[feature_cols].values.astype(np.float32)
y = df["tissue"].values
splits = df["split"].values

print(f"Full dataset shape: {X.shape}")
print(f"Number of genes/features: {len(feature_cols)}")


# =========================
# Encode tissue labels
# =========================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

n_features = X.shape[1]
n_classes = len(label_encoder.classes_)

print(f"Number of classes: {n_classes}")


# =========================
# Create train, validation, and test arrays
# =========================

train_mask = splits == "train"
val_mask = splits == "val"
test_mask = splits == "test"

X_train = X[train_mask]
y_train = y_encoded[train_mask]
sample_train = df.loc[train_mask, "sample_id"].values

X_val = X[val_mask]
y_val = y_encoded[val_mask]
sample_val = df.loc[val_mask, "sample_id"].values

X_test = X[test_mask]
y_test = y_encoded[test_mask]
sample_test = df.loc[test_mask, "sample_id"].values

print(f"Train samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")


# =========================
# Preprocessing
# Apply log1p to reduce count scale differences, then fit StandardScaler on the training set only
# =========================

X_train = np.log1p(X_train)
X_val = np.log1p(X_val)
X_test = np.log1p(X_test)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# =========================
# Convert NumPy arrays to PyTorch datasets and data loaders
# =========================

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================
# Model definition
# =========================

class MultinomialLogisticRegression(nn.Module):
    """Single linear layer used as multinomial logistic regression."""

    def __init__(self, n_features, n_classes):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x):
        """Return one logit score per class for each input sample."""
        return self.linear(x)


# =========================
# Helper functions for regularization, evaluation, training, and saving
# =========================

def compute_weight_norm(model):
    """Return the L2 norm of the classifier weight matrix."""
    return model.linear.weight.detach().norm(p=2).item()


def compute_grad_norm(model):
    """Compute the total L2 norm of all gradients before optional clipping."""
    total_norm_sq = 0.0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().norm(2)
            total_norm_sq += param_norm.item() ** 2

    return total_norm_sq ** 0.5


def elastic_net_penalty(model, lambda_l1, lambda_l2):
    """Compute the elastic net penalty from model weights only."""
    l1_norm = 0.0
    l2_norm = 0.0

    for name, p in model.named_parameters():
        # Apply elastic net only to weights; biases are usually not regularized
        if "weight" in name:
            l1_norm = l1_norm + p.abs().sum()
            l2_norm = l2_norm + p.pow(2).sum()

    penalty = lambda_l1 * l1_norm + lambda_l2 * l2_norm

    return penalty, l1_norm.item(), l2_norm.item()


def evaluate(model, loader, criterion, device):
    """Evaluate loss, accuracy, MCC, predictions, and probabilities for one data loader."""
    model.eval()

    total_loss = 0.0
    total = 0

    all_true = []
    all_pred = []
    all_prob = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())
            all_prob.extend(probs.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = accuracy_score(all_true, all_pred)
    mcc = matthews_corrcoef(all_true, all_pred)

    return (
        avg_loss,
        accuracy,
        mcc,
        np.array(all_true),
        np.array(all_pred),
        np.array(all_prob)
    )


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    lambda_l1,
    lambda_l2,
    grad_clip_max_norm
):
    """Train the model and store training and validation metrics for each epoch."""
    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "train_mcc": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_mcc": [],
        "weight_norm": [],
        "grad_norm": [],
        "l1_norm": [],
        "l2_norm": []
    }

    for epoch in range(num_epochs):
        model.train()

        epoch_grad_norms = []
        epoch_l1_norms = []
        epoch_l2_norms = []

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(inputs)

            ce_loss = criterion(logits, labels)
            penalty, l1_norm, l2_norm = elastic_net_penalty(
                model,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2
            )

            loss = ce_loss + penalty

            loss.backward()

            grad_norm = compute_grad_norm(model)
            epoch_grad_norms.append(grad_norm)
            epoch_l1_norms.append(l1_norm)
            epoch_l2_norms.append(l2_norm)

            if grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip_max_norm
                )

            optimizer.step()

        train_loss, train_acc, train_mcc, _, _, _ = evaluate(
            model,
            train_loader,
            criterion,
            device
        )

        val_loss, val_acc, val_mcc, _, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        weight_norm = compute_weight_norm(model)
        mean_grad_norm = float(np.mean(epoch_grad_norms))
        mean_l1_norm = float(np.mean(epoch_l1_norms))
        mean_l2_norm = float(np.mean(epoch_l2_norms))

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["train_mcc"].append(train_mcc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_mcc"].append(val_mcc)
        history["weight_norm"].append(weight_norm)
        history["grad_norm"].append(mean_grad_norm)
        history["l1_norm"].append(mean_l1_norm)
        history["l2_norm"].append(mean_l2_norm)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train loss: {train_loss:.6f}, "
                f"Train acc: {train_acc:.4f}, "
                f"Train MCC: {train_mcc:.4f}, "
                f"Val loss: {val_loss:.6f}, "
                f"Val acc: {val_acc:.4f}, "
                f"Val MCC: {val_mcc:.4f}, "
                f"Weight norm: {weight_norm:.4f}, "
                f"Grad norm: {mean_grad_norm:.4f}"
            )

    return history


def plot_history(history_df, output_dir):
    """Create and save training-history plots."""
    # Plot cross-entropy loss over training epochs
    plt.figure(figsize=(7, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("MLR elastic net loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mlr_elastic_net_loss.png", dpi=300)
    plt.close()

    # Plot accuracy over training epochs
    plt.figure(figsize=(7, 5))
    plt.plot(history_df["epoch"], history_df["train_accuracy"], label="Train")
    plt.plot(history_df["epoch"], history_df["val_accuracy"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MLR elastic net accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mlr_elastic_net_accuracy.png", dpi=300)
    plt.close()

    # Plot Matthews correlation coefficient over training epochs
    plt.figure(figsize=(7, 5))
    plt.plot(history_df["epoch"], history_df["train_mcc"], label="Train")
    plt.plot(history_df["epoch"], history_df["val_mcc"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MCC")
    plt.title("MLR elastic net MCC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mlr_elastic_net_mcc.png", dpi=300)
    plt.close()

    # Plot the L2 norm of the learned weight matrix
    plt.figure(figsize=(7, 5))
    plt.plot(history_df["epoch"], history_df["weight_norm"])
    plt.xlabel("Epoch")
    plt.ylabel("L2 norm of weights")
    plt.title("MLR elastic net weight norm")
    plt.tight_layout()
    plt.savefig(output_dir / "mlr_elastic_net_weight_norm.png", dpi=300)
    plt.close()

    # Plot the average gradient norm before clipping
    plt.figure(figsize=(7, 5))
    plt.plot(history_df["epoch"], history_df["grad_norm"])
    plt.xlabel("Epoch")
    plt.ylabel("Gradient norm")
    plt.title("MLR elastic net gradient norm")
    plt.tight_layout()
    plt.savefig(output_dir / "mlr_elastic_net_grad_norm.png", dpi=300)
    plt.close()

    # Plot the L1 norm of the learned weight matrix
    plt.figure(figsize=(7, 5))
    plt.plot(history_df["epoch"], history_df["l1_norm"])
    plt.xlabel("Epoch")
    plt.ylabel("L1 norm of weights")
    plt.title("MLR elastic net L1 norm")
    plt.tight_layout()
    plt.savefig(output_dir / "mlr_elastic_net_l1_norm.png", dpi=300)
    plt.close()

    # Plot the squared L2 norm of the learned weight matrix
    plt.figure(figsize=(7, 5))
    plt.plot(history_df["epoch"], history_df["l2_norm"])
    plt.xlabel("Epoch")
    plt.ylabel("Squared L2 norm of weights")
    plt.title("MLR elastic net L2 norm")
    plt.tight_layout()
    plt.savefig(output_dir / "mlr_elastic_net_l2_norm.png", dpi=300)
    plt.close()


def save_predictions(
    output_dir,
    split_name,
    sample_ids,
    y_true,
    y_pred,
    probs,
    label_encoder
):
    """Save sample IDs, labels, predictions, and class probabilities to CSV."""
    true_labels = label_encoder.inverse_transform(y_true)
    pred_labels = label_encoder.inverse_transform(y_pred)

    pred_df = pd.DataFrame({
        "sample_id": sample_ids,
        "true_label": true_labels,
        "predicted_label": pred_labels,
        "correct": true_labels == pred_labels
    })

    # Store the model confidence for the predicted class
    pred_df["predicted_probability"] = probs.max(axis=1)

    # Store one probability column for each possible class
    for i, class_name in enumerate(label_encoder.classes_):
        pred_df[f"prob_{class_name}"] = probs[:, i]

    pred_df.to_csv(
        output_dir / f"mlr_elastic_net_{split_name}_predictions.csv",
        index=False
    )


# =========================
# Train the final model
# =========================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MultinomialLogisticRegression(n_features, n_classes).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    num_epochs=NUM_EPOCHS,
    lambda_l1=LAMBDA_L1,
    lambda_l2=LAMBDA_L2,
    grad_clip_max_norm=GRAD_CLIP_MAX_NORM
)


# =========================
# Save training curves and epoch-level metrics
# =========================

history_df = pd.DataFrame(history)
history_df.to_csv(
    OUTPUT_DIR / "mlr_elastic_net_training_history.csv",
    index=False
)

plot_history(history_df, OUTPUT_DIR)


# =========================
# Evaluate the trained model on train, validation, and test sets
# =========================

train_loss, train_acc, train_mcc, y_train_true, y_train_pred, train_probs = evaluate(
    model,
    train_loader,
    criterion,
    device
)

val_loss, val_acc, val_mcc, y_val_true, y_val_pred, val_probs = evaluate(
    model,
    val_loader,
    criterion,
    device
)

test_loss, test_acc, test_mcc, y_test_true, y_test_pred, test_probs = evaluate(
    model,
    test_loader,
    criterion,
    device
)

metrics = {
    "train_loss": float(train_loss),
    "train_accuracy": float(train_acc),
    "train_mcc": float(train_mcc),
    "val_loss": float(val_loss),
    "val_accuracy": float(val_acc),
    "val_mcc": float(val_mcc),
    "test_loss": float(test_loss),
    "test_accuracy": float(test_acc),
    "test_mcc": float(test_mcc),
    "final_weight_norm": float(compute_weight_norm(model)),
    "lambda_l1": LAMBDA_L1,
    "lambda_l2": LAMBDA_L2,
    "gradient_clipping_max_norm": GRAD_CLIP_MAX_NORM
}

with open(OUTPUT_DIR / "mlr_elastic_net_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nFinal metrics")
print(json.dumps(metrics, indent=2))


# =========================
# Save per-class metrics and confusion matrices
# =========================

for split_name, y_true, y_pred in [
    ("train", y_train_true, y_train_pred),
    ("val", y_val_true, y_val_pred),
    ("test", y_test_true, y_test_pred)
]:
    report = classification_report(
        y_true,
        y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(
        OUTPUT_DIR / f"mlr_elastic_net_{split_name}_classification_report.csv"
    )

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )

    cm_df.to_csv(
        OUTPUT_DIR / f"mlr_elastic_net_{split_name}_confusion_matrix.csv"
    )


# =========================
# Save sample-level predictions and class probabilities
# =========================

save_predictions(
    output_dir=OUTPUT_DIR,
    split_name="train",
    sample_ids=sample_train,
    y_true=y_train_true,
    y_pred=y_train_pred,
    probs=train_probs,
    label_encoder=label_encoder
)

save_predictions(
    output_dir=OUTPUT_DIR,
    split_name="val",
    sample_ids=sample_val,
    y_true=y_val_true,
    y_pred=y_val_pred,
    probs=val_probs,
    label_encoder=label_encoder
)

save_predictions(
    output_dir=OUTPUT_DIR,
    split_name="test",
    sample_ids=sample_test,
    y_true=y_test_true,
    y_pred=y_test_pred,
    probs=test_probs,
    label_encoder=label_encoder
)


# =========================
# Save the trained model, scaler, feature names, labels, and configuration
# =========================

checkpoint = {
    "model_state_dict": model.state_dict(),
    "n_features": n_features,
    "n_classes": n_classes,
    "feature_cols": feature_cols,
    "label_classes": label_encoder.classes_.tolist(),
    "config": {
        "model": "MultinomialLogisticRegression",
        "regularization": "elastic_net",
        "lambda_l1": LAMBDA_L1,
        "lambda_l2": LAMBDA_L2,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "gradient_clipping_max_norm": GRAD_CLIP_MAX_NORM,
        "input_transform": "log1p",
        "scaler": "StandardScaler fitted on training set",
        "seed": SEED,
        "raw_counts_path": RAW_COUNTS_PATH,
        "metadata_path": METADATA_PATH,
        "splits_path": SPLITS_PATH
    }
}

torch.save(
    checkpoint,
    OUTPUT_DIR / "mlr_elastic_net_model.pt"
)

joblib.dump(
    scaler,
    OUTPUT_DIR / "mlr_elastic_net_scaler.joblib"
)

with open(OUTPUT_DIR / "mlr_elastic_net_feature_cols.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

with open(OUTPUT_DIR / "mlr_elastic_net_label_classes.json", "w") as f:
    json.dump(label_encoder.classes_.tolist(), f, indent=2)

with open(OUTPUT_DIR / "mlr_elastic_net_config.json", "w") as f:
    json.dump(checkpoint["config"], f, indent=2)


# =========================
# Save learned coefficients and biases for interpretation
# =========================

weights = model.linear.weight.detach().cpu().numpy()
bias = model.linear.bias.detach().cpu().numpy()

weights_df = pd.DataFrame(
    weights,
    index=label_encoder.classes_,
    columns=feature_cols
)

weights_df.to_csv(
    OUTPUT_DIR / "mlr_elastic_net_coefficients.csv"
)

bias_df = pd.DataFrame({
    "class": label_encoder.classes_,
    "bias": bias
})

bias_df.to_csv(
    OUTPUT_DIR / "mlr_elastic_net_bias.csv",
    index=False
)


print("\nTraining completed successfully.")
print(f"All outputs saved in: {OUTPUT_DIR.resolve()}")