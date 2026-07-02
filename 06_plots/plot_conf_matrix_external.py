import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.colors import PowerNorm


# =========================
# Input and output paths
# =========================

INPUT_XLSX = "labelsMLR_withPurple.xlsx"  
SHEET_NAME = 0                           

OUTPUT_DIR = "confusion_matrix_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Mapping from each external diagnosis to the expected tissue label(s)
# =========================

diagnosis_to_tissues = {
    "Biliary tract cancer": ["bile duct"],
    "Breast cancer": ["breast"],
    "Cervical cancer": ["cervix"],
    "Colorectal cancer": ["colorectal"],
    "Endometrial cancer": ["uterus", "endometrium"],
    "Esophageal cancer": ["esophagus"],
    "Gastric/GEJ cancer": ["stomach"],
    "Head and neck cancer": ["head and neck"],
    "Melanoma": ["skin"],
    "Mesothelioma": ["pleura"],
    "Neuroendocrine carcinoma (NEC)": ["stomach", "lung", "pancreas"],
    "Non-small cell lung cancer (NSCLC)": ["lung"],
    "Ovarian cancer": ["ovary"],
    "Pancreatic cancer": ["pancreas"],
    "Prostate cancer": ["prostate"],
    "Sarcoma": ["soft tissue"],
    "Small cell lung cancer (SCLC)": ["lung"],
    "Urothelial cancer": ["bladder", "urinary tract"]
}


# =========================
# Helper functions
# =========================

def clean_label(x):
    """
    Standardizes labels for matching.
    Keeps readable labels but normalizes case and spaces.
    """
    if pd.isna(x):
        return np.nan
    return str(x).strip()


def clean_tissue(x):
    """
    Standardizes tissue names.
    """
    if pd.isna(x):
        return np.nan
    return str(x).strip().lower()


def find_column(df, candidates):
    """
    Finds a column by trying several possible names.
    Useful because Excel columns sometimes have slightly different names.
    """
    cols_lower = {c.lower().strip(): c for c in df.columns}

    for candidate in candidates:
        candidate_lower = candidate.lower().strip()
        if candidate_lower in cols_lower:
            return cols_lower[candidate_lower]

    raise ValueError(
        f"Could not find any of these columns: {candidates}\n"
        f"Available columns are: {list(df.columns)}"
    )


# =========================
# Load data
# =========================

# Load the external validation spreadsheet.
df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)

# Identify relevant columns
true_col = find_column(df, ["True label", "true_label", "Diagnosis", "diagnosis"])
pred_col = find_column(df, ["Predicted Tissue", "Predicted tissue", "predicted_tissue"])
qc_col = find_column(df, ["QC_purple", "Purity QC", "tumor_purity_QC", "purity_QC", "QC"])

# Clean labels and QC values before filtering and matching.
df[true_col] = df[true_col].apply(clean_label)
df[pred_col] = df[pred_col].apply(clean_tissue)
df[qc_col] = df[qc_col].astype(str).str.strip().str.upper()


# =========================
# Filter samples used in the external validation matrix
# =========================

df_filt = df.copy()

# Keep only samples that pass the tumour purity QC filter.
df_filt = df_filt[df_filt[qc_col] == "PASS"]

# Remove samples where no predicted tissue is available.
df_filt = df_filt.dropna(subset=[pred_col])

# Keep only diagnoses that are included in the expected-tissue dictionary.
df_filt = df_filt[df_filt[true_col].isin(diagnosis_to_tissues.keys())]

print(f"Initial samples: {len(df)}")
print(f"Samples after QC PASS and removing NA predictions: {len(df_filt)}")


# =========================
# Build confusion matrix as raw counts
# =========================

cm_counts = pd.crosstab(
    df_filt[true_col],
    df_filt[pred_col]
)

# Ensure all expected true labels appear as rows, even if some have 0 samples after filtering
all_true_labels = list(diagnosis_to_tissues.keys())
cm_counts = cm_counts.reindex(all_true_labels).fillna(0).astype(int)

# Remove rows with zero samples after filtering
row_totals = cm_counts.sum(axis=1)
cm_counts = cm_counts.loc[row_totals > 0]

# Convert counts to percentages within each diagnosis row.
cm_percent = cm_counts.div(cm_counts.sum(axis=1), axis=0) * 100


# =========================
# Reorder rows so expected tissue matches appear close to a diagonal pattern
# =========================

def get_best_expected_column(row_label, available_columns):
    expected_tissues = [
        clean_tissue(t)
        for t in diagnosis_to_tissues.get(row_label, [])
    ]

    expected_tissues = [
        t for t in expected_tissues
        if t in available_columns
    ]

    if len(expected_tissues) == 0:
        return None

    # Choose the expected tissue with the highest observed percentage for that diagnosis
    values = cm_percent.loc[row_label, expected_tissues]
    return values.idxmax()


# Assign each true label to its expected predicted tissue column
row_to_expected_col = {
    row: get_best_expected_column(row, cm_percent.columns)
    for row in cm_percent.index
}

# Rows with expected tissue present in the matrix
rows_with_expected = [
    row for row, col in row_to_expected_col.items()
    if col is not None
]

# Sort rows according to the position of their expected tissue in the columns
rows_ordered = sorted(
    rows_with_expected,
    key=lambda row: list(cm_percent.columns).index(row_to_expected_col[row])
)

# Rows whose expected tissue is not present as predicted tissue
rows_without_expected = [
    row for row in cm_percent.index
    if row not in rows_ordered
]

# Final row order
final_row_order = rows_ordered + rows_without_expected

cm_counts = cm_counts.loc[final_row_order]
cm_percent = cm_percent.loc[final_row_order]


# =========================
# Reorder columns so expected tissues appear first and in a meaningful order
# =========================

expected_col_order = []

for row in final_row_order:
    expected = diagnosis_to_tissues.get(row, [])
    for tissue in expected:
        tissue_clean = clean_tissue(tissue)
        if tissue_clean in cm_percent.columns and tissue_clean not in expected_col_order:
            expected_col_order.append(tissue_clean)

remaining_cols = [
    col for col in cm_percent.columns
    if col not in expected_col_order
]

final_col_order = expected_col_order + remaining_cols

cm_counts = cm_counts[final_col_order]
cm_percent = cm_percent[final_col_order]


# =========================
# Create heatmap annotations showing both percentage and count
# =========================

annot = cm_percent.copy().astype(str)

for i in cm_percent.index:
    for j in cm_percent.columns:
        percent = cm_percent.loc[i, j]
        count = cm_counts.loc[i, j]

        if count == 0:
            annot.loc[i, j] = ""
        else:
            annot.loc[i, j] = f"{percent:.1f}%\n({count})"


# =========================
# Save count and percentage matrices
# =========================

counts_path = os.path.join(
    OUTPUT_DIR,
    "external_confusion_matrix_counts_QC_PASS.csv"
)

percent_path = os.path.join(
    OUTPUT_DIR,
    "external_confusion_matrix_percent_QC_PASS.csv"
)

cm_counts.to_csv(counts_path)
cm_percent.to_csv(percent_path)


# =========================
# Plot the external validation heatmap
# =========================

n_rows = cm_percent.shape[0]
n_cols = cm_percent.shape[1]

plt.figure(figsize=(max(16, n_cols * 0.65), max(8, n_rows * 0.45)))

# Use PowerNorm to highlight both low and high percentage values.
ax = sns.heatmap(
    cm_percent,
    annot=annot,
    fmt="",
    cmap="Blues",
    norm=PowerNorm(gamma=0.5, vmin=0, vmax=100),
    linewidths=0.2,
    linecolor="lightgrey",
    cbar_kws={"label": "% of samples within diagnosis"}
)

plt.xlabel("Predicted tissue", fontsize=14)
plt.ylabel("True diagnosis", fontsize=14)
plt.title(
    "External validation cohort: predicted tissue distribution by true diagnosis",
    fontsize=16,
    pad=12
)

plt.xticks(rotation=45, ha="right", fontsize=11)
plt.yticks(rotation=0, fontsize=11)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=9)
cbar.set_label("% of samples within diagnosis", fontsize=12)

plt.tight_layout()

output_path = os.path.join(
    OUTPUT_DIR,
    "external_confusion_matrix_QC_PASS_ordered.png"
)

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {output_path}")
print(f"Saved count matrix to: {counts_path}")
print(f"Saved percentage matrix to: {percent_path}")