# ============================================================
# Differential Expression (PSEUDOBULK + Scanpy Wilcoxon DE)
# ============================================================

import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================
# Paths
# ============================

INPUT_FILE = "data/processed/adata_clustered.h5ad"
OUTPUT_DIR = "results/de_genes"
FIG_DIR = "results/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ============================
# Load Data
# ============================

print("Loading clustered AnnData...")
adata = sc.read_h5ad(INPUT_FILE)
print("Data shape:", adata.shape)

# ============================
# Detect Malignant Label
# ============================

POSSIBLE_COLUMNS = ["cell_type", "malignancy", "tumor_status", "annotation"]

label_col = None
for c in POSSIBLE_COLUMNS:
    if c in adata.obs.columns:
        label_col = c
        break

if label_col is None:
    raise ValueError(
        "No malignant/non-malignant column found.\n"
        "Expected one of: cell_type, malignancy, tumor_status, annotation\n"
        "Please add this manually before running DE."
    )

print(f"Using label column: {label_col}")

adata.obs["malignant_binary"] = (
    adata.obs[label_col]
    .astype(str)
    .str.lower()
    .str.contains("malignant|tumor|cancer")
)

print("Malignant counts:")
print(adata.obs["malignant_binary"].value_counts())

# ============================
# Detect Sample Column
# ============================

POSSIBLE_SAMPLE_COLUMNS = ["sample", "patient", "donor", "orig.ident", "batch"]

sample_col = None
for c in POSSIBLE_SAMPLE_COLUMNS:
    if c in adata.obs.columns:
        sample_col = c
        break

if sample_col is None:
    raise ValueError(
        "No sample/patient column found.\n"
        "Expected one of: sample, patient, donor, orig.ident, batch\n"
        "Pseudobulk requires real biological replicates."
    )

print(f"Using sample column: {sample_col}")

# ============================
# TEMP FIX: Use Current Expression as Pseudocounts
# (we'll treat adata.X as expression; this is exploratory)
# ============================

print("Using current adata.X as expression for pseudobulk (exploratory mode).")

X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

expr_df = pd.DataFrame(
    X,
    index=adata.obs_names,
    columns=adata.var_names,
)

# ============================
# Build Pseudobulk Metadata
# ============================

meta = adata.obs[[sample_col, "malignant_binary"]].copy()
meta["malignant_binary"] = meta["malignant_binary"].astype(int)

meta["pseudobulk_id"] = (
    meta[sample_col].astype(str) + "_"
    + meta["malignant_binary"].astype(str)
)

print("Aggregating into pseudobulk samples...")

# ============================
# Aggregate to Pseudobulk (mean expression per sample×condition)
# ============================

pseudobulk_expr = expr_df.groupby(meta["pseudobulk_id"]).mean()

pseudobulk_meta = (
    meta
    .drop_duplicates("pseudobulk_id")
    .set_index("pseudobulk_id")
)

# Keep only malignant_binary as group label
pseudobulk_meta = pseudobulk_meta.loc[pseudobulk_expr.index]
pseudobulk_meta = pseudobulk_meta[["malignant_binary"]]

print("Pseudobulk matrix shape:", pseudobulk_expr.shape)
print("Pseudobulk metadata shape:", pseudobulk_meta.shape)

# ============================
# Create Pseudobulk AnnData
# ============================

pb_adata = sc.AnnData(
    X=pseudobulk_expr.values,
    obs=pseudobulk_meta.copy(),
    var=pd.DataFrame(index=pseudobulk_expr.columns),
)

# Make group labels categorical and nicer
pb_adata.obs["malignant_binary"] = pb_adata.obs["malignant_binary"].astype(int)
pb_adata.obs["group"] = pb_adata.obs["malignant_binary"].map({0: "non_malignant", 1: "malignant"})
pb_adata.obs["group"] = pb_adata.obs["group"].astype("category")

print("Running Scanpy rank_genes_groups on pseudobulk...")

sc.tl.rank_genes_groups(
    pb_adata,
    groupby="group",
    method="wilcoxon",
    use_raw=False
)

# ============================
# Extract DE Table
# ============================

de_df = sc.get.rank_genes_groups_df(pb_adata, group=None)
# This gives one big table with columns: group, names, scores, pvals, pvals_adj, logfoldchanges

de_df = de_df.rename(
    columns={
        "names": "gene",
        "logfoldchanges": "log2FoldChange",
        "pvals_adj": "padj",
    }
)

# Define significance
de_df["significant"] = (
    (de_df["padj"] < 0.05) &
    (np.abs(de_df["log2FoldChange"]) > 1)
)

# Save full + sig
de_df.to_csv(
    f"{OUTPUT_DIR}/de_genes_malignant_vs_nonmalignant_PSEUDOBULK_SCANPY_FULL.csv",
    index=False
)

sig_df = de_df[de_df["significant"]].sort_values("padj")
sig_df.to_csv(
    f"{OUTPUT_DIR}/de_genes_malignant_vs_nonmalignant_PSEUDOBULK_SCANPY_SIG.csv",
    index=False
)

print("Top significant rows:")
print(sig_df.head(10))

# ============================
# Volcano Plot (malignant vs non-malignant)
# ============================

# For volcano, pick malignant vs rest rows
malignant_rows = de_df[de_df["group"] == "malignant"].copy()

plt.figure(figsize=(8, 7))
plt.scatter(
    malignant_rows["log2FoldChange"],
    -np.log10(malignant_rows["padj"]),
    c=malignant_rows["significant"],
    alpha=0.6,
)

plt.axvline(1, linestyle="--")
plt.axvline(-1, linestyle="--")
plt.axhline(-np.log10(0.05), linestyle="--")

plt.xlabel("Log2 Fold Change (malignant vs non_malignant)")
plt.ylabel("-Log10 Adjusted P-value")
plt.title("Malignant vs Non-Malignant (Pseudobulk • Scanpy Wilcoxon)")

plt.savefig(
    f"{FIG_DIR}/volcano_malignant_vs_nonmalignant_PSEUDOBULK_SCANPY.png",
    dpi=300,
)
plt.close()

# ============================
# Top 100 Biomarker Genes (malignant up/down)
# ============================

top100 = malignant_rows[malignant_rows["significant"]].sort_values("padj").head(100)
top100.to_csv(
    f"{OUTPUT_DIR}/top100_biomarker_genes_PSEUDOBULK_SCANPY.csv",
    index=False,
)

