
# no velocity, only PAGA 


import os
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

INPUT_FILE = "data/processed/adata_clustered.h5ad"
FIG_DIR = "results/figures"
OUTPUT_FILE = "data/processed/adata_paga_dpt.h5ad"

os.makedirs(FIG_DIR, exist_ok=True)
print("Loading clustered AnnData...")
adata = sc.read_h5ad(INPUT_FILE)
print("Data shape:", adata.shape)


# recreating the malignant binary because the data does not have this structure


POSSIBLE_COLUMNS = ["cell_type", "malignancy", "tumor_status", "annotation"]

label_col = None
for c in POSSIBLE_COLUMNS:
    if c in adata.obs.columns:
        label_col = c
        break

if label_col is None:
    raise ValueError(
        "No malignant/non-malignant column found.\n"
        "Expected one of: cell_type, malignancy, tumor_status, annotation"
    )

adata.obs["malignant_binary"] = (
    adata.obs[label_col]
    .astype(str)
    .str.lower()
    .str.contains("malignant|tumor|cancer")
)

print("Malignant counts:")
print(adata.obs["malignant_binary"].value_counts())

print("Running PAGA topology...")
sc.tl.paga(adata, groups="leiden")

sc.pl.paga(adata, show=False)
plt.savefig(f"{FIG_DIR}/paga_graph.png", dpi=300, bbox_inches="tight")
plt.close()


print("Computing diffusion pseudotime...")

sc.tl.diffmap(adata)


root_cluster = (
    adata.obs
    .query("malignant_binary == False")["leiden"]
    .value_counts()
    .idxmax()
)

print("Using root cluster:", root_cluster)

# Set root cell index
adata.uns["iroot"] = np.flatnonzero(
    adata.obs["leiden"] == root_cluster
)[0]

# Run DPT
sc.tl.dpt(adata)

# Plot DPT
sc.pl.umap(
    adata,
    color="dpt_pseudotime",
    cmap="viridis",
    show=False
)
plt.savefig(f"{FIG_DIR}/dpt_pseudotime.png", dpi=300, bbox_inches="tight")
plt.close()


sc.pl.umap(
    adata,
    color="leiden",
    edges=True,
    show=False
)
plt.savefig(f"{FIG_DIR}/paga_on_umap.png", dpi=300, bbox_inches="tight")
plt.close()


adata.write(OUTPUT_FILE)


