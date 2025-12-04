# ============================================================
# PCA + UMAP + Leiden Clustering + Marker Gene Detection
# ============================================================

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os

INPUT_FILE = "adata_preprocessed.h5ad"
OUTPUT_DIR = "results/figures"
MARKER_DIR = "results/de_genes"
OUTPUT_ADATA = "data/processed/adata_clustered.h5ad"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MARKER_DIR, exist_ok=True)

print("Loading preprocessed AnnData object...")
adata = sc.read_h5ad(INPUT_FILE)
print("Data shape:", adata.shape)

print("Running PCA...")
sc.tl.pca(adata, svd_solver="arpack")

# Save PCA variance plot
sc.pl.pca_variance_ratio(adata, log=True, show=False)
plt.savefig(f"{OUTPUT_DIR}/pca_variance_ratio.png", dpi=300, bbox_inches="tight")
plt.close()

print("Computing neighborhood graph...")
sc.pp.neighbors(adata, use_rep="X_pca")
sc.tl.umap(adata, min_dist=0.3)
sc.pl.umap(adata, show=False)
plt.savefig(f"{OUTPUT_DIR}/umap_overall.png", dpi=300, bbox_inches="tight")
plt.close()

#leiden clustering
print("Running Leiden clustering...")
sc.tl.leiden(
    adata,
    resolution=0.6,
    key_added="leiden"
)

sc.pl.umap(
    adata,
    color="leiden",
    legend_loc="on data",
    show=False
)
plt.savefig(f"{OUTPUT_DIR}/umap_leiden_clusters.png", dpi=300, bbox_inches="tight")
plt.close()

#marker genes 
print("Computing marker genes per cluster...")
sc.tl.rank_genes_groups(
    adata,
    groupby="leiden",
    method="wilcoxon"
)

# Save top marker genes per cluster
marker_df = sc.get.rank_genes_groups_df(adata, group=None)
marker_df.to_csv(
    f"{MARKER_DIR}/leiden_marker_genes.csv",
    index=False
)

# Plot top markers
sc.pl.rank_genes_groups(
    adata,
    n_genes=20,
    sharey=False,
    show=False
)
plt.savefig(f"{OUTPUT_DIR}/marker_genes_leiden.png", dpi=300, bbox_inches="tight")
plt.close()

#clustering 
print("Saving clustered AnnData...")
adata.write(OUTPUT_ADATA)

print("============================================")
print("CLUSTERING + UMAP PIPELINE COMPLETE")
print("Saved:")
print("- UMAP plots")
print("- Marker gene tables")
print("- Clustered AnnData object")
print("============================================")
