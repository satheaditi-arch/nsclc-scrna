#!/usr/bin/env python3

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc


def main():
    # ==============================
    # 0. Settings
    # ==============================
    in_file = "high_res.h5ad"
    subsampled_file = "high_res_subsampled.h5ad"
    out_file = "adata_preprocessed.h5ad"

    # Number of cells to keep for local analysis
    # (you can lower this to 20_000 or 10_000 if you still hit memory issues)
    n_keep_target = 30_000

    np.random.seed(0)

    # ==============================
    # 1. Subsample in backed mode
    # ==============================
    print(f"Reading {in_file} in backed='r' mode...")
    adata_b = sc.read_h5ad(in_file, backed="r")
    print("Full dataset shape (backed):", adata_b.shape)

    n_total = adata_b.n_obs
    n_keep = min(n_keep_target, n_total)
    print(f"Subsampling {n_keep} of {n_total} cells...")

    idx = np.random.choice(n_total, n_keep, replace=False)

    # This loads only selected cells into memory
    adata_small = adata_b[idx, :].to_memory()
    adata_b.file.close()
    print("Subsampled in-memory shape:", adata_small.shape)

    # (Optional) Save subsampled file
    print(f"Writing subsampled AnnData to {subsampled_file}...")
    adata_small.write(subsampled_file)

    # Work with a copy so we don’t accidentally modify views
    adata = adata_small.copy()
    del adata_small
    gc.collect()

    # ==============================
    # 2. Gene name cleanup + mt genes
    # ==============================
    print("Making var_names unique...")
    adata.var_names_make_unique()

    print("Flagging mitochondrial genes...")
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")

    # ==============================
    # 3. QC Metrics
    # ==============================
    print("Calculating QC metrics...")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        inplace=True,
        percent_top=None,  # avoid extra large intermediate arrays
    )

    # ==============================
    # 4. QC Filtering
    # ==============================
    print("Applying QC filters...")
    # Adjust thresholds if needed
    qc_mask = (
        (adata.obs["total_counts"] > 800)
        & (adata.obs["n_genes_by_counts"] > 400)
        & (adata.obs["pct_counts_mt"] < 15)
    )

    print(f"Cells before QC: {adata.n_obs}")
    adata = adata[qc_mask, :].copy()
    print(f"Cells after QC:  {adata.n_obs}, Genes: {adata.n_vars}")

    gc.collect()

    # ==============================
    # 5. Normalization + log1p
    # ==============================
    print("Normalizing total counts and log1p-transforming...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ==============================
    # 6. Highly variable genes
    # ==============================
    print("Selecting highly variable genes (Seurat v3, 2000 genes)...")
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=2000,
    )

    hvgs = adata.var["highly_variable"].sum()
    print(f"Number of HVGs selected: {hvgs}")

    adata = adata[:, adata.var["highly_variable"]].copy()
    print("Shape after HVG selection:", adata.shape)

    gc.collect()

    # ==============================
    # 7. Save preprocessed AnnData
    # ==============================
    print(f"Writing preprocessed AnnData to {out_file}...")
    adata.write(out_file)

    print("Done!")


if __name__ == "__main__":
    main()
