
#top 100 genes simply do NOT reach FDR < 0.05 for GO/KEGG

#ZERO ENRICHED PATHWAYS 
#No pathways reached FDR < 0.05 using strict pseudobulk differential expression and multiple-testing correction. For visualization, a relaxed FDR threshold of 0.20 was used to explore trend-level biological enrichment

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gseapy as gp

INPUT_FILE = "results/de_genes/top100_biomarker_genes_PSEUDOBULK_SCANPY.csv"
OUT_DIR = "results/pathways"
FIG_DIR = "results/figures"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("Loading top biomarker genes...")

df = pd.read_csv(INPUT_FILE)

possible_gene_cols = ["gene", "Gene", "symbol", "Symbol"]
gene_col = None

for c in possible_gene_cols:
    if c in df.columns:
        gene_col = c
        break

if gene_col is None:
    raise ValueError(
        f" Could not detect gene column.\n"
        f"Available columns: {list(df.columns)}\n"
        "Rename your gene column to 'gene'."
    )

gene_list = df[gene_col].dropna().astype(str).tolist()
print(f"Loaded {len(gene_list)} genes.")
print("Running GO Biological Process enrichment...")

go_bp = gp.enrichr(
    gene_list=gene_list,
    gene_sets="GO_Biological_Process_2021",
    organism="Human",
    outdir=OUT_DIR,
    cutoff=0.20,
    no_plot=True     
)

go_bp_res = None
if hasattr(go_bp, "results") and go_bp.results is not None and not go_bp.results.empty:
    go_bp_res = go_bp.results
    go_bp_res.to_csv(f"{OUT_DIR}/go_biological_process.csv", index=False)
    print(" GO results saved.")
else:
    print("No GO terms enriched at FDR < 0.20")

print("Running KEGG pathway enrichment...")

kegg = gp.enrichr(
    gene_list=gene_list,
    gene_sets="KEGG_2021_Human",
    organism="Human",
    outdir=OUT_DIR,
    cutoff=0.20,
    no_plot=True     
)

kegg_res = None
if hasattr(kegg, "results") and kegg.results is not None and not kegg.results.empty:
    kegg_res = kegg.results
    kegg_res.to_csv(f"{OUT_DIR}/kegg_pathways.csv", index=False)
    print("KEGG results saved.")
else:
    print(" No KEGG pathways enriched at FDR < 0.20")


if go_bp_res is not None:
    print("Saving GO enrichment plot...")

    top_go = go_bp_res.sort_values("Adjusted P-value").head(15)

    plt.figure(figsize=(7, 6))
    plt.barh(
        top_go["Term"],
        -np.log10(top_go["Adjusted P-value"])
    )
    plt.xlabel("-log10(FDR)")
    plt.title("Top GO Biological Processes (Relaxed FDR)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/go_biological_process.png", dpi=300)
    plt.close()
else:
    print(" Skipping GO plot (no enriched terms).")

if kegg_res is not None:
    print("Saving KEGG pathway plot...")

    top_kegg = kegg_res.sort_values("Adjusted P-value").head(15)

    plt.figure(figsize=(7, 6))
    plt.barh(
        top_kegg["Term"],
        -np.log10(top_kegg["Adjusted P-value"])
    )
    plt.xlabel("-log10(FDR)")
    plt.title("Top KEGG Pathways (Relaxed FDR)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/kegg_pathways.png", dpi=300)
    plt.close()
else:
    print("Skipping KEGG plot (no enriched terms).")