import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gseapy as gp
import mygene

DE_FILE = "results/de_genes/de_genes_malignant_vs_nonmalignant_PSEUDOBULK_SCANPY_FULL.csv"
OUT_DIR = "results/pathways_gsea"
FIG_DIR = "results/figures"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


print("Loading DE results for GSEA (pseudobulk)...")
df = pd.read_csv(DE_FILE)

possible_gene_cols = ["gene", "Gene", "symbol", "Symbol"]
gene_col = next((c for c in possible_gene_cols if c in df.columns), None)
if gene_col is None:
    raise ValueError(f"No gene column found in: {list(df.columns)}")

rank_col = "stat" if "stat" in df.columns else "log2FoldChange"
if rank_col not in df.columns:
    raise ValueError("Need either 'stat' or 'log2FoldChange' column for ranking")

print(f"Using gene column: {gene_col}")
print(f"Using ranking metric: {rank_col}")

if df[gene_col].astype(str).str.startswith("ENSG").any():
    print("ENSEMBL IDs detected — converting to gene symbols for GSEA...")

    mg = mygene.MyGeneInfo()
    ens_ids = df[gene_col].dropna().unique().tolist()

    query = mg.querymany(
        ens_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species="human"
    )

    ens_to_symbol = {
        item["query"]: item.get("symbol")
        for item in query
        if "symbol" in item
    }

    df["gene_symbol"] = df[gene_col].map(ens_to_symbol)

    before = df.shape[0]
    df = df.dropna(subset=["gene_symbol"])
    after = df.shape[0]

    print(f" Converted {after}/{before} genes to SYMBOLS.")
    gene_col = "gene_symbol"


rnk = (
    df[[gene_col, rank_col]]
    .dropna()
    .rename(columns={gene_col: "gene", rank_col: "score"})
)

rnk["score_abs"] = rnk["score"].abs()
rnk = (
    rnk.sort_values("score_abs", ascending=False)
       .drop_duplicates("gene")
       .drop(columns="score_abs")
       .sort_values("score", ascending=False)
)

print(f"Ranked genes: {rnk.shape[0]}")

rnk_path = os.path.join(OUT_DIR, "malignant_vs_nonmalignant_preranked.rnk")
rnk.to_csv(rnk_path, sep="\t", header=False, index=False)
print(f"Saved ranked list → {rnk_path}")


def run_prerank_and_plot(rnk_df, gene_sets, label):

    print(f"\nRunning preranked GSEA for {label} ...")

    pre_res = gp.prerank(
        rnk=rnk_df,
        gene_sets=gene_sets,
        outdir=OUT_DIR,
        permutation_num=100,
        min_size=10,
        max_size=500,
        seed=42,
        verbose=True,
    )

    res = pre_res.res2d

    if res is None or res.empty:
        print(f"NO enriched pathways detected for {label}.")
        return None


    fdr_col = next((c for c in res.columns if "fdr" in c.lower()), None)
    if fdr_col is None:
        raise ValueError(f"No FDR column found in GSEA output columns: {res.columns}")
    table_path = os.path.join(OUT_DIR, f"gsea_{label.lower()}_full.csv")
    res.to_csv(table_path)
    print(f"Saved {label} table → {table_path}")

    top = res.sort_values(fdr_col).head(15)
    top_path = os.path.join(OUT_DIR, f"gsea_{label.lower()}_top15.csv")
    top.to_csv(top_path)
    print(f"Saved {label} top pathways → {top_path}")

  
    plt.figure(figsize=(8, 6))
    colors = -np.log10(top[fdr_col].astype(float).values.clip(min=1e-16))
    colors = colors / colors.max()


    nes_col = next((c for c in top.columns if c.lower() in ["nes", "normalized_enrichment_score", "enrichment_score", "es"]), None)

    if nes_col is None:
        raise ValueError(f"No NES column found in GSEA output columns: {top.columns}")

    plt.barh(top.index, top[nes_col], color=plt.cm.viridis(colors))

    plt.axvline(0, color="k", linewidth=0.8)
    plt.xlabel("Normalized Enrichment Score (NES)")
    plt.title(f"GSEA {label} — Malignant vs Non-Malignant")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, f"gsea_{label.lower()}_bar.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"Saved {label} barplot → {fig_path}")

    return res


hallmark_res = run_prerank_and_plot(
    rnk_df=rnk,
    gene_sets="MSigDB_Hallmark_2020",
    label="HALLMARK"
)

kegg_res = run_prerank_and_plot(
    rnk_df=rnk,
    gene_sets="KEGG_2021_Human",
    label="KEGG"
)

