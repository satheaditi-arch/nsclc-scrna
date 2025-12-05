
import os
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
import matplotlib.pyplot as plt

AD_FILE = "data/processed/adata_clustered.h5ad"
DE_FILE = "results/de_genes/top100_biomarker_genes_PSEUDOBULK_SCANPY.csv"

OUT_DIR = "results/regulatory"
FIG_DIR = "results/figures"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("Loading clustered AnnData...")
adata = sc.read_h5ad(AD_FILE)
print("adata shape:", adata.shape)

print("Loading top biomarker genes...")
de = pd.read_csv(DE_FILE)

possible_gene_cols = ["gene", "Gene", "symbol", "Symbol"]
gene_col = next((c for c in possible_gene_cols if c in de.columns), None)
if gene_col is None:
    raise ValueError(f"❌ No gene column found in DE file. Columns: {list(de.columns)}")


N_TOP = min(50, de.shape[0])
genes = (
    de[gene_col]
    .dropna()
    .astype(str)
    .str.strip()
    .head(N_TOP)
    .tolist()
)

print(f"Requested top {N_TOP} genes; got {len(genes)} DE genes.")

# Keep only genes present in adata
genes_in_adata = [g for g in genes if g in adata.var_names]
print(f"Genes present in AnnData: {len(genes_in_adata)}")

if len(genes_in_adata) < 5:
    raise ValueError("❌ Too few genes found in AnnData to build a network.")


if "malignant_binary" in adata.obs.columns:
    print("Using malignant cells only (malignant_binary == 1)...")
    adata_sub = adata[adata.obs["malignant_binary"] == 1, :].copy()
else:
    print("⚠ malignant_binary not found in adata.obs; using ALL cells.")
    adata_sub = adata.copy()

print("Subset shape:", adata_sub.shape)


adata_sub = adata_sub[:, genes_in_adata]


X = adata_sub.X
if not isinstance(X, np.ndarray):
    X = X.toarray()

expr_df = pd.DataFrame(
    X,
    columns=genes_in_adata,
    index=adata_sub.obs_names
)

print("Expression matrix for network:", expr_df.shape)

print("Computing gene–gene correlation matrix...")
corr = expr_df.corr(method="pearson")

corr_path = os.path.join(OUT_DIR, "regnet_correlation_matrix.tsv")
corr.to_csv(corr_path, sep="\t")
print("Saved correlation matrix to:", corr_path)


plt.figure(figsize=(10, 8))
im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, label="Pearson r")
plt.xticks(range(len(genes_in_adata)), genes_in_adata, rotation=90)
plt.yticks(range(len(genes_in_adata)), genes_in_adata)
plt.title("Co-expression network (malignant biomarkers)")
plt.tight_layout()

heatmap_path = os.path.join(FIG_DIR, "regulatory_corr_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()
print("Saved correlation heatmap to:", heatmap_path)


THRESH = 0.6

print(f"Building network with |r| ≥ {THRESH} ...")

edges = []
for i, g1 in enumerate(genes_in_adata):
    for j, g2 in enumerate(genes_in_adata):
        if j <= i:
            continue
        r = corr.loc[g1, g2]
        if np.isfinite(r) and abs(r) >= THRESH:
            edges.append((g1, g2, r))

edge_df = pd.DataFrame(edges, columns=["gene1", "gene2", "correlation"])
edge_path = os.path.join(OUT_DIR, "regnet_edges.tsv")
edge_df.to_csv(edge_path, sep="\t", index=False)
print(f"Saved {edge_df.shape[0]} edges to:", edge_path)

if edge_df.empty:
    print("⚠ No edges passed the correlation threshold. "
          "Try lowering THRESH in the script if needed.")
    # still continue to save an empty node table
    node_df = pd.DataFrame({"gene": genes_in_adata, "degree": 0})
    node_path = os.path.join(OUT_DIR, "regnet_nodes.tsv")
    node_df.to_csv(node_path, sep="\t", index=False)
    print("Saved node table with zero-degree nodes to:", node_path)
    raise SystemExit("No network to visualize at this threshold.")

G = nx.Graph()
for _, row in edge_df.iterrows():
    G.add_edge(row["gene1"], row["gene2"], weight=row["correlation"])

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Degree centrality
deg = dict(G.degree())

node_df = pd.DataFrame({
    "gene": list(G.nodes()),
    "degree": [deg[g] for g in G.nodes()]
})

node_path = os.path.join(OUT_DIR, "regnet_nodes.tsv")
node_df.to_csv(node_path, sep="\t", index=False)
print("Saved node table to:", node_path)

print("Drawing regulatory (co-expression) network...")

plt.figure(figsize=(9, 9))

pos = nx.spring_layout(G, seed=42)

node_sizes = [80 + 40 * deg[g] for g in G.nodes()]

nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    alpha=0.85
)
nx.draw_networkx_edges(
    G, pos,
    alpha=0.4,
    width=0.8
)

top_hubs = node_df.sort_values("degree", ascending=False).head(10)["gene"].tolist()
labels = {g: g for g in G.nodes() if g in top_hubs}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

plt.title("Co-expression 'Regulatory' Network — Malignant Biomarkers")
plt.axis("off")
net_fig = os.path.join(FIG_DIR, "regulatory_network.png")
plt.savefig(net_fig, dpi=300, bbox_inches="tight")
plt.close()

