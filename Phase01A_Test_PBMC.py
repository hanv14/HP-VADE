import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cellxgene_census
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

### List all columns in metadata
# with cellxgene_census.open_soma(census_version="2025-01-30") as census:
#     obs = census["census_data"]["homo_sapiens"].obs
    
#     # Print all available column names
#     print("Available columns:")
#     print(obs.keys())
    
#     # Or get schema info
#     print("\nColumn schema:")
#     for col_name in obs.keys():
#         print(f"  - {col_name}")

# Column schema:
#   - soma_joinid
#   - dataset_id
#   - assay
#   - assay_ontology_term_id
#   - cell_type
#   - cell_type_ontology_term_id
#   - development_stage
#   - development_stage_ontology_term_id
#   - disease
#   - disease_ontology_term_id
#   - donor_id
#   - is_primary_data
#   - observation_joinid
#   - self_reported_ethnicity
#   - self_reported_ethnicity_ontology_term_id
#   - sex
#   - sex_ontology_term_id
#   - suspension_type
#   - tissue
#   - tissue_ontology_term_id
#   - tissue_type
#   - tissue_general
#   - tissue_general_ontology_term_id
#   - raw_sum
#   - nnz
#   - raw_mean_nnz
#   - raw_variance_nnz
#   - n_measured_vars


# ### STEP 1: Get the PBMCs data from CellxGene ###
# # The dataset id is 2c820d53-cbd7-4e0a-be7a-a0ad1989a98f
# with cellxgene_census.open_soma(census_version="2025-01-30") as census:
#     adata = cellxgene_census.get_anndata(
#         census=census,
#         organism="Homo sapiens",
#         obs_value_filter='dataset_id == "2c820d53-cbd7-4e0a-be7a-a0ad1989a98f"',
#         X_layers=["raw"]  # Get raw counts instead of normalized
#     )
    
#     print(f"\nData retrieved with raw counts:")
#     print(f"  Shape: {adata.shape}")
#     print(f"  X data type: {adata.X.dtype}")
#     print(f"  Sample values: {adata.X[0, :5].toarray() if hasattr(adata.X, 'toarray') else adata.X[0, :5]}")

# print(f"Initial shape: {adata.shape}")
# print(f"X data type: {adata.X.dtype}")
# print(f"Cell types: {adata.obs['cell_type'].nunique()} unique")

# # Store raw counts in layers for safe keeping
# adata.layers['counts'] = adata.X.copy()

# # Normalize and log-transform
# print("\nNormalizing and log-transforming data...")
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# print(f"After normalization - X data type:{adata.X.dtype}")

# # Save full dataset (all genes, normalized)
# adata.write('/nfs/blanche/share/han/scalebio_pmbcs/adata_full.h5ad')
# print("Saved: /nfs/blanche/share/han/scalebio_pmbcs/adata_full.h5ad")

#### Done! Just need to reload ####

adata = sc.read_h5ad('/nfs/blanche/share/han/scalebio_pmbcs/adata_full.h5ad')
 
# Create integer-mapped cell type labels
print("\nCreating integer-mapped cell type labels...")
cell_types = adata.obs['cell_type'].astype('category')
cell_types = cell_types.cat.remove_unused_categories()
adata.obs['cell_type_int'] = cell_types.cat.codes.astype(int)

print(f"\nCell type mapping:")
cell_type_mapping = dict(enumerate(cell_types.cat.categories))
for code, label in cell_type_mapping.items():
    count = (adata.obs['cell_type_int'] == code).sum()
    print(f" {code}: {label:30s} ({count:,} cells)")


### STEP 2: Filter Genes ###
print(f"\n[Step 2] Filtering to highly variable genes...")
print(f"Initial gene count: {adata.n_vars:,}")

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=5000,
    subset=False,
    flavor='seurat_v3',
    layer='counts'
)

print(f"Highly variable genes identified: {adata.var['highly_variable'].sum():,}")

adata_hvg = adata[:, adata.var['highly_variable']].copy()
print(f"Filtered shape: {adata_hvg.shape}")

# Save HVG-filtered dataset
adata_hvg.write('/nfs/blanche/share/han/scalebio_pmbcs/adata_hvg.h5ad')
print("✓ Saved: /nfs/blanche/share/han/scalebio_pmbcs/adata_hvg.h5ad")

### STEP 3: Split data ###
print(f"\n[Step 3] Splitting data into train and test sets...")

n_cells = adata_hvg.n_obs
cell_indices = np.arange(n_cells)

train_idx, test_idx = train_test_split(
    cell_indices,
    test_size=0.2,
    random_state=42,
    stratify=adata_hvg.obs['cell_type_int']
)

adata_train = adata_hvg[train_idx, :].copy()
adata_test = adata_hvg[test_idx, :].copy()

print(f"\nTraining set: {adata_train.shape} ({adata_train.n_obs:,} cells)")
print(f"Test set: {adata_test.shape} ({adata_test.n_obs:,} cells)")

print("\nCell type distribution in training set:")
print(adata_train.obs['cell_type'].value_counts())

print("\nCell type distribution in test set:")
print(adata_test.obs['cell_type'].value_counts())

# Save Training set
adata_train.write('/nfs/blanche/share/han/scalebio_pmbcs/adata_train.h5ad')
print("✓ Saved: /nfs/blanche/share/han/scalebio_pmbcs/adata_train.h5ad")

# Save Test set
adata_test.write('/nfs/blanche/share/han/scalebio_pmbcs/adata_test.h5ad')
print("✓ Saved: /nfs/blanche/share/han/scalebio_pmbcs/adata_test.h5ad")
