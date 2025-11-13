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

# ============================================================================
# CONFIGURATION - Update these paths for your environment
# ============================================================================
DATA_DIR = '/nfs/blanche/share/han/scalebio_pmbcs'
INPUT_FILE = f'{DATA_DIR}/adata_full.h5ad'
OUTPUT_HVG_FILE = f'{DATA_DIR}/adata_hvg.h5ad'
OUTPUT_TRAIN_FILE = f'{DATA_DIR}/adata_train.h5ad'
OUTPUT_TEST_FILE = f'{DATA_DIR}/adata_test.h5ad'

# Processing parameters
N_TOP_GENES = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# DATA LOADING
# ============================================================================

# Load preprocessed data
# Note: This assumes data has already been downloaded from CellxGene census,
# normalized (target_sum=1e4), log-transformed (log1p), and saved with raw counts
# in layers['counts']. See documentation for initial data retrieval steps.
adata = sc.read_h5ad(INPUT_FILE)
 
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
    n_top_genes=N_TOP_GENES,
    subset=False,
    flavor='seurat_v3',
    layer='counts'
)

print(f"Highly variable genes identified: {adata.var['highly_variable'].sum():,}")

adata_hvg = adata[:, adata.var['highly_variable']].copy()
print(f"Filtered shape: {adata_hvg.shape}")

# Save HVG-filtered dataset
adata_hvg.write(OUTPUT_HVG_FILE)
print(f"✓ Saved: {OUTPUT_HVG_FILE}")

### STEP 3: Split data ###
print(f"\n[Step 3] Splitting data into train and test sets...")

n_cells = adata_hvg.n_obs
cell_indices = np.arange(n_cells)

train_idx, test_idx = train_test_split(
    cell_indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
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
adata_train.write(OUTPUT_TRAIN_FILE)
print(f"✓ Saved: {OUTPUT_TRAIN_FILE}")

# Save Test set
adata_test.write(OUTPUT_TEST_FILE)
print(f"✓ Saved: {OUTPUT_TEST_FILE}")
