import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import scanpy as sc

# ============================================================================
# CONFIGURATION - Update these paths and parameters for your environment
# ============================================================================
DATA_DIR = '/nfs/blanche/share/han/scalebio_pmbcs'
INPUT_TRAIN_FILE = f'{DATA_DIR}/adata_train.h5ad'
OUTPUT_DIR = f'{DATA_DIR}/phase1b_outputs'

# Simulation parameters
N_TRAIN_SAMPLES = 10000
N_VAL_SAMPLES = 2000
CELLS_PER_SAMPLE = 1000
DIRICHLET_ALPHA = 1.0
TRAIN_SEED = 42
VAL_SEED = 123

# DataLoader parameters
BATCH_SIZE = 128
NUM_WORKERS = 0

print("=" * 70)
print("PHASE I-B: SIMULATION AND DATA MODULES")
print("=" * 70) 

### 1. SIMULATION FUNCTION

def create_simulated_bulk(adata_train, n_samples, cells_per_sample=1000, alpha=1.0, seed=None):
    """
    Generate simulated bulk RNA-seq data by sampling and summing single cells.

    Parameters:
    -----------
    adata_train: AnnData
        Training single-cell with raw counts in .layers['counts']
    n_samples: int
        Number of bulk samples to generate
    cells_per_sample: int
        Number of cells to mix in each bulk sample
    alpha: float
        Dirichlet concentration parameter (lower = more sparse proportions)
    seed: int, optional
        Random seed for reproducibility

    Returns:
    --------
    bulk_data: np.ndarray
        Array of shape (n_samples, n_genes) with simulated bulk expression
    true_proportions: np.ndarray
        Array of shape (n_samples, n_cell_types) with ground truth proportions
    """
    if seed is not None:
        np.random.seed(seed)

    # Get parameters
    n_cell_types = adata_train.obs['cell_type_int'].nunique()
    n_genes = adata_train.n_vars

    print(f"\n[Simulation Parameters]")
    print(f" Samples to generate: {n_samples:,}")
    print(f" Cells per sample: {cells_per_sample:,}")
    print(f" Cell types: {n_cell_types}")
    print(f" Genes: {n_genes:,}")
    print(f" Dirichlet alpha: {alpha}")

    # Initialize output arrays
    bulk_data = np.zeros((n_samples, n_genes), dtype=np.float32)
    true_proportions = np.zeros((n_samples, n_cell_types), dtype=np.float32)

    # Get raw counts (convert sparse to dense if needed)
    raw_counts = adata_train.layers['counts']
    if hasattr(raw_counts, 'toarray'):
        raw_counts = raw_counts.toarray()

    # Get cell type labels
    cell_type_labels = adata_train.obs['cell_type_int'].values

    # Pre-compute cell indices for each cell type (for faster sampling)
    cell_type_indices = {}
    for ct in range(n_cell_types):
        cell_type_indices[ct] = np.where(cell_type_labels == ct)[0]

    print(f"\n[Generating {n_samples:,} simulated bulk samples...]")

    for i in range(n_samples):
        if (i + 1) % 1000 == 0:
            print(f" Generated {i + 1:,} / {n_samples:,} samples...")
        
        # Generate random ground-truth proportions using Dirichlet
        P_true = np.random.dirichlet(alpha=np.ones(n_cell_types) * alpha)
        true_proportions[i] = P_true

        # Calculate number of cells to sample from each type
        n_cells_per_type = np.round(P_true * cells_per_sample).astype(int)

        # Ensure we have at least cells_per_sample total
        # (rounding might cause slight deviation)
        diff = cells_per_sample - n_cells_per_type.sum()
        if diff > 0:
            # Add extra cells to most abundant type
            max_idx = np.argmax(P_true)
            n_cells_per_type[max_idx] += diff
        elif diff < 0:
            # Remove cells from most abundant type
            max_idx = np.argmax(P_true)
            n_cells_per_type[max_idx] += diff

        # Sample cells and sum their counts
        B_sim = np.zeros(n_genes, dtype=np.float32)

        for ct in range(n_cell_types):
            n_cells = n_cells_per_type[ct]
            if n_cells > 0:
                # Sample with replacement from this cell type
                available_cells = cell_type_indices[ct]
                sampled_idx = np.random.choice(
                    available_cells,
                    size=n_cells,
                    replace=True
                )

                # Sum the raw counts
                B_sim += raw_counts[sampled_idx].sum(axis=0)

        bulk_data[i] = B_sim

    print(f" Generated {n_samples:,} simulated bulk samples")
    print(f"\nBulk data statistics (raw counts)")
    print(f" Shape: {bulk_data.shape}")
    print(f" Mean total counts: {bulk_data.sum(axis=1).mean():.0f}")
    print(f" Min total counts: {bulk_data.sum(axis=1).min():.0f}")
    print(f" Max total counts: {bulk_data.sum(axis=1).max():.0f}")

    # CRITICAL: Normalize bulk data the same way as single-cell data
    # to avoid scale mismatch and NaN losses
    print(f"\nNormalizing bulk data (target_sum=1e4, log1p)...")

    # Total count normalization (CPM-like)
    bulk_data_normalized = np.zeros_like(bulk_data, dtype=np.float32)
    for i in range(n_samples):
        total_counts = bulk_data[i].sum()
        if total_counts > 0:
            bulk_data_normalized[i] = (bulk_data[i] / total_counts) * 1e4
        else:
            bulk_data_normalized[i] = bulk_data[i]

    # Log transformation
    bulk_data_normalized = np.log1p(bulk_data_normalized)

    print(f" Normalized bulk data range: [{bulk_data_normalized.min():.2f}, {bulk_data_normalized.max():.2f}]")
    print(f" Normalized bulk data mean: {bulk_data_normalized.mean():.2f}")

    return bulk_data_normalized, true_proportions


### 2. PYTORCH DATASET CLASS

class SingleCellBulkDataset(Dataset):
    """
    Custom PyTorch Dataset that serves both single-cell and bulk data.

    This dataset allows training with batches containing both:
    - Single-cell data with cell type labels (for VAE reconstruction)
    - Simulated bulk data with ground truth proportions (for deconvolution)
    """

    def __init__(self, adata_train, bulk_data, true_proportions, 
                 use_normalized=True):
        """
        Parameters:
        -----------
        adata_train: AnnData
            Training single-cell data
        bulk_data: np.ndarray
            Simulated bulk expression data (n_bulk_samples, n_genes)
        true_proportions: np.ndarray
            Ground truth cell type proportions (n_bulk_samples, n_cell_types)
        use_normalized: bool
            If True, use normalized data (adata.X), else use raw counts
        """
        self.adata_train = adata_train
        self.bulk_data = torch.FloatTensor(bulk_data)
        self.true_proportions = torch.FloatTensor(true_proportions)
        self.use_normalized = use_normalized

        # Get single-cell data
        if use_normalized:
            # Use normalized, log-transformed data
            sc_data = adata_train.X
        else:
            # Use raw counts
            sc_data = adata_train.layers['counts']

        # Convert to dense array if sparse
        if hasattr(sc_data, 'toarray'):
            sc_data = sc_data.toarray()

        self.sc_data = torch.FloatTensor(sc_data)
        self.sc_labels = torch.LongTensor(adata_train.obs['cell_type_int'].values)

        self.n_sc = len(self.sc_data)
        self.n_bulk = len(self.bulk_data)

        # Validate data for NaN/Inf
        if torch.isnan(self.sc_data).any():
            raise ValueError("NaN detected in single-cell data!")
        if torch.isnan(self.bulk_data).any():
            raise ValueError("NaN detected in bulk data!")
        if torch.isnan(self.true_proportions).any():
            raise ValueError("NaN detected in proportions!")
        if torch.isinf(self.sc_data).any():
            raise ValueError("Inf detected in single-cell data!")
        if torch.isinf(self.bulk_data).any():
            raise ValueError("Inf detected in bulk data!")

    
    def __len__(self):
        # Return the maximum of the two dataset sizes
        # We'll use modulo to wrap around the smaller dataset
        return max(self.n_sc, self.n_bulk)
    
    def __getitem__(self, idx):
        # Use modulo to wrap indices for both datasets
        idx_sc = idx % self.n_sc
        idx_bulk = idx % self.n_bulk

        return {
            'sc_data': self.sc_data[idx_sc],
            'sc_label': self.sc_labels[idx_sc],
            'bulk_data': self.bulk_data[idx_bulk],
            'bulk_prop': self.true_proportions[idx_bulk]
        }
    
### 3. GENERATE SIMULATED BULK DATA

print("\n" + "=" * 70)
print("GENERATING SIMULATED BULK DATA")
print("=" * 70)

# Load saved train data
adata_train = sc.read_h5ad(INPUT_TRAIN_FILE)

# Generate training bulk data
bulk_train, props_train = create_simulated_bulk(
    adata_train,
    n_samples=N_TRAIN_SAMPLES,
    cells_per_sample=CELLS_PER_SAMPLE,
    alpha=DIRICHLET_ALPHA,
    seed=TRAIN_SEED
)

# Generate validation bulk data (smaller, separate seed)
bulk_val, props_val = create_simulated_bulk(
    adata_train,
    n_samples=N_VAL_SAMPLES,
    cells_per_sample=CELLS_PER_SAMPLE,
    alpha=DIRICHLET_ALPHA,
    seed=VAL_SEED
)

### 4. CREATE PYTORCH DATALOADERS

print("\n" + "=" * 70)
print("CREATING PYTORCH DATALOADERS")
print("=" * 70)

# Create datasets
train_dataset = SingleCellBulkDataset(
    adata_train,
    bulk_train,
    props_train,
    use_normalized=True
)

val_dataset = SingleCellBulkDataset(
    adata_train,
    bulk_val,
    props_val,
    use_normalized=True
)

print(f"\nTraining dataset:")
print(f"  Single-cell samples: {train_dataset.n_sc:,}")
print(f"  Bulk samples: {train_dataset.n_bulk:,}")
print(f"  Total __len__: {len(train_dataset):,}")

print(f"\nValidation dataset:")
print(f"  Single-cell samples: {val_dataset.n_sc:,}")
print(f"  Bulk samples: {val_dataset.n_bulk:,}")
print(f"  Total __len__: {len(val_dataset):,}")

# Create dataloaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,  # Set to 0 to avoid multiprocessing issues
    pin_memory=True  # Faster data transfer to GPU
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"\nDataLoader settings:")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Training batches per epoch: {len(train_dataloader):,}")
print(f"  Validation batches: {len(val_dataloader):,}")

### 5. TEST THE DATALOADER

print("\n" + "=" * 70)
print("TESTING DATALOADER")
print("=" * 70)

# Get one batch to verify everything works
test_batch = next(iter(train_dataloader))

print(f"\nBatch contents:")
print(f"  sc_data shape: {test_batch['sc_data'].shape}")
print(f"  sc_label shape: {test_batch['sc_label'].shape}")
print(f"  bulk_data shape: {test_batch['bulk_data'].shape}")
print(f"  bulk_prop shape: {test_batch['bulk_prop'].shape}")

print(f"\nSample values:")
print(f"  SC data (first cell, first 5 genes): {test_batch['sc_data'][0, :5]}")
print(f"  SC label (first cell): {test_batch['sc_label'][0].item()}")
print(f"  Bulk data (first sample, first 5 genes): {test_batch['bulk_data'][0, :5]}")
print(f"  Bulk proportions (first sample): {test_batch['bulk_prop'][0]}")
print(f"  Bulk proportions sum: {test_batch['bulk_prop'][0].sum():.4f}")


### 6. SAVE EVERYTHING

print("\n" + "=" * 70)
print("SAVING SIMULATION DATA")
print("=" * 70)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save bulk data
np.save(f'{OUTPUT_DIR}/bulk_train.npy', bulk_train)
np.save(f'{OUTPUT_DIR}/props_train.npy', props_train)
np.save(f'{OUTPUT_DIR}/bulk_val.npy', bulk_val)
np.save(f'{OUTPUT_DIR}/props_val.npy', props_val)

print(f"✓ Saved bulk training data: {bulk_train.shape}")
print(f"✓ Saved bulk validation data: {bulk_val.shape}")

# Save a summary
summary = {
    'n_train_samples': N_TRAIN_SAMPLES,
    'n_val_samples': N_VAL_SAMPLES,
    'cells_per_sample': CELLS_PER_SAMPLE,
    'alpha': DIRICHLET_ALPHA,
    'n_genes': bulk_train.shape[1],
    'n_cell_types': props_train.shape[1]
}

with open(f'{OUTPUT_DIR}/simulation_summary.pkl', 'wb') as f:
    pickle.dump(summary, f)

print(f"✓ Saved simulation summary to {OUTPUT_DIR}")

print("\n" + "=" * 70)
print("PHASE I-B COMPLETE")
print("=" * 70)
print(f"✓ Simulation function created")
print(f"✓ Generated {N_TRAIN_SAMPLES:,} training bulk samples")
print(f"✓ Generated {N_VAL_SAMPLES:,} validation bulk samples")
print(f"✓ Custom PyTorch Dataset class created")
print(f"✓ DataLoaders ready for training")
print("=" * 70)