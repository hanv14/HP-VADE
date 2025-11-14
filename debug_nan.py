"""
Comprehensive diagnostic to identify NaN source in HP-VADE training.
This script simulates the model initialization and first forward pass.
"""
import numpy as np
import torch
import torch.nn.functional as F
import sys

print("=" * 80)
print("HP-VADE NaN DIAGNOSTIC")
print("=" * 80)

# Step 1: Check bulk data files
print("\n[1] CHECKING BULK DATA FILES")
print("-" * 80)

try:
    bulk_train = np.load('/nfs/blanche/share/han/scalebio_pmbcs/phase1b_outputs/bulk_train.npy')
    prop_train = np.load('/nfs/blanche/share/han/scalebio_pmbcs/phase1b_outputs/proportions_train.npy')

    print(f"Bulk train shape: {bulk_train.shape}")
    print(f"Bulk train range: [{bulk_train.min():.2f}, {bulk_train.max():.2f}]")
    print(f"Bulk train mean: {bulk_train.mean():.2f}")
    print(f"Bulk train has NaN: {np.isnan(bulk_train).any()}")
    print(f"Bulk train has Inf: {np.isinf(bulk_train).any()}")

    print(f"\nProportions shape: {prop_train.shape}")
    print(f"Proportions range: [{prop_train.min():.4f}, {prop_train.max():.4f}]")
    print(f"Proportions row sums (first 5): {prop_train[:5].sum(axis=1)}")
    print(f"Proportions has NaN: {np.isnan(prop_train).any()}")

except Exception as e:
    print(f"ERROR loading bulk data: {e}")
    sys.exit(1)

# Step 2: Load and check single-cell data
print("\n[2] CHECKING SINGLE-CELL DATA")
print("-" * 80)

try:
    import scanpy as sc
    adata_train = sc.read_h5ad('/nfs/blanche/share/han/scalebio_pmbcs/adata_train.h5ad')

    X_data = adata_train.X
    if hasattr(X_data, 'toarray'):
        X_data = X_data.toarray()

    print(f"Single-cell .X shape: {X_data.shape}")
    print(f"Single-cell .X range: [{X_data.min():.4f}, {X_data.max():.4f}]")
    print(f"Single-cell .X mean: {X_data.mean():.4f}")
    print(f"Single-cell .X has NaN: {np.isnan(X_data).any()}")
    print(f"Single-cell .X has Inf: {np.isinf(X_data).any()}")

    # Check if 'counts' layer exists
    if 'counts' in adata_train.layers:
        counts = adata_train.layers['counts']
        if hasattr(counts, 'toarray'):
            counts = counts.toarray()
        print(f"\nCounts layer shape: {counts.shape}")
        print(f"Counts layer range: [{counts.min():.2f}, {counts.max():.2f}]")
        print(f"Counts layer mean: {counts.mean():.2f}")
        print(f"First cell sum: {counts[0].sum():.2f}")
    else:
        print("\n⚠️  No 'counts' layer found!")

    # Check cell types
    n_cell_types = adata_train.obs['cell_type_int'].nunique()
    print(f"\nNumber of cell types: {n_cell_types}")
    print(f"Cell type distribution:")
    for ct in range(n_cell_types):
        count = (adata_train.obs['cell_type_int'] == ct).sum()
        print(f"  Cell type {ct}: {count:,} cells")

except ImportError:
    print("⚠️  scanpy not available, skipping single-cell data check")
    print("   (This is OK, we can still debug using bulk data)")
    adata_train = None
except Exception as e:
    print(f"ERROR loading single-cell data: {e}")
    adata_train = None

# Step 3: Simulate signature matrix initialization
print("\n[3] SIMULATING SIGNATURE MATRIX INITIALIZATION")
print("-" * 80)

if adata_train is not None:
    try:
        cell_type_labels = adata_train.obs['cell_type_int'].values
        n_cell_types = adata_train.obs['cell_type_int'].nunique()
        n_genes = X_data.shape[1]

        signature_matrix = np.zeros((n_genes, n_cell_types), dtype=np.float32)

        print(f"Building signature matrix: ({n_genes}, {n_cell_types})")

        for ct in range(min(3, n_cell_types)):  # Check first 3 cell types
            ct_mask = cell_type_labels == ct
            ct_data = X_data[ct_mask]
            n_cells_ct = ct_mask.sum()

            print(f"\nCell type {ct}:")
            print(f"  Number of cells: {n_cells_ct:,}")
            print(f"  Data range: [{ct_data.min():.4f}, {ct_data.max():.4f}]")
            print(f"  Data mean: {ct_data.mean():.4f}")

            # Inverse log transform
            ct_counts = np.expm1(ct_data)
            print(f"  After expm1 range: [{ct_counts.min():.2f}, {ct_counts.max():.2f}]")
            print(f"  After expm1 mean: {ct_counts.mean():.2f}")

            # Aggregate
            aggregated_counts = ct_counts.sum(axis=0)
            print(f"  Aggregated sum: {aggregated_counts.sum():.2f}")
            print(f"  Aggregated mean: {aggregated_counts.mean():.2f}")

            # CPM normalize
            total = aggregated_counts.sum()
            if total > 0:
                ct_signature = (aggregated_counts / total) * 1e6
            else:
                ct_signature = np.zeros(n_genes, dtype=np.float32)

            signature_matrix[:, ct] = ct_signature

            print(f"  Signature range: [{ct_signature.min():.2f}, {ct_signature.max():.2f}]")
            print(f"  Signature mean: {ct_signature.mean():.2f}")
            print(f"  Signature sum: {ct_signature.sum():.2f}")
            print(f"  Has NaN: {np.isnan(ct_signature).any()}")
            print(f"  Has Inf: {np.isinf(ct_signature).any()}")

        print(f"\nFull signature matrix:")
        print(f"  Range: [{signature_matrix.min():.2f}, {signature_matrix.max():.2f}]")
        print(f"  Mean: {signature_matrix.mean():.2f}")
        print(f"  Has NaN: {np.isnan(signature_matrix).any()}")
        print(f"  Has Inf: {np.isinf(signature_matrix).any()}")

        # Check variance
        if n_cell_types > 1:
            variance_per_gene = signature_matrix.std(axis=1)
            print(f"  Mean variance across cell types: {variance_per_gene.mean():.2f}")

    except Exception as e:
        print(f"ERROR during signature initialization: {e}")
        import traceback
        traceback.print_exc()
        signature_matrix = None
else:
    print("⚠️  Single-cell data not available, using random signature matrix")
    n_genes = bulk_train.shape[1]
    n_cell_types = prop_train.shape[1]
    signature_matrix = np.random.randn(n_genes, n_cell_types).astype(np.float32) * 100 + 200

# Step 4: Simulate forward pass
print("\n[4] SIMULATING FORWARD PASS")
print("-" * 80)

if signature_matrix is not None:
    try:
        # Convert to torch
        S = torch.FloatTensor(signature_matrix)
        bulk_batch = torch.FloatTensor(bulk_train[:16])  # First batch
        prop_batch = torch.FloatTensor(prop_train[:16])

        print(f"Batch size: {bulk_batch.shape[0]}")
        print(f"\nBulk batch:")
        print(f"  Range: [{bulk_batch.min().item():.2f}, {bulk_batch.max().item():.2f}]")
        print(f"  Mean: {bulk_batch.mean().item():.2f}")

        print(f"\nSignature matrix S:")
        print(f"  Range: [{S.min().item():.2f}, {S.max().item():.2f}]")
        print(f"  Mean: {S.mean().item():.2f}")

        # Simulate deconvolution network (just use true proportions for now)
        p_pred = prop_batch.clone()
        p_pred = p_pred / p_pred.sum(dim=1, keepdim=True)  # Normalize

        print(f"\nPredicted proportions:")
        print(f"  Range: [{p_pred.min().item():.4f}, {p_pred.max().item():.4f}]")
        print(f"  Mean: {p_pred.mean().item():.4f}")
        print(f"  Sum (should be 1.0): {p_pred.sum(dim=1).mean().item():.4f}")

        # Reconstruct bulk
        b_rec = torch.matmul(p_pred, S.T)
        print(f"\nReconstructed bulk (p_pred @ S.T):")
        print(f"  Range: [{b_rec.min().item():.2f}, {b_rec.max().item():.2f}]")
        print(f"  Mean: {b_rec.mean().item():.2f}")
        print(f"  Has NaN: {torch.isnan(b_rec).any().item()}")
        print(f"  Has Inf: {torch.isinf(b_rec).any().item()}")

        # Compute scale ratio
        scale_ratio = bulk_batch.mean().item() / (b_rec.mean().item() + 1e-10)
        print(f"\nScale ratio (bulk / reconstructed): {scale_ratio:.2f}")
        if scale_ratio > 10 or scale_ratio < 0.1:
            print(f"  ⚠️  SEVERE SCALE MISMATCH!")

        # Compute losses
        print(f"\n[5] COMPUTING LOSSES")
        print("-" * 80)

        # Proportion loss
        p_true_safe = prop_batch.clamp(min=1e-10)
        p_true_safe = p_true_safe / p_true_safe.sum(dim=1, keepdim=True)

        loss_prop_mse = F.mse_loss(p_pred, p_true_safe)
        print(f"loss_prop_mse: {loss_prop_mse.item():.4f}")
        print(f"  Has NaN: {torch.isnan(loss_prop_mse).item()}")

        loss_prop_kl = F.kl_div(p_pred.log(), p_true_safe, reduction='batchmean')
        print(f"loss_prop_kl: {loss_prop_kl.item():.4f}")
        print(f"  Has NaN: {torch.isnan(loss_prop_kl).item()}")

        # Check if log has issues
        p_pred_log = p_pred.log()
        print(f"p_pred.log() range: [{p_pred_log.min().item():.4f}, {p_pred_log.max().item():.4f}]")
        print(f"p_pred.log() has NaN: {torch.isnan(p_pred_log).any().item()}")
        print(f"p_pred.log() has -Inf: {torch.isinf(p_pred_log).any().item()}")

        loss_prop = loss_prop_mse + 0.1 * loss_prop_kl
        print(f"\nloss_prop (combined): {loss_prop.item():.4f}")
        print(f"  Has NaN: {torch.isnan(loss_prop).item()}")

        # Bulk reconstruction loss
        loss_bulk_recon = F.mse_loss(b_rec, bulk_batch)
        print(f"\nloss_bulk_recon: {loss_bulk_recon.item():.4f}")
        print(f"  Has NaN: {torch.isnan(loss_bulk_recon).item()}")

        # Check difference magnitude
        diff = (b_rec - bulk_batch).abs()
        print(f"\nReconstruction error stats:")
        print(f"  Mean absolute error: {diff.mean().item():.2f}")
        print(f"  Max absolute error: {diff.max().item():.2f}")
        print(f"  MSE value: {(diff ** 2).mean().item():.2f}")

        print("\n" + "=" * 80)
        print("DIAGNOSTIC COMPLETE")
        print("=" * 80)

        if torch.isnan(loss_prop).item() or torch.isnan(loss_bulk_recon).item():
            print("\n⚠️  NaN DETECTED!")
            print("The diagnostic shows where NaN originates.")
        else:
            print("\n✅ NO NaN DETECTED in forward pass!")
            print("The issue might be in the VAE path or during backpropagation.")

    except Exception as e:
        print(f"\nERROR during forward pass simulation: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠️  Cannot simulate forward pass without signature matrix")

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)
print("1. Check the output above for NaN/Inf in any intermediate values")
print("2. If scale ratio is >10 or <0.1, we have a scale mismatch problem")
print("3. If p_pred.log() has -Inf, we need to add epsilon to proportions")
print("4. If bulk_recon loss is huge (>1e6), scale mismatch is the issue")
print("=" * 80)
