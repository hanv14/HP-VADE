# Bugfix: Signature Matrix Initialization and NaN Loss

## Problem Report

User reported two critical issues:
1. **Loss is NaN during training**
2. **Signature matrix S initialized to "200 everywhere"**

## Root Cause Analysis

### Issue 1: S = 200 Everywhere

The original initialization code had a fundamental flaw:

```python
# OLD CODE (WRONG):
for ct in range(n_cell_types):
    ct_mask = cell_type_labels == ct
    ct_counts = raw_counts[ct_mask]  # Using 'counts' layer

    # CPM normalize each cell individually
    ct_cpm = np.zeros_like(ct_counts, dtype=np.float32)
    for i in range(ct_counts.shape[0]):
        total = ct_counts[i].sum()
        if total > 0:
            ct_cpm[i] = (ct_counts[i] / total) * 1e6

    # Average CPM across cells
    signature_matrix[:, ct] = ct_cpm.mean(axis=0)
```

**Problems:**
1. The `adata.layers['counts']` contains normalized counts (target_sum=1e4), NOT raw counts
2. When each cell is already normalized to sum to 10,000:
   - CPM normalization scales to 1,000,000
   - With 5,000 genes, average expression = 1,000,000 / 5,000 = **200**
3. Averaging CPM-normalized cells washes out cell-type-specific patterns
4. Result: All genes, all cell types → mean value of ~200

### Issue 2: NaN Loss

With S ≈ 200 everywhere:
1. Bulk data: CPM-normalized, values in thousands (e.g., 1000-50000)
2. Reconstructed bulk: `b_rec = p_pred @ S.T`
   - p_pred: proportions summing to 1
   - S.T: All values ~200
   - b_rec: Values ~200 (much smaller than actual bulk!)
3. MSE loss: `||b_rec - b_sim||²` where b_rec ~200 and b_sim ~10000
   - Loss is HUGE (100x scale mismatch)
4. Gradients explode → NaN

## Solution

### Fixed Initialization Approach

**Key insight:** Aggregate counts at cell-type level FIRST, then normalize.

```python
# NEW CODE (CORRECT):
for ct in range(n_cell_types):
    ct_mask = cell_type_labels == ct
    ct_data = X_data[ct_mask]  # Use .X (log1p normalized)

    # Inverse log transform: expm1(log1p(x)) = x
    ct_counts = np.expm1(ct_data)  # Back to CPM with target_sum=1e4

    # AGGREGATE FIRST: Sum across all cells of this type
    aggregated_counts = ct_counts.sum(axis=0)  # (n_genes,)

    # THEN NORMALIZE: CPM normalize the aggregated profile
    total = aggregated_counts.sum()
    if total > 0:
        ct_signature = (aggregated_counts / total) * 1e6
    else:
        ct_signature = np.zeros(n_genes, dtype=np.float32)

    signature_matrix[:, ct] = ct_signature
```

**Why this works:**
1. Uses `.X` data (log1p normalized, more reliable than 'counts' layer)
2. Aggregates at cell-type level before normalization
3. Preserves cell-type-specific expression patterns
4. Results in CPM-scale signatures (thousands) matching bulk data scale
5. Different cell types have different expression profiles

### Additional Safety Measures

#### 1. Scale Validation During Initialization

Added diagnostics to check signature matrix properties:
- Range, mean, std across cell types
- Per-cell-type signature statistics
- Variance check (warns if all cell types look the same)

```python
# Check variance across cell types
s_std_per_gene = self.S.std(dim=1)
print(f"Mean std across cell types per gene: {s_std_per_gene.mean().item():.2f}")
if s_std_per_gene.mean().item() < 10:
    print("⚠️  WARNING: Very low variance across cell types!")
    print("This suggests cell types have similar expression profiles!")
```

#### 2. First-Batch Scale Validation

Added comprehensive diagnostics on first training batch:
```python
if batch_idx == 0 and self.current_epoch == 0:
    # Print bulk data scale
    # Print signature matrix scale
    # Print reconstructed bulk scale
    # Check scale ratio
    scale_ratio = b_sim.mean() / b_rec.mean()
    if scale_ratio > 10 or scale_ratio < 0.1:
        print("⚠️  WARNING: SEVERE SCALE MISMATCH!")
```

This catches scale mismatches immediately before they cause NaN.

#### 3. Increased Gradient Clipping

```python
# Phase01C_Train.py
'gradient_clip_val': 5.0,  # Increased from 1.0
'gradient_clip_algorithm': 'norm',
```

More permissive gradient clipping allows the model to adjust to initial scale differences without immediately clamping gradients.

## Expected Results After Fix

### Signature Matrix:
- **Before**: All values ~200, no variance across cell types
- **After**:
  - Values in CPM scale (hundreds to tens of thousands)
  - Clear differences between cell types
  - Per-gene variance across cell types > 10

### Training:
- **Before**: NaN loss immediately or within first epoch
- **After**:
  - Stable training from epoch 1
  - Scale ratio between bulk and reconstructed ~1.0
  - Gradients remain finite

### Deconvolution Performance:
- Signature matrix properly initialized with biological meaning
- Model can learn cell-type-specific patterns
- Expected correlation > 0.85, MAE < 0.05

## Validation Steps

After applying this fix:

1. **Check initialization output:**
   ```
   [HP-VADE] Initializing signature matrix from cell type means...
   [HP-VADE] Data inspection:
     X data range: [0.0000, 10.5432]
     X data mean: 0.8234

   Cell type 0: 45,123 cells
     Signature range: [0.00, 35234.56]
     Signature mean: 4532.12
     Signature sum: 1000000.00
   ```

   ✅ Signature values should be in thousands, NOT all 200
   ✅ Different cell types should have different means

2. **Check first batch validation:**
   ```
   FIRST BATCH VALIDATION - CHECKING SCALES
   Bulk data (b_sim):
     Range: [0.00, 45000.00]
     Mean: 5234.56

   Signature matrix (S):
     Range: [0.00, 35000.00]
     Mean: 3456.78

   Reconstructed bulk (b_rec):
     Range: [0.00, 40000.00]
     Mean: 4987.12

   Scale ratio (bulk_data / bulk_reconstructed): 1.05
   ```

   ✅ Bulk and reconstructed should have similar scales
   ✅ Scale ratio should be close to 1.0 (within 0.5 - 2.0 range)

3. **Monitor training:**
   ```
   Epoch 0: train_loss=1234.56, val_loss=1456.78
   Epoch 1: train_loss=987.65, val_loss=1123.45
   ```

   ✅ Losses should be finite (no NaN)
   ✅ Losses should decrease over epochs

## Files Modified

1. **Phase01C_Model.py**:
   - `init_signature_from_celltype_means()`: Complete rewrite using aggregation approach
   - `training_step()`: Added first-batch scale validation
   - Lines changed: 277-365, 479-521

2. **Phase01C_Train.py**:
   - Increased gradient clipping: 1.0 → 5.0
   - Added gradient clipping algorithm specification
   - Lines changed: 142-143

## Related Issues

This fix also addresses:
- Poor deconvolution performance (S had no discriminative power)
- Training instability (scale mismatches)
- Gradient explosion (huge MSE losses from scale mismatch)

## Testing

After regenerating data and retraining:
```bash
python Phase01B_Test_PBMC.py  # Regenerate bulk data
python train_hp_vade.py -y     # Retrain with fixes
```

Watch for:
- ✅ Signature initialization shows varied values per cell type
- ✅ First batch validation shows scale ratio ~1.0
- ✅ Training progresses without NaN
- ✅ Deconvolution performance improves dramatically
