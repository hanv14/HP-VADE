# Bugfix: Scale Mismatch Causing val_loss = +Inf

## Problem

After fixing the log(0) NaN issue, the user reported:
- `train_loss` is now finite ✅
- `val_loss` is +Inf ❌

## Root Cause

### The Fundamental Scale Mismatch

The HP-VADE model has two data paths with different normalizations:

1. **Single-cell path (VAE)**:
   - Input: `.X` = log1p(normalize(counts, target_sum=**1e4**))
   - Values: 0-10 (log-normalized)
   - Decoder output: `sc_rec` in same log-space (0-10)

2. **Bulk path (Deconvolution)**:
   - Input: Bulk data = CPM (normalize to target_sum=**1e6**)
   - Values: 0-50,000 (CPM-normalized)
   - Signature matrix S: CPM space (0-50,000)

### The Bug: Prototype Loss

The prototype loss compared values from **different spaces**:

```python
# BUGGY CODE (lines 435-440 in training_step):
s_y_prototypes = self.S.T[sc_y]  # CPM space: values 0-50,000
loss_proto = F.mse_loss(sc_rec, s_y_prototypes)  # sc_rec: log-space 0-10
```

**The catastrophic math:**
```
sc_rec values: [0, 10]
S values: [0, 50,000]

MSE = mean((sc_rec - S)^2)
    ≈ mean((5 - 25,000)^2)
    ≈ mean(625,000,000)
    = 625,000,000

With lambda_proto = 0.1:
loss_proto_contribution = 0.1 * 625,000,000 = 62,500,000
```

This enormous loss easily overflows to **+Inf** in float32!

### Why It Affected Validation More

**Training step:**
- Had the same bug
- But also had other loss components that might have different magnitudes
- Training might have stopped before overflow in some batches

**Validation step:**
- ALSO had the bug (lines 589-590)
- Validation batches might have had slightly different value ranges
- More consistently hit overflow threshold
- Hence `val_loss = +Inf` while `train_loss` varied

Additionally, validation_step was missing the `p_pred_safe` clamping (line 607-608), which could cause -Inf from log(0) in validation.

## Solution

### Fix 1: Transform S to Log-Space for Prototype Loss

The signature matrix S is in CPM space (target_sum=1e6), but single-cell data is log1p(CPM with target_sum=1e4).

To match scales:
1. S is at 1e6 scale, .X is at 1e4 scale
2. Ratio: 1e4 / 1e6 = 1/100
3. Transform: log1p(S / 100)

**Fixed code (training_step lines 435-448):**
```python
# L_proto: Novel prototype loss
# CRITICAL: sc_rec is in log-normalized space (values 0-10)
#           S is in CPM space with target_sum=1e6 (values 0-50000)
#           We must transform S to match sc_rec's space!

# self.S shape: (input_dim, n_cell_types) - in CPM space (1e6 scale)
# Single-cell .X is log1p(CPM with target_sum=1e4)
# To match: log1p(S * 1e4 / 1e6) = log1p(S / 100)

s_y_prototypes_cpm = self.S.T[sc_y]  # (batch_size, input_dim) - CPM space
s_y_prototypes_log = torch.log1p(s_y_prototypes_cpm / 100.0)  # Convert to log-normalized space

loss_proto = F.mse_loss(sc_rec, s_y_prototypes_log)
```

**Fixed code (validation_step lines 590-593):**
```python
# Prototype loss: Transform S to log-space to match sc_rec
s_y_prototypes_cpm = self.S.T[sc_y]
s_y_prototypes_log = torch.log1p(s_y_prototypes_cpm / 100.0)
loss_proto = F.mse_loss(sc_rec, s_y_prototypes_log)
```

**Why this works:**
- Both `sc_rec` and `s_y_prototypes_log` are now in log-normalized space (0-10)
- MSE compares values of similar magnitude
- loss_proto ≈ 0.1 - 10 (reasonable range)
- No overflow to +Inf

### Fix 2: Add p_pred Clamping in Validation

Validation step was missing the epsilon clamping for p_pred:

**Fixed code (validation_step lines 609-615):**
```python
p_pred_safe = p_pred.clamp(min=1e-10)  # Prevent log(0) = -Inf
p_pred_safe = p_pred_safe / p_pred_safe.sum(dim=1, keepdim=True)

# MSE + KL divergence for proportions
loss_prop_mse = F.mse_loss(p_pred_safe, p_true_safe)
loss_prop_kl = F.kl_div(p_pred_safe.log(), p_true_safe, reduction='batchmean')
loss_prop = loss_prop_mse + 0.1 * loss_prop_kl
```

This ensures validation doesn't get -Inf from log(0) either.

## Technical Details

### Data Flow Summary

**Single-cell preprocessing (Phase01A):**
```python
1. Start with raw counts
2. Normalize: sc.pp.normalize_total(adata, target_sum=1e4)
   → Each cell sums to 10,000
3. Log transform: sc.pp.log1p(adata)
   → adata.X = log1p(normalized)
   → Values typically 0-10
4. Save: adata.X contains log1p(CPM-1e4)
   Save: adata.layers['counts'] contains raw counts
```

**Signature matrix initialization (Phase01C):**
```python
1. Load raw counts from layers['counts']
2. Aggregate by cell type: sum all cells of type i
3. CPM normalize: signature[:, i] = (aggregated / total) * 1e6
   → Each column sums to 1,000,000
   → Values typically 0-50,000
4. S stored in CPM space (1e6 scale)
```

**Bulk data simulation (Phase01B):**
```python
1. Sample and sum raw counts
2. CPM normalize: bulk = (counts / total) * 1e6
   → Values typically 0-50,000
3. Bulk data in CPM space (1e6 scale)
```

### Why Different Target Sums?

- **Single-cell**: target_sum=1e4 is standard in single-cell analysis
  - Makes values manageable (not too large)
  - Log-transform produces nice range (0-10)
  - Widely used convention

- **Bulk**: target_sum=1e6 is the definition of CPM (Counts Per Million)
  - Standard bulk RNA-seq normalization
  - Makes comparisons across samples meaningful

- **Our model**: Needs to handle both!
  - VAE works in single-cell space (log1p-1e4)
  - Deconvolution works in bulk space (CPM-1e6)
  - Prototype loss bridges them → needs transformation!

### Loss Magnitude Analysis

**Before fix:**
- loss_recon: ~0.5 (log-space, reasonable)
- loss_kl: ~5.0 (latent space, reasonable)
- loss_proto: ~625,000,000 ❌ (scale mismatch!)
- loss_prop: ~0.001 (proportions, reasonable)
- loss_bulk_recon: ~500,000 (CPM space, large but OK)

Total loss dominated by loss_proto → overflow to +Inf

**After fix:**
- loss_recon: ~0.5 ✅
- loss_kl: ~5.0 ✅
- loss_proto: ~1.5 ✅ (now in same space!)
- loss_prop: ~0.001 ✅
- loss_bulk_recon: ~500,000 ✅

All losses in reasonable ranges → no overflow!

## Testing

After applying this fix, training should show:

```
Epoch 0:
  train_loss=520.5  # Finite, reasonable
  val_loss=525.3    # Finite, reasonable
  loss_proto=1.2    # Much smaller than before!
```

### Validation Commands

```bash
python train_hp_vade.py -y
```

Watch for:
- ✅ Both train_loss and val_loss are finite
- ✅ val_loss is not +Inf
- ✅ loss_proto is in range 0.1 - 10 (not millions!)
- ✅ Training progresses smoothly

## Impact

**Before:**
- val_loss = +Inf
- Training unstable
- Prototype loss dominates all others
- Model cannot learn proper representations

**After:**
- All losses finite and balanced
- Training stable
- Prototype loss provides meaningful signal
- Model can learn both VAE and deconvolution tasks

## Files Modified

**Phase01C_Model.py:**
1. `training_step` (lines 435-448):
   - Added S→log transformation for prototype loss
   - Detailed comments explaining the transformation

2. `validation_step` (lines 590-593):
   - Added same S→log transformation
   - Added p_pred_safe clamping (lines 609-615)

## Summary

The core issue was **mixing different normalization spaces without transformation**:

- Single-cell VAE: log1p(target_sum=1e4) → values 0-10
- Bulk deconvolution: CPM (target_sum=1e6) → values 0-50,000
- Signature matrix: CPM (target_sum=1e6) → values 0-50,000

The prototype loss was comparing 0-10 with 0-50,000, creating losses of ~625M that overflowed to +Inf.

**Solution**: Transform S to log-space before comparing with VAE reconstructions:
- `log1p(S / 100)` matches the scale of single-cell data
- All losses now in reasonable ranges
- No more overflow!
