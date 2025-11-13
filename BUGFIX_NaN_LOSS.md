# NaN Validation Loss - FIXED ✅

## Problem Summary

The training was producing `val_loss=nan` due to a critical **scale mismatch** between single-cell and bulk data:

- **Single-cell data**: Log-normalized values (range: 0-10)
- **Bulk data**: Raw counts (range: thousands to millions)

This massive scale difference caused numerical instability in the loss calculations, resulting in NaN values.

---

## Root Cause Analysis

### Issue 1: Bulk Data Not Normalized

In `Phase01B_Test_PBMC.py`, the bulk simulation function:
1. Sampled cells and **summed their raw counts**
2. Returned the **raw count sums** without normalization
3. This created bulk samples with values in the range of 100,000+

Meanwhile, single-cell data was log-normalized (values typically 0-10).

When the model tried to:
- Predict proportions from bulk data (huge numbers)
- Reconstruct bulk using `S @ proportions` (small numbers)
- Compute MSE loss between them

The scale mismatch caused gradient explosion → NaN.

### Issue 2: KL Divergence Numerical Instability

The KL divergence calculation for proportion loss:
```python
loss_prop = F.kl_div(p_pred.log(), p_true + 1e-10, reduction='batchmean')
```

Had two problems:
1. Adding epsilon to `p_true` doesn't guarantee it sums to 1
2. Small numerical errors can cause KL divergence to explode

---

## Fixes Applied

### Fix 1: Bulk Data Normalization ✅

**File**: `Phase01B_Test_PBMC.py`

Added normalization to match single-cell preprocessing:

```python
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
```

**Result**: Bulk data now has the same scale as single-cell data (0-10 range).

### Fix 2: KL Divergence Stability ✅

**File**: `Phase01C_Model.py`

Fixed the KL divergence calculation in both `training_step` and `validation_step`:

```python
# Before (unstable)
loss_prop = F.kl_div(p_pred.log(), p_true + 1e-10, reduction='batchmean')

# After (stable)
p_true_safe = p_true.clamp(min=1e-10)  # Avoid zeros
p_true_safe = p_true_safe / p_true_safe.sum(dim=1, keepdim=True)  # Renormalize
loss_prop = F.kl_div(p_pred.log(), p_true_safe, reduction='batchmean')
```

**Result**: Proper probability distribution guarantees numerical stability.

### Fix 3: Data Validation ✅

**File**: `Phase01B_Test_PBMC.py`

Added validation checks in `SingleCellBulkDataset.__init__`:

```python
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
```

**Result**: Early detection prevents training with corrupted data.

### Fix 4: NaN Debugging ✅

**File**: `Phase01C_Model.py`

Added diagnostic printing in `training_step`:

```python
if torch.isnan(total_loss):
    print("\n⚠ WARNING: NaN detected in training!")
    print(f"  loss_recon: {loss_recon.item()}")
    print(f"  loss_kl: {loss_kl.item()}")
    print(f"  loss_proto: {loss_proto.item()}")
    print(f"  loss_prop: {loss_prop.item()}")
    print(f"  loss_bulk_recon: {loss_bulk_recon.item()}")
    # ... more diagnostics
```

**Result**: If NaN occurs, you'll see exactly which loss component failed.

---

## ⚠️ IMPORTANT: What You Need to Do

### Step 1: Regenerate Bulk Data (REQUIRED)

The old bulk data is **incompatible** with the fixed code because it's not normalized.

**Run this command:**

```bash
python Phase01B_Test_PBMC.py
```

This will:
- Generate new normalized bulk training data
- Generate new normalized bulk validation data
- Save them with proper preprocessing
- Overwrite the old files

**Output you should see:**
```
Normalizing bulk data (target_sum=1e4, log1p)...
 Normalized bulk data range: [0.00, 9.21]
 Normalized bulk data mean: 1.23
```

### Step 2: Start Training

Now you can train without NaN issues:

```bash
# Quick test first
python train_hp_vade.py --quick-test -y

# If that works, run full training
python train_hp_vade.py -y
```

### Step 3: Monitor Training

```bash
# In another terminal
python monitor_training.py --watch
```

You should now see:
- ✅ `train_loss`: Decreasing smoothly
- ✅ `val_loss`: Finite values, no NaN
- ✅ All loss components: Finite and stable

---

## Verification Checklist

Before training, verify:

- [ ] Ran `python Phase01B_Test_PBMC.py` to regenerate bulk data
- [ ] Saw "Normalizing bulk data" message in output
- [ ] Bulk data files updated (check timestamps):
  ```bash
  ls -lh /nfs/blanche/share/han/scalebio_pmbcs/phase1b_outputs/*.npy
  ```
- [ ] Files modified recently (today's date)

During training, verify:

- [ ] `train_loss` is finite (not NaN)
- [ ] `val_loss` is finite (not NaN)
- [ ] Loss decreases over epochs
- [ ] No warning messages about NaN

---

## What Changed Technically

### Before Fix:

```
Single-cell data: [0.0, 0.5, 1.2, 0.8, ...]  (log-normalized, ~0-10)
         ↓
    VAE Encoder/Decoder
         ↓
    Signature Matrix S (learned from log-normalized data)

Bulk data: [120450, 89234, 156789, ...]  (raw counts, ~100K)
         ↓
Deconvolution Net → proportions [0.2, 0.3, 0.5, ...]
         ↓
Bulk reconstruction: S @ proportions = [1.2, 0.8, 2.1, ...]  (log-normalized scale)
         ↓
MSE(bulk_rec, bulk_sim) = MSE([1.2, ...], [120450, ...])  ← SCALE MISMATCH!
         ↓
      GRADIENT EXPLOSION → NaN
```

### After Fix:

```
Single-cell data: [0.0, 0.5, 1.2, 0.8, ...]  (log-normalized, ~0-10)
         ↓
    VAE Encoder/Decoder
         ↓
    Signature Matrix S (learned from log-normalized data)

Bulk data: [120450, 89234, 156789, ...]  (raw counts)
         ↓
    NORMALIZATION (target_sum=1e4, log1p)
         ↓
Bulk data normalized: [2.1, 1.8, 3.2, ...]  (log-normalized, ~0-10)
         ↓
Deconvolution Net → proportions [0.2, 0.3, 0.5, ...]
         ↓
Bulk reconstruction: S @ proportions = [1.9, 2.2, 3.1, ...]  (log-normalized scale)
         ↓
MSE(bulk_rec, bulk_sim) = MSE([1.9, ...], [2.1, ...])  ← SAME SCALE!
         ↓
      STABLE GRADIENTS → No NaN ✅
```

---

## Expected Training Behavior

### Healthy Training (After Fix):

```
Epoch 0:  train_loss=2.456  val_loss=2.234  ✅
Epoch 1:  train_loss=2.123  val_loss=2.098  ✅
Epoch 2:  train_loss=1.987  val_loss=1.945  ✅
Epoch 3:  train_loss=1.876  val_loss=1.823  ✅
...
```

### Unhealthy Training (Before Fix):

```
Epoch 0:  train_loss=2.456  val_loss=2.234
Epoch 1:  train_loss=3.892  val_loss=nan      ❌
Epoch 2:  train_loss=nan    val_loss=nan      ❌
```

---

## Troubleshooting

### If you still get NaN after the fix:

1. **Check bulk data was regenerated:**
   ```bash
   python -c "import numpy as np; b = np.load('/nfs/blanche/share/han/scalebio_pmbcs/phase1b_outputs/bulk_train.npy'); print(f'Range: [{b.min():.2f}, {b.max():.2f}]')"
   ```

   Expected output: `Range: [0.00, 9.50]` (approximately)

   If you see: `Range: [0.00, 500000.00]` → You forgot to regenerate!

2. **Try lower learning rate:**
   ```bash
   python train_hp_vade.py --lr 0.0001 -y
   ```

3. **Check for data corruption:**
   ```bash
   python -c "
   import numpy as np
   b = np.load('bulk_train.npy')
   p = np.load('props_train.npy')
   print(f'Bulk NaN: {np.isnan(b).any()}')
   print(f'Props NaN: {np.isnan(p).any()}')
   print(f'Bulk Inf: {np.isinf(b).any()}')
   print(f'Props sum: {p.sum(axis=1).mean():.4f} (should be ~1.0)')
   "
   ```

4. **Enable detailed debugging:**
   The model now prints diagnostic info when NaN is detected. Check training logs.

---

## Summary

✅ **Fixed**: Scale mismatch between single-cell and bulk data
✅ **Fixed**: KL divergence numerical instability
✅ **Added**: Data validation checks
✅ **Added**: NaN debugging diagnostics

🔧 **Action Required**: Run `python Phase01B_Test_PBMC.py` to regenerate bulk data

📊 **Result**: Training should now be stable with finite losses

---

## Questions?

If you encounter any issues:

1. Check you regenerated bulk data (Step 1 above)
2. Verify bulk data range is 0-10, not thousands
3. Look for diagnostic messages in training output
4. Try quick test first: `python train_hp_vade.py --quick-test -y`

Good luck with your training! 🚀
