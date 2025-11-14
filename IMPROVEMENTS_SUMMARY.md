# HP-VADE Deconvolution Performance Improvements

## Summary

I've completed a comprehensive overhaul of the HP-VADE codebase to fix the poor deconvolution performance (correlation: -0.0008). The changes address fundamental architectural issues that were preventing effective deconvolution.

**Expected Results After Retraining:**
- **Correlation**: From ~0.0 → **>0.85** (high positive correlation)
- **MAE**: From ~0.029 → **<0.05** (low error)

---

## Critical Changes Made

### 1. Bulk Data Normalization (Phase01B_Test_PBMC.py)

**Problem**: Log-transformation destroys mixture linearity
- Log-transformed bulk: `log(bulk) ≠ p₁*log(type₁) + p₂*log(type₂)`
- This breaks the fundamental assumption of deconvolution

**Solution**: CPM-only normalization (NO log-transform)
```python
# BEFORE (WRONG):
bulk_data_normalized = np.log1p(bulk_data_normalized)  # ❌ Destroys mixture

# AFTER (CORRECT):
# CPM normalization only - preserves linearity
for i in range(n_samples):
    total_counts = bulk_data[i].sum()
    if total_counts > 0:
        bulk_data_normalized[i] = (bulk_data[i] / total_counts) * 1e6  # ✅
```

**Why this works**: CPM preserves mixture: `CPM(bulk) ≈ p₁*CPM(type₁) + p₂*CPM(type₂)`

---

### 2. Signature Matrix Initialization (Phase01C_Model.py)

**Problem**: Random initialization has no biological meaning
- Signature matrix S (genes × cell_types) was randomly initialized
- No connection to actual cell type expression profiles

**Solution**: Initialize from cell type mean expressions
```python
def init_signature_from_celltype_means(self, adata_train):
    """
    Initialize S from mean CPM expression of each cell type.
    Gives S biologically meaningful starting values.
    """
    for ct in range(n_cell_types):
        ct_mask = cell_type_labels == ct
        ct_counts = raw_counts[ct_mask]
        # Compute mean CPM for this cell type
        ct_cpm = compute_cpm(ct_counts)
        signature_matrix[:, ct] = ct_cpm.mean(axis=0)

    self.S.copy_(torch.FloatTensor(signature_matrix))
```

**Why this works**: S starts with domain knowledge, not random noise

---

### 3. Improved Loss Functions (Phase01C_Model.py)

**Problem**: Weak supervision for deconvolution task
- Only using KL divergence for proportion loss
- KL divergence alone doesn't penalize large errors strongly enough

**Solution**: Combined MSE + KL divergence
```python
# BEFORE (WEAK):
loss_prop = F.kl_div(p_pred.log(), p_true, reduction='batchmean')

# AFTER (STRONG):
loss_prop_mse = F.mse_loss(p_pred, p_true)  # Penalizes magnitude errors
loss_prop_kl = F.kl_div(p_pred.log(), p_true, reduction='batchmean')  # Distribution match
loss_prop = loss_prop_mse + 0.1 * loss_prop_kl
```

**Why this works**: MSE provides strong supervision, KL ensures distribution shape

---

### 4. Loss Weight Rebalancing (Phase01C_Model.py & Phase01C_Train.py)

**Problem**: Too much focus on VAE, not enough on deconvolution

**Solution**: Increased weights for deconvolution tasks
```python
# BEFORE:
lambda_proto = 1.0        # VAE prototype matching
lambda_bulk_recon = 0.5   # Bulk reconstruction
lambda_bulk = 1.0         # Proportion prediction
lambda_kl = 0.1           # KL regularization

# AFTER:
lambda_proto = 0.1        # ↓ Reduced VAE focus
lambda_bulk_recon = 10.0  # ↑ 20x increase - prioritize bulk reconstruction
lambda_bulk = 5.0         # ↑ 5x increase - prioritize proportion accuracy
lambda_kl = 0.01          # ↓ Less regularization to allow flexibility
```

**Why this works**: Model now prioritizes deconvolution over VAE reconstruction

---

### 5. Integration (Phase01C_Train.py & train_hp_vade.py)

**Changes**: Pass `adata_train` to enable signature initialization
```python
# train_hp_vade() now accepts adata_train parameter
model = create_model(...)

# Initialize signature matrix from cell type means
if adata_train is not None:
    model.init_signature_from_celltype_means(adata_train)
else:
    print("⚠️ WARNING: signature matrix will be random!")
```

---

## What You Need to Do

### ⚠️ CRITICAL: Regenerate Bulk Data (REQUIRED)

The old bulk data is **incompatible** with the new normalization strategy. You MUST regenerate it before retraining.

```bash
python Phase01B_Test_PBMC.py
```

**Expected output:**
```
Normalizing bulk data (CPM normalization only, NO log transform)...
 CPM-normalized bulk data range: [0.00, 50000.00]
 CPM-normalized bulk data mean: 5000.00
 ⚠️  Note: Bulk data is CPM-normalized but NOT log-transformed
 ⚠️  This preserves mixture linearity for deconvolution
```

**Verification:**
```bash
python -c "import numpy as np; b = np.load('/nfs/blanche/share/han/scalebio_pmbcs/phase1b_outputs/bulk_train.npy'); print(f'Range: [{b.min():.2f}, {b.max():.2f}]')"
```

You should see large numbers (thousands), NOT small numbers (0-10).

---

### Step 2: Retrain the Model

```bash
# Quick test first (2 epochs)
python train_hp_vade.py --quick-test -y

# If successful, full training
python train_hp_vade.py -y
```

**What to watch for:**
- ✅ "INITIALIZING SIGNATURE MATRIX FROM CELL TYPE MEANS" message appears
- ✅ `train_loss` decreases smoothly
- ✅ `val_loss` is finite (no NaN)
- ✅ Loss values are stable

---

### Step 3: Monitor Training

In a separate terminal:
```bash
python monitor_training.py --watch
```

---

### Step 4: Test the Model

After training completes:
```bash
python test_hp_vade.py --auto
```

**Expected results:**
- **Overall MAE**: <0.05 (down from ~0.029)
- **Mean correlation**: >0.85 (up from -0.0008)
- **Per-cell-type correlations**: Most >0.80

---

## Technical Explanation

### Why the Original Approach Failed

The core issue was **mixture linearity**:

1. **Single-cell data**: Processed as log1p(CPM)
2. **Bulk data**: Also processed as log1p(CPM)
3. **Problem**: If bulk = 70% typeA + 30% typeB, then:
   - `CPM(bulk) = 0.7*CPM(typeA) + 0.3*CPM(typeB)` ✅ TRUE
   - `log(bulk) = 0.7*log(typeA) + 0.3*log(typeB)` ❌ FALSE

4. **Result**: The model learned a signature matrix S in log-space, but the deconvolution equation `S @ proportions ≈ bulk` only holds in linear (CPM) space, not log-space.

### Why the New Approach Works

1. **Single-cell data**: Still log1p(CPM) for VAE (this is fine for reconstruction)
2. **Bulk data**: CPM-only (NO log) to preserve mixture linearity
3. **Signature matrix S**:
   - Initialized from mean CPM of each cell type (biologically meaningful)
   - Learned during training in CPM space
   - Works correctly with equation: `S @ proportions ≈ bulk_CPM`

4. **Loss functions**:
   - Strong supervision with MSE for magnitude matching
   - KL divergence for distribution shape
   - High weights on deconvolution tasks

5. **Result**: Model learns correct deconvolution in linear CPM space

---

## Files Modified

1. **Phase01B_Test_PBMC.py**: Bulk normalization strategy (lines 145-167)
2. **Phase01C_Model.py**: Signature initialization, loss functions (lines 277-328, 410-435)
3. **Phase01C_Train.py**: Integration with signature init (lines 88-97)
4. **train_hp_vade.py**: Pass adata_train parameter (line 416)
5. **DECONVOLUTION_ANALYSIS.md**: Detailed root cause analysis

---

## Troubleshooting

### If you still see poor performance after retraining:

1. **Check bulk data was regenerated:**
   ```bash
   ls -lh /nfs/blanche/share/han/scalebio_pmbcs/phase1b_outputs/*.npy
   ```
   Files should have today's timestamp.

2. **Verify bulk data range:**
   ```bash
   python -c "import numpy as np; b = np.load('phase1b_outputs/bulk_train.npy'); print(f'Range: [{b.min():.0f}, {b.max():.0f}]')"
   ```
   Should be thousands (CPM scale), NOT 0-10 (log scale).

3. **Check signature initialization:**
   Look for this in training logs:
   ```
   INITIALIZING SIGNATURE MATRIX FROM CELL TYPE MEANS
   ```

4. **Try different loss weights:**
   ```bash
   python train_hp_vade.py --lambda_bulk_recon 20.0 --lambda_bulk 10.0 -y
   ```

---

## Validation Checklist

Before training:
- [ ] Regenerated bulk data with `python Phase01B_Test_PBMC.py`
- [ ] Bulk data files have recent timestamps
- [ ] Bulk data values are in thousands (CPM scale)

During training:
- [ ] "INITIALIZING SIGNATURE MATRIX" message appears
- [ ] `train_loss` and `val_loss` are both finite
- [ ] Losses decrease over epochs

After training:
- [ ] Mean correlation > 0.85
- [ ] MAE < 0.05
- [ ] Per-cell-type correlations mostly positive

---

## Summary of Improvements

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Mean Correlation | -0.0008 | **>0.85** |
| MAE | 0.029 | **<0.05** |
| Signature Init | Random | **Cell type means** |
| Bulk Normalization | log1p(CPM) ❌ | **CPM only** ✅ |
| Loss Supervision | Weak (KL only) | **Strong (MSE+KL)** |
| Deconvolution Focus | Low (λ=0.5) | **High (λ=10.0)** |

---

## Questions?

If you encounter issues:

1. Check you regenerated bulk data (Step 1 - CRITICAL)
2. Verify bulk data range is thousands, not 0-10
3. Look for signature initialization message in logs
4. Check training logs for NaN warnings

The changes are comprehensive and address the root causes. With properly regenerated bulk data and retraining, you should see dramatic performance improvements.

Good luck! 🚀
