# HP-VADE Training Guide for Remote Server

Complete guide for training the HP-VADE model on a remote server.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Training Workflow](#training-workflow)
4. [Monitoring Training](#monitoring-training)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Verify Data Preparation
```bash
# Ensure Phase 01A and 01B have been run
ls -lh /nfs/blanche/share/han/scalebio_pmbcs/

# You should see:
# - adata_train.h5ad
# - phase1b_outputs/bulk_train.npy
# - phase1b_outputs/props_train.npy
# - phase1b_outputs/bulk_val.npy
# - phase1b_outputs/props_val.npy
```

### 2. Quick Test (2 epochs)
```bash
# Test that everything works
python train_hp_vade.py --quick-test -y
```

### 3. Full Training
```bash
# Run full training with default settings
python train_hp_vade.py -y

# Or with custom parameters
python train_hp_vade.py \
    --learning-rate 0.001 \
    --batch-size 256 \
    --max-epochs 200 \
    -y
```

### 4. Monitor Training (in another terminal)
```bash
# One-time snapshot
python monitor_training.py

# Continuous monitoring (updates every 30 seconds)
python monitor_training.py --watch

# Or use TensorBoard
tensorboard --logdir=./hp_vade_training/logs --host=0.0.0.0 --port=6006
```

---

## Prerequisites

### Required Python Packages
```bash
# Core dependencies
pip install torch pytorch-lightning
pip install scanpy anndata
pip install numpy pandas scikit-learn
pip install tensorboard

# For monitoring script
pip install matplotlib pandas
```

### Data Files Required
The following files must exist (created by Phase01A and Phase01B):
- `adata_train.h5ad` - Preprocessed single-cell training data
- `bulk_train.npy` - Simulated bulk training samples
- `props_train.npy` - Training proportion labels
- `bulk_val.npy` - Simulated bulk validation samples
- `props_val.npy` - Validation proportion labels

---

## Training Workflow

### Step 1: Prepare Data (if not done)
```bash
# Run Phase 01A - Data preprocessing
python Phase01A_Test_PBMC.py

# Run Phase 01B - Bulk simulation
python Phase01B_Test_PBMC.py
```

### Step 2: Configure Training (Optional)

Edit `train_hp_vade.py` configuration section or use command-line arguments:

```python
# Key parameters to adjust:
LEARNING_RATE = 1e-3       # Lower for more stable training
MAX_EPOCHS = 100           # Increase for longer training
BATCH_SIZE = 128           # Increase if you have enough GPU memory
LATENT_DIM = 32            # Latent space dimensionality
```

### Step 3: Start Training

#### Basic Training
```bash
python train_hp_vade.py -y
```

#### Custom Configuration
```bash
python train_hp_vade.py \
    --data-dir /path/to/your/data \
    --output-dir ./my_experiment \
    --learning-rate 0.0005 \
    --batch-size 256 \
    --max-epochs 150 \
    --experiment-name my_hp_vade_run \
    -y
```

#### Run on CPU (if no GPU)
```bash
python train_hp_vade.py --cpu -y
```

### Step 4: Run in Background (Remote Server)

#### Option A: Using nohup
```bash
nohup python train_hp_vade.py -y > training.log 2>&1 &

# Check progress
tail -f training.log

# Find process ID
ps aux | grep train_hp_vade
```

#### Option B: Using screen
```bash
# Start a screen session
screen -S hp_vade_training

# Run training
python train_hp_vade.py -y

# Detach: Ctrl+A then D
# Reattach later
screen -r hp_vade_training
```

#### Option C: Using tmux
```bash
# Start tmux session
tmux new -s training

# Run training
python train_hp_vade.py -y

# Detach: Ctrl+B then D
# Reattach later
tmux attach -t training
```

---

## Monitoring Training

### Real-time Monitoring Script

#### Single Snapshot
```bash
python monitor_training.py
```
- Generates plots in `./hp_vade_training/monitoring/`
- Creates summary reports
- Exports metrics to CSV

#### Continuous Monitoring
```bash
python monitor_training.py --watch --interval 60
```
- Auto-refreshes every 60 seconds
- Updates plots and metrics
- Run in separate terminal or screen session

#### Custom Directories
```bash
python monitor_training.py \
    --log-dir ./my_experiment/logs \
    --output-dir ./my_experiment/monitoring \
    --watch
```

### TensorBoard

#### Start TensorBoard
```bash
# Local server
tensorboard --logdir=./hp_vade_training/logs

# Remote server (accessible from outside)
tensorboard --logdir=./hp_vade_training/logs --host=0.0.0.0 --port=6006
```

#### Access TensorBoard
- Local: http://localhost:6006
- Remote: http://your-server-ip:6006
- Via SSH tunnel: `ssh -L 6006:localhost:6006 user@server`

### Check Training Progress

```bash
# View latest logs
tail -f ./hp_vade_training/logs/*/events.out.tfevents.*

# Check checkpoints
ls -lh ./hp_vade_training/checkpoints/

# View configuration
cat ./hp_vade_training/results/training_config.json

# Check GPU usage (if using GPU)
watch -n 1 nvidia-smi
```

---

## Advanced Usage

### Resume from Checkpoint
```bash
# The training script will automatically use the best checkpoint
# To manually specify:
python train_hp_vade.py --resume ./checkpoints/epoch=50.ckpt
```

### Hyperparameter Tuning

#### Learning Rate
```bash
# Lower learning rate for stability
python train_hp_vade.py --lr 0.0001 -y

# Higher learning rate for faster convergence
python train_hp_vade.py --lr 0.005 -y
```

#### Batch Size
```bash
# Larger batch for better gradients (needs more GPU memory)
python train_hp_vade.py --batch-size 512 -y

# Smaller batch for limited memory
python train_hp_vade.py --batch-size 64 -y
```

#### Architecture
```bash
# Deeper latent space
python train_hp_vade.py --latent-dim 64 -y

# Wider hidden layers
python train_hp_vade.py --hidden-dim 256 -y
```

### Multiple Experiments

Run several experiments with different hyperparameters:

```bash
# Experiment 1: Standard
python train_hp_vade.py \
    --experiment-name exp1_standard \
    --learning-rate 0.001 \
    -y &

# Experiment 2: High LR
python train_hp_vade.py \
    --experiment-name exp2_high_lr \
    --learning-rate 0.005 \
    -y &

# Experiment 3: Deep latent
python train_hp_vade.py \
    --experiment-name exp3_deep_latent \
    --latent-dim 64 \
    -y &
```

---

## Output Files

After training, you'll find:

### Checkpoints
```
./hp_vade_training/checkpoints/hp_vade_YYYYMMDD_HHMMSS/
├── hp_vade-epoch=XX-val_loss=Y.YYYY.ckpt  # Best checkpoint
├── hp_vade-epoch=XX-val_loss=Y.YYYY.ckpt  # Top-k checkpoints
└── last.ckpt                               # Last epoch checkpoint
```

### Logs (TensorBoard)
```
./hp_vade_training/logs/hp_vade_YYYYMMDD_HHMMSS/
└── version_0/
    └── events.out.tfevents.*
```

### Results
```
./hp_vade_training/results/
├── training_config.json       # Full configuration
├── training_summary.txt       # Training summary
└── signature_matrix.npy       # Learned signature matrix (genes × cell types)
```

### Monitoring Outputs
```
./hp_vade_training/monitoring/
├── total_loss.png             # Training/validation loss curves
├── single-cell_losses.png     # SC-specific losses
├── bulk_losses.png            # Bulk-specific losses
├── latent_metrics.png         # Latent space metrics
├── training_summary.txt       # Text summary
└── metrics_comparison.csv     # Train vs validation comparison
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution:**
```bash
# Reduce batch size
python train_hp_vade.py --batch-size 64 -y

# Or use CPU
python train_hp_vade.py --cpu -y

# Or use gradient accumulation (modify Phase01C_Train.py)
```

### Issue: Training Too Slow

**Solution:**
```bash
# Increase batch size (if memory allows)
python train_hp_vade.py --batch-size 256 -y

# Use GPU
python train_hp_vade.py -y  # (default uses GPU if available)

# Reduce data size for testing
python train_hp_vade.py --quick-test -y
```

### Issue: Loss Not Decreasing

**Possible causes:**
1. Learning rate too high → Try `--lr 0.0001`
2. Learning rate too low → Try `--lr 0.005`
3. Bad initialization → Run again (different random seed)
4. Data issues → Check data preprocessing

**Debugging:**
```bash
# Check data
python -c "import scanpy as sc; adata = sc.read_h5ad('adata_train.h5ad'); print(adata)"

# Test with quick run
python train_hp_vade.py --quick-test -y

# Monitor all metrics
python monitor_training.py --watch
```

### Issue: NaN Loss

**Solution:**
```bash
# Lower learning rate
python train_hp_vade.py --lr 0.0001 -y

# Check for data issues
python -c "
import numpy as np
bulk = np.load('bulk_train.npy')
print('NaN in bulk:', np.isnan(bulk).any())
print('Inf in bulk:', np.isinf(bulk).any())
"
```

### Issue: Validation Loss Increasing (Overfitting)

**Solutions:**
- Lower `--max-epochs`
- The model uses early stopping (patience=20 by default)
- Check `--patience` parameter
- Add regularization (modify model code)

### Issue: Training Interrupted

**Recovery:**
```bash
# Training auto-saves checkpoints
# Check latest checkpoint
ls -lht ./hp_vade_training/checkpoints/*/

# Resume from last checkpoint (if implemented)
# Or restart training - it will create a new experiment
```

---

## Performance Tips

### For Remote Server

1. **Use screen/tmux** for persistent sessions
2. **Monitor GPU usage**: `watch nvidia-smi`
3. **Use nohup** for background execution
4. **Set up SSH tunnel** for TensorBoard access
5. **Monitor disk space**: Training creates checkpoints

### For Faster Training

1. **Increase batch size** (if GPU memory allows)
2. **Use GPU** (100x faster than CPU)
3. **Set `NUM_WORKERS > 0`** (in Phase01B config) if multiprocessing works
4. **Use mixed precision** (modify trainer settings)

### For Better Results

1. **Tune learning rate** (most important hyperparameter)
2. **Adjust loss weights** (λ values in config)
3. **Try different architectures** (latent_dim, hidden_dim)
4. **Run longer** (more epochs with patience)
5. **Ensemble multiple runs**

---

## Command Reference

### Training
```bash
python train_hp_vade.py -h                 # Show help
python train_hp_vade.py --quick-test       # Quick 2-epoch test
python train_hp_vade.py -y                 # Auto-confirm training
python train_hp_vade.py --cpu              # Force CPU
python train_hp_vade.py --lr 0.001         # Set learning rate
python train_hp_vade.py --batch-size 256   # Set batch size
python train_hp_vade.py --max-epochs 200   # Set max epochs
```

### Monitoring
```bash
python monitor_training.py                 # Single update
python monitor_training.py --watch         # Continuous monitoring
python monitor_training.py --interval 60   # Update every 60s
tensorboard --logdir=./hp_vade_training/logs  # TensorBoard
```

---

## Example Workflow

Complete workflow from start to finish:

```bash
# 1. Prepare data (if needed)
python Phase01A_Test_PBMC.py
python Phase01B_Test_PBMC.py

# 2. Quick test
python train_hp_vade.py --quick-test -y

# 3. Start full training in background
screen -S hp_vade
python train_hp_vade.py -y
# Press Ctrl+A then D to detach

# 4. Monitor in another terminal
screen -S monitoring
python monitor_training.py --watch --interval 60
# Press Ctrl+A then D to detach

# 5. Check progress anytime
screen -r hp_vade      # Training
screen -r monitoring   # Monitoring

# 6. After training completes
ls ./hp_vade_training/results/
cat ./hp_vade_training/results/training_summary.txt
```

---

## Questions?

Check the code documentation:
- `train_hp_vade.py` - Main training script
- `monitor_training.py` - Monitoring utilities
- `Phase01C_Model.py` - Model architecture
- `Phase01C_Train.py` - Training function

Good luck with your training! 🚀
