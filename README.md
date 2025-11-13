# HP-VADE: Hierarchical Prototype Variational Autoencoder for Deconvolution

PyTorch Lightning implementation of HP-VADE for bulk RNA-seq deconvolution using single-cell reference data.

## Project Structure

```
HP-VADE/
├── Phase01A_Test_PBMC.py      # Data preprocessing (HVG filtering, train/test split)
├── Phase01B_Test_PBMC.py      # Bulk data simulation and PyTorch Dataset creation
├── Phase01C_Model.py          # HP-VADE model architecture (Lightning Module)
├── Phase01C_Train.py          # Training function with Lightning Trainer
├── train_hp_vade.py           # ⭐ Main training script (USE THIS)
├── monitor_training.py        # Real-time training monitoring
├── run_training.sh            # Interactive training launcher
├── TRAINING_GUIDE.md          # 📖 Complete training guide
└── Instruction.txt            # Technical specification
```

## Quick Start

### 1. Prepare Data
```bash
# Step 1: Preprocess single-cell data
python Phase01A_Test_PBMC.py

# Step 2: Generate simulated bulk data
python Phase01B_Test_PBMC.py
```

### 2. Train Model

**Option A: Interactive Menu** (Recommended for beginners)
```bash
./run_training.sh
# Then select: 1 for quick test, or 2 for full training
```

**Option B: Command Line** (Recommended for remote servers)
```bash
# Quick test (2 epochs)
python train_hp_vade.py --quick-test -y

# Full training with defaults
python train_hp_vade.py -y

# Custom hyperparameters
python train_hp_vade.py \
    --learning-rate 0.001 \
    --batch-size 256 \
    --max-epochs 200 \
    --experiment-name my_experiment \
    -y
```

**Option C: Background Training** (For remote servers)
```bash
# Using nohup
nohup python train_hp_vade.py -y > training.log 2>&1 &

# Using screen
screen -S training
python train_hp_vade.py -y
# Press Ctrl+A then D to detach

# Using tmux
tmux new -s training
python train_hp_vade.py -y
# Press Ctrl+B then D to detach
```

### 3. Monitor Training

**Real-time Monitoring**
```bash
# One-time snapshot
python monitor_training.py

# Continuous monitoring (updates every 30s)
python monitor_training.py --watch

# Or use TensorBoard
tensorboard --logdir=./hp_vade_training/logs --host=0.0.0.0 --port=6006
```

**Check Status**
```bash
# View latest training log
tail -f training_*.log

# Check GPU usage
watch -n 1 nvidia-smi

# Interactive status check
./run_training.sh  # Then select option 7
```

## Command Reference

### Training Options
```bash
python train_hp_vade.py -h                    # Show all options

# Data paths
--data-dir DIR                                # Data directory
--output-dir DIR                              # Output directory

# Model architecture
--latent-dim N                                # Latent dimension (default: 32)
--hidden-dim N                                # Hidden layer size (default: 128)

# Training
--learning-rate LR, --lr LR                   # Learning rate (default: 1e-3)
--batch-size BS, --bs BS                      # Batch size (default: 128)
--max-epochs N                                # Maximum epochs (default: 100)
--patience N                                  # Early stopping patience (default: 20)

# Hardware
--cpu                                         # Force CPU (disable GPU)

# Utilities
--quick-test                                  # Run 2 epochs for testing
--experiment-name NAME                        # Custom experiment name
-y, --yes                                     # Skip confirmation prompt
```

### Monitoring Options
```bash
python monitor_training.py -h                 # Show all options

--log-dir DIR                                 # TensorBoard log directory
--output-dir DIR                              # Output directory for plots
--watch                                       # Continuous monitoring
--interval N                                  # Update interval in seconds
```

## Output Files

After training, you'll find:

```
hp_vade_training/
├── checkpoints/
│   └── hp_vade_YYYYMMDD_HHMMSS/
│       ├── hp_vade-epoch=XX-val_loss=Y.ckpt  # Best checkpoints
│       └── last.ckpt                         # Last checkpoint
├── logs/
│   └── hp_vade_YYYYMMDD_HHMMSS/
│       └── version_0/
│           └── events.out.tfevents.*         # TensorBoard logs
├── results/
│   ├── training_config.json                  # Configuration
│   ├── training_summary.txt                  # Summary
│   └── signature_matrix.npy                  # Learned signatures (genes × cell types)
└── monitoring/
    ├── total_loss.png                        # Loss curves
    ├── single-cell_losses.png                # SC losses
    ├── bulk_losses.png                       # Bulk losses
    ├── latent_metrics.png                    # Latent metrics
    └── metrics_comparison.csv                # Train/val comparison
```

## Model Architecture

HP-VADE consists of:

1. **Encoder**: Maps single-cell expression to latent distribution (μ, log σ²)
2. **Decoder**: Reconstructs expression from latent vectors
3. **Deconvolution Network**: Predicts cell type proportions from bulk data
4. **Signature Matrix (S)**: Learnable parameter connecting both paths (genes × cell types)

### Loss Function

Total Loss = SC Loss + λ_bulk × Bulk Loss

Where:
- **SC Loss** = L_recon + λ_kl × L_kl + λ_proto × L_proto
  - L_recon: Reconstruction loss (MSE)
  - L_kl: KL divergence from N(0,1)
  - L_proto: Novel prototype matching loss

- **Bulk Loss** = L_prop + λ_bulk_recon × L_bulk_recon
  - L_prop: Proportion prediction (KL divergence)
  - L_bulk_recon: Bulk reconstruction from S @ proportions

## Requirements

```bash
# Core
torch >= 1.10
pytorch-lightning >= 1.9
scanpy >= 1.9
anndata >= 0.8

# Utilities
numpy
pandas
scikit-learn
matplotlib
tensorboard
```

Install all:
```bash
pip install torch pytorch-lightning scanpy anndata numpy pandas scikit-learn matplotlib tensorboard
```

## Configuration

Default hyperparameters are in `train_hp_vade.py`:

```python
# Model
INPUT_DIM = 5000          # HVG genes
LATENT_DIM = 32           # Latent space
N_CELL_TYPES = 8          # PBMC cell types
N_HIDDEN = 128            # Hidden layer size

# Loss weights
LAMBDA_PROTO = 1.0        # Prototype loss
LAMBDA_BULK_RECON = 0.5   # Bulk reconstruction
LAMBDA_BULK = 1.0         # Total bulk path
LAMBDA_KL = 0.1           # KL divergence

# Training
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
BATCH_SIZE = 128
EARLY_STOPPING_PATIENCE = 20
```

Override via command line:
```bash
python train_hp_vade.py --lr 0.0005 --batch-size 256 --max-epochs 200 -y
```

## For Remote Servers

### SSH Tunnel for TensorBoard
```bash
# On remote server
tensorboard --logdir=./hp_vade_training/logs --port=6006

# On local machine
ssh -L 6006:localhost:6006 user@remote-server

# Then open: http://localhost:6006
```

### Background Training with Monitoring
```bash
# Terminal 1: Start training
screen -S training
python train_hp_vade.py -y
# Ctrl+A then D to detach

# Terminal 2: Monitor
screen -S monitor
python monitor_training.py --watch --interval 60
# Ctrl+A then D to detach

# Reattach anytime
screen -r training
screen -r monitor
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python train_hp_vade.py --batch-size 64 -y

# Or use CPU
python train_hp_vade.py --cpu -y
```

### Training Too Slow
```bash
# Increase batch size (if GPU allows)
python train_hp_vade.py --batch-size 512 -y

# Check GPU usage
nvidia-smi
```

### Loss Not Decreasing
```bash
# Try different learning rates
python train_hp_vade.py --lr 0.0001 -y   # Lower
python train_hp_vade.py --lr 0.005 -y    # Higher

# Quick test to debug
python train_hp_vade.py --quick-test -y
```

### NaN Loss
```bash
# Lower learning rate significantly
python train_hp_vade.py --lr 0.0001 -y

# Check data
python -c "import numpy as np; b = np.load('bulk_train.npy'); print('NaN:', np.isnan(b).any())"
```

## Documentation

- **TRAINING_GUIDE.md**: Complete guide for remote server training
- **Instruction.txt**: Technical specification and architecture details
- Code documentation: See docstrings in each Python file

## Example Workflow

```bash
# 1. Data preparation
python Phase01A_Test_PBMC.py
python Phase01B_Test_PBMC.py

# 2. Quick validation
python train_hp_vade.py --quick-test -y

# 3. Full training (background)
nohup python train_hp_vade.py -y > training.log 2>&1 &

# 4. Monitor progress
python monitor_training.py --watch

# 5. View results
cat ./hp_vade_training/results/training_summary.txt
ls ./hp_vade_training/monitoring/
```

## Citation

If you use this code, please cite:
```bibtex
@software{hp_vade_2024,
  title={HP-VADE: Hierarchical Prototype Variational Autoencoder for Deconvolution},
  author={HP-VADE Development Team},
  year={2024}
}
```

## License

[Specify your license here]

## Support

For issues and questions:
- Check TRAINING_GUIDE.md for detailed troubleshooting
- Review code documentation in Phase01C_Model.py
- See Instruction.txt for architecture details

---

**Ready to train?**
```bash
./run_training.sh
```

or

```bash
python train_hp_vade.py --quick-test -y
```
