"""
HP-VADE Training Script
=======================
Main training script for HP-VADE model with comprehensive monitoring and logging.
Designed for remote server execution with file-based logging and checkpointing.

Usage:
    python train_hp_vade.py [--config config.yaml]
    python train_hp_vade.py --resume checkpoint.ckpt
    python train_hp_vade.py --quick-test  # Run 2 epochs for testing

Author: HP-VADE Development Team
Date: 2024
"""

import os
import sys
import argparse
import torch
import numpy as np
import scanpy as sc
from datetime import datetime
from pathlib import Path

# Import HP-VADE components
from Phase01C_Model import HP_VADE, create_model
from Phase01C_Train import train_hp_vade
from Phase01B_Test_PBMC import SingleCellBulkDataset
from torch.utils.data import DataLoader

# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Configuration class for HP-VADE training."""

    # Data paths
    DATA_DIR = '/nfs/blanche/share/han/scalebio_pmbcs'
    TRAIN_DATA_PATH = f'{DATA_DIR}/adata_train.h5ad'
    BULK_TRAIN_PATH = f'{DATA_DIR}/phase1b_outputs/bulk_train.npy'
    PROPS_TRAIN_PATH = f'{DATA_DIR}/phase1b_outputs/props_train.npy'
    BULK_VAL_PATH = f'{DATA_DIR}/phase1b_outputs/bulk_val.npy'
    PROPS_VAL_PATH = f'{DATA_DIR}/phase1b_outputs/props_val.npy'

    # Output paths
    OUTPUT_DIR = './hp_vade_training'
    CHECKPOINT_DIR = f'{OUTPUT_DIR}/checkpoints'
    LOG_DIR = f'{OUTPUT_DIR}/logs'
    RESULTS_DIR = f'{OUTPUT_DIR}/results'

    # Model architecture
    INPUT_DIM = 5000  # Number of HVG genes
    LATENT_DIM = 32
    N_CELL_TYPES = 8  # PBMC cell types
    N_HIDDEN = 128

    # Loss weights
    LAMBDA_PROTO = 1.0
    LAMBDA_BULK_RECON = 0.5
    LAMBDA_BULK = 1.0
    LAMBDA_KL = 0.1

    # Training parameters
    LEARNING_RATE = 1e-3
    MAX_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 20
    BATCH_SIZE = 128
    NUM_WORKERS = 0  # For remote server, set to 0 to avoid multiprocessing issues

    # GPU settings
    USE_GPU = True

    # Experiment naming
    EXPERIMENT_NAME = f"hp_vade_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    @classmethod
    def from_args(cls, args):
        """Update configuration from command-line arguments."""
        config = cls()
        if args.data_dir:
            config.DATA_DIR = args.data_dir
            config.TRAIN_DATA_PATH = f'{args.data_dir}/adata_train.h5ad'
            config.BULK_TRAIN_PATH = f'{args.data_dir}/phase1b_outputs/bulk_train.npy'
            config.PROPS_TRAIN_PATH = f'{args.data_dir}/phase1b_outputs/props_train.npy'
            config.BULK_VAL_PATH = f'{args.data_dir}/phase1b_outputs/bulk_val.npy'
            config.PROPS_VAL_PATH = f'{args.data_dir}/phase1b_outputs/props_val.npy'

        if args.output_dir:
            config.OUTPUT_DIR = args.output_dir
            config.CHECKPOINT_DIR = f'{args.output_dir}/checkpoints'
            config.LOG_DIR = f'{args.output_dir}/logs'
            config.RESULTS_DIR = f'{args.output_dir}/results'

        if args.batch_size:
            config.BATCH_SIZE = args.batch_size

        if args.learning_rate:
            config.LEARNING_RATE = args.learning_rate

        if args.max_epochs:
            config.MAX_EPOCHS = args.max_epochs

        if args.experiment_name:
            config.EXPERIMENT_NAME = args.experiment_name

        config.USE_GPU = not args.cpu

        return config


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_directories(config):
    """Create necessary directories for training."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print(f"✓ Created output directories in: {config.OUTPUT_DIR}")


def check_data_availability(config):
    """Verify all required data files exist."""
    required_files = [
        config.TRAIN_DATA_PATH,
        config.BULK_TRAIN_PATH,
        config.PROPS_TRAIN_PATH,
        config.BULK_VAL_PATH,
        config.PROPS_VAL_PATH
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("ERROR: Missing required data files:")
        for file_path in missing_files:
            print(f"  ✗ {file_path}")
        print("\nPlease run Phase01A and Phase01B scripts first to prepare the data.")
        return False

    print("✓ All required data files found")
    return True


def load_data(config):
    """Load all training data."""
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Load single-cell data
    print(f"Loading single-cell data from: {config.TRAIN_DATA_PATH}")
    adata_train = sc.read_h5ad(config.TRAIN_DATA_PATH)
    print(f"  ✓ Single-cell data: {adata_train.shape}")
    print(f"  ✓ Cell types: {adata_train.obs['cell_type_int'].nunique()}")

    # Update input_dim based on actual data
    config.INPUT_DIM = adata_train.n_vars
    print(f"  ✓ Input dimension set to: {config.INPUT_DIM}")

    # Load bulk data
    print(f"\nLoading simulated bulk data...")
    bulk_train = np.load(config.BULK_TRAIN_PATH)
    props_train = np.load(config.PROPS_TRAIN_PATH)
    bulk_val = np.load(config.BULK_VAL_PATH)
    props_val = np.load(config.PROPS_VAL_PATH)

    print(f"  ✓ Training bulk samples: {bulk_train.shape}")
    print(f"  ✓ Validation bulk samples: {bulk_val.shape}")
    print(f"  ✓ Proportion shape: {props_train.shape}")

    # Update n_cell_types based on actual data
    config.N_CELL_TYPES = props_train.shape[1]
    print(f"  ✓ Number of cell types set to: {config.N_CELL_TYPES}")

    return adata_train, bulk_train, props_train, bulk_val, props_val


def create_dataloaders(adata_train, bulk_train, props_train, bulk_val, props_val, config):
    """Create PyTorch DataLoaders for training and validation."""
    print("\n" + "=" * 80)
    print("CREATING DATALOADERS")
    print("=" * 80)

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

    print(f"Training dataset:")
    print(f"  Single-cell samples: {train_dataset.n_sc:,}")
    print(f"  Bulk samples: {train_dataset.n_bulk:,}")

    print(f"\nValidation dataset:")
    print(f"  Single-cell samples: {val_dataset.n_sc:,}")
    print(f"  Bulk samples: {val_dataset.n_bulk:,}")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available() and config.USE_GPU
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # Don't shuffle validation
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available() and config.USE_GPU
    )

    print(f"\nDataLoader settings:")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Training batches per epoch: {len(train_dataloader):,}")
    print(f"  Validation batches: {len(val_dataloader):,}")

    return train_dataloader, val_dataloader


def print_training_summary(config):
    """Print a comprehensive training configuration summary."""
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION SUMMARY")
    print("=" * 80)

    print(f"\nExperiment: {config.EXPERIMENT_NAME}")
    print(f"\nModel Architecture:")
    print(f"  Input dimension (genes):     {config.INPUT_DIM:,}")
    print(f"  Latent dimension:            {config.LATENT_DIM}")
    print(f"  Number of cell types:        {config.N_CELL_TYPES}")
    print(f"  Hidden layer size:           {config.N_HIDDEN}")

    print(f"\nLoss Weights:")
    print(f"  λ_proto (prototype loss):    {config.LAMBDA_PROTO}")
    print(f"  λ_bulk_recon (bulk recon):   {config.LAMBDA_BULK_RECON}")
    print(f"  λ_bulk (total bulk path):    {config.LAMBDA_BULK}")
    print(f"  λ_kl (KL divergence):        {config.LAMBDA_KL}")

    print(f"\nTraining Parameters:")
    print(f"  Learning rate:               {config.LEARNING_RATE}")
    print(f"  Max epochs:                  {config.MAX_EPOCHS}")
    print(f"  Early stopping patience:     {config.EARLY_STOPPING_PATIENCE}")
    print(f"  Batch size:                  {config.BATCH_SIZE}")

    print(f"\nHardware:")
    print(f"  GPU available:               {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU device:                  {torch.cuda.get_device_name(0)}")
    print(f"  Using GPU:                   {config.USE_GPU and torch.cuda.is_available()}")

    print(f"\nOutput Directories:")
    print(f"  Checkpoints:                 {config.CHECKPOINT_DIR}")
    print(f"  Logs:                        {config.LOG_DIR}")
    print(f"  Results:                     {config.RESULTS_DIR}")

    print("=" * 80)


def save_training_config(config):
    """Save training configuration to file."""
    import json

    config_dict = {
        'experiment_name': config.EXPERIMENT_NAME,
        'timestamp': datetime.now().isoformat(),
        'model_architecture': {
            'input_dim': config.INPUT_DIM,
            'latent_dim': config.LATENT_DIM,
            'n_cell_types': config.N_CELL_TYPES,
            'n_hidden': config.N_HIDDEN
        },
        'loss_weights': {
            'lambda_proto': config.LAMBDA_PROTO,
            'lambda_bulk_recon': config.LAMBDA_BULK_RECON,
            'lambda_bulk': config.LAMBDA_BULK,
            'lambda_kl': config.LAMBDA_KL
        },
        'training_parameters': {
            'learning_rate': config.LEARNING_RATE,
            'max_epochs': config.MAX_EPOCHS,
            'early_stopping_patience': config.EARLY_STOPPING_PATIENCE,
            'batch_size': config.BATCH_SIZE
        },
        'data_paths': {
            'train_data': config.TRAIN_DATA_PATH,
            'bulk_train': config.BULK_TRAIN_PATH,
            'props_train': config.PROPS_TRAIN_PATH
        }
    }

    config_path = f'{config.RESULTS_DIR}/training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"\n✓ Training configuration saved to: {config_path}")


def save_final_results(model, trainer, config):
    """Save final training results and signature matrix."""
    print("\n" + "=" * 80)
    print("SAVING FINAL RESULTS")
    print("=" * 80)

    # Save signature matrix
    S = model.get_signature_matrix()
    sig_path = f'{config.RESULTS_DIR}/signature_matrix.npy'
    np.save(sig_path, S)
    print(f"✓ Signature matrix saved: {sig_path}")
    print(f"  Shape: {S.shape}")

    # Save training summary
    summary = {
        'experiment_name': config.EXPERIMENT_NAME,
        'final_epoch': trainer.current_epoch,
        'best_checkpoint': trainer.checkpoint_callback.best_model_path,
        'signature_matrix_shape': S.shape,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

    summary_path = f'{config.RESULTS_DIR}/training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("HP-VADE Training Summary\n")
        f.write("=" * 80 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print(f"✓ Training summary saved: {summary_path}")
    print("=" * 80)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main(args):
    """Main training workflow."""

    # Banner
    print("\n" + "=" * 80)
    print("HP-VADE MODEL TRAINING")
    print("Hierarchical Prototype Variational Autoencoder for Deconvolution")
    print("=" * 80)

    # Load configuration
    config = TrainingConfig.from_args(args)

    # Quick test mode
    if args.quick_test:
        print("\n⚡ QUICK TEST MODE - Running 2 epochs only")
        config.MAX_EPOCHS = 2
        config.EARLY_STOPPING_PATIENCE = 5
        config.EXPERIMENT_NAME = f"quick_test_{datetime.now().strftime('%H%M%S')}"

    # Setup directories
    setup_directories(config)

    # Check data availability
    if not check_data_availability(config):
        sys.exit(1)

    # Load data
    adata_train, bulk_train, props_train, bulk_val, props_val = load_data(config)

    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        adata_train, bulk_train, props_train, bulk_val, props_val, config
    )

    # Print training summary
    print_training_summary(config)

    # Save configuration
    save_training_config(config)

    # Confirmation prompt (skip in quick test mode)
    if not args.quick_test and not args.yes:
        response = input("\nProceed with training? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Training cancelled.")
            sys.exit(0)

    # Start training
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"\nMonitor training progress with TensorBoard:")
    print(f"  tensorboard --logdir={config.LOG_DIR}")
    print(f"\nCheckpoints will be saved to:")
    print(f"  {config.CHECKPOINT_DIR}/{config.EXPERIMENT_NAME}")
    print("\n" + "=" * 80 + "\n")

    try:
        model, trainer = train_hp_vade(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            input_dim=config.INPUT_DIM,
            latent_dim=config.LATENT_DIM,
            n_cell_types=config.N_CELL_TYPES,
            n_hidden=config.N_HIDDEN,
            lambda_proto=config.LAMBDA_PROTO,
            lambda_bulk_recon=config.LAMBDA_BULK_RECON,
            lambda_bulk=config.LAMBDA_BULK,
            lambda_kl=config.LAMBDA_KL,
            learning_rate=config.LEARNING_RATE,
            max_epochs=config.MAX_EPOCHS,
            patience=config.EARLY_STOPPING_PATIENCE,
            use_gpu=config.USE_GPU,
            experiment_name=config.EXPERIMENT_NAME
        )

        # Save final results
        save_final_results(model, trainer, config)

        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nBest model checkpoint:")
        print(f"  {trainer.checkpoint_callback.best_model_path}")
        print(f"\nResults saved to:")
        print(f"  {config.RESULTS_DIR}")
        print("\nTo view training metrics:")
        print(f"  tensorboard --logdir={config.LOG_DIR}")
        print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train HP-VADE model for bulk RNA-seq deconvolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default settings
  python train_hp_vade.py

  # Quick test (2 epochs)
  python train_hp_vade.py --quick-test

  # Custom data directory
  python train_hp_vade.py --data-dir /path/to/data

  # Custom hyperparameters
  python train_hp_vade.py --learning-rate 0.001 --batch-size 256 --max-epochs 200

  # Run on CPU (no GPU)
  python train_hp_vade.py --cpu

  # Auto-confirm (no prompt)
  python train_hp_vade.py -y
        """
    )

    # Data arguments
    parser.add_argument('--data-dir', type=str,
                        help='Directory containing training data (default: /nfs/blanche/share/han/scalebio_pmbcs)')
    parser.add_argument('--output-dir', type=str,
                        help='Directory for outputs (default: ./hp_vade_training)')

    # Model arguments
    parser.add_argument('--latent-dim', type=int,
                        help='Latent dimension (default: 32)')
    parser.add_argument('--hidden-dim', type=int,
                        help='Hidden layer dimension (default: 128)')

    # Training arguments
    parser.add_argument('--learning-rate', '--lr', type=float,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--batch-size', '--bs', type=int,
                        help='Batch size (default: 128)')
    parser.add_argument('--max-epochs', type=int,
                        help='Maximum number of epochs (default: 100)')
    parser.add_argument('--patience', type=int,
                        help='Early stopping patience (default: 20)')

    # Hardware arguments
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage (disable GPU)')

    # Utility arguments
    parser.add_argument('--quick-test', action='store_true',
                        help='Run quick test with 2 epochs')
    parser.add_argument('--experiment-name', type=str,
                        help='Custom experiment name')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Auto-confirm training without prompt')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
