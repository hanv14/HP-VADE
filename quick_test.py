"""
Quick Test Script for HP-VADE
==============================
Simplified testing script for quick model evaluation.

Usage:
    python quick_test.py
    python quick_test.py --checkpoint path/to/checkpoint.ckpt
"""

import sys
import numpy as np
import torch
import scanpy as sc
from pathlib import Path

# Check if testing script is available
try:
    from test_hp_vade import (
        load_model, load_test_data, generate_test_bulk,
        evaluate_deconvolution, TestConfig, get_cell_type_names
    )
except ImportError:
    print("ERROR: test_hp_vade.py not found!")
    sys.exit(1)


def main():
    """Quick test workflow."""

    print("\n" + "=" * 70)
    print("HP-VADE QUICK TEST")
    print("=" * 70)

    # Configuration
    config = TestConfig()
    config.N_TEST_BULK_SAMPLES = 500  # Smaller for quick test

    # Auto-find checkpoint
    checkpoint_dir = Path('./hp_vade_training/checkpoints')

    if not checkpoint_dir.exists():
        print("\nERROR: No training checkpoints found!")
        print("Please train a model first using: python train_hp_vade.py")
        return

    # Find latest experiment
    exp_dirs = list(checkpoint_dir.glob('hp_vade_*'))
    if not exp_dirs:
        print("\nERROR: No experiment directories found!")
        return

    latest_exp = max(exp_dirs, key=lambda x: x.stat().st_mtime)
    print(f"\nUsing experiment: {latest_exp.name}")

    # Find best checkpoint
    checkpoints = list(latest_exp.glob('*.ckpt'))
    if not checkpoints:
        print("\nERROR: No checkpoints found in experiment!")
        return

    # Use last.ckpt or first checkpoint
    checkpoint_path = latest_exp / 'last.ckpt'
    if not checkpoint_path.exists():
        checkpoint_path = checkpoints[0]

    print(f"Using checkpoint: {checkpoint_path.name}")

    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    model = load_model(str(checkpoint_path), device)

    # Load test data
    print("\nLoading test data...")
    adata_test = load_test_data(config)

    # Get cell type names
    cell_type_names = get_cell_type_names(config)

    # Generate test bulk
    print(f"\nGenerating {config.N_TEST_BULK_SAMPLES} test bulk samples...")
    bulk_test, props_test = generate_test_bulk(adata_test, config)

    # Evaluate
    print("\nEvaluating deconvolution...")
    results = evaluate_deconvolution(model, bulk_test, props_test, device)

    # Print summary
    print("\n" + "=" * 70)
    print("QUICK TEST RESULTS")
    print("=" * 70)
    print(f"\nOverall Performance:")
    print(f"  MAE:         {results['overall_mae']:.6f}")
    print(f"  RMSE:        {results['overall_rmse']:.6f}")
    print(f"  Correlation: {results['mean_correlation']:.6f}")

    print(f"\nPer-Cell-Type MAE:")
    for i, (ct_name, mae) in enumerate(zip(cell_type_names, results['per_type_mae'])):
        print(f"  {ct_name:30s}: {mae:.6f}")

    print("\n" + "=" * 70)
    print("✅ Quick test completed!")
    print("\nFor full testing with plots, run:")
    print(f"  python test_hp_vade.py --auto")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
