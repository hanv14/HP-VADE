"""
HP-VADE Model Testing and Evaluation
=====================================
Test the trained HP-VADE model on held-out test data and evaluate deconvolution performance.

Usage:
    python test_hp_vade.py --checkpoint path/to/checkpoint.ckpt
    python test_hp_vade.py --checkpoint best --experiment hp_vade_20241113
    python test_hp_vade.py --auto  # Automatically find latest checkpoint

Author: HP-VADE Development Team
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import scanpy as sc
from Phase01C_Model import HP_VADE
from Phase01B_Test_PBMC import create_simulated_bulk

# ============================================================================
# CONFIGURATION
# ============================================================================

class TestConfig:
    """Configuration for testing."""

    # Data paths
    DATA_DIR = '/nfs/blanche/share/han/scalebio_pmbcs'
    TEST_DATA_PATH = f'{DATA_DIR}/adata_test.h5ad'
    TRAIN_DATA_PATH = f'{DATA_DIR}/adata_train.h5ad'  # For cell type mapping

    # Test bulk generation parameters
    N_TEST_BULK_SAMPLES = 5000
    CELLS_PER_SAMPLE = 1000
    DIRICHLET_ALPHA = 1.0
    TEST_SEED = 999

    # Output
    OUTPUT_DIR = './hp_vade_testing'
    RESULTS_DIR = f'{OUTPUT_DIR}/results'
    PLOTS_DIR = f'{OUTPUT_DIR}/plots'

    # Checkpoint (will be set from args)
    CHECKPOINT_PATH = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_directories(config):
    """Create output directories."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    print(f"✓ Created output directories in: {config.OUTPUT_DIR}")


def find_best_checkpoint(experiment_name=None):
    """Find the best checkpoint automatically."""
    checkpoint_dir = Path('./hp_vade_training/checkpoints')

    if not checkpoint_dir.exists():
        return None

    # Find experiment directories
    if experiment_name:
        exp_dirs = list(checkpoint_dir.glob(f'*{experiment_name}*'))
    else:
        exp_dirs = list(checkpoint_dir.glob('hp_vade_*'))

    if not exp_dirs:
        return None

    # Get the most recent experiment
    latest_exp = max(exp_dirs, key=lambda x: x.stat().st_mtime)

    # Find best checkpoint (lowest val_loss)
    checkpoints = list(latest_exp.glob('hp_vade-epoch=*-val_loss=*.ckpt'))

    if not checkpoints:
        # Try last.ckpt
        last_ckpt = latest_exp / 'last.ckpt'
        if last_ckpt.exists():
            return str(last_ckpt)
        return None

    # Sort by val_loss (extract from filename)
    def extract_val_loss(ckpt_path):
        name = ckpt_path.stem
        try:
            val_loss = float(name.split('val_loss=')[1])
            return val_loss
        except:
            return float('inf')

    best_ckpt = min(checkpoints, key=extract_val_loss)
    return str(best_ckpt)


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    print(f"\nLoading model from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load model
    model = HP_VADE.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Move to device
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        print(f"✓ Model loaded on GPU")
    else:
        model = model.cpu()
        print(f"✓ Model loaded on CPU")

    return model


def load_test_data(config):
    """Load test dataset."""
    print(f"\nLoading test data from: {config.TEST_DATA_PATH}")

    if not os.path.exists(config.TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data not found: {config.TEST_DATA_PATH}")

    adata_test = sc.read_h5ad(config.TEST_DATA_PATH)
    print(f"✓ Test data loaded: {adata_test.shape}")
    print(f"✓ Test cells: {adata_test.n_obs:,}")
    print(f"✓ Genes: {adata_test.n_vars:,}")

    return adata_test


def get_cell_type_names(config):
    """Get cell type names from training data."""
    adata_train = sc.read_h5ad(config.TRAIN_DATA_PATH)
    cell_types = adata_train.obs['cell_type'].astype('category')
    cell_types = cell_types.cat.remove_unused_categories()
    cell_type_names = list(cell_types.cat.categories)
    return cell_type_names


def generate_test_bulk(adata_test, config):
    """Generate test bulk samples."""
    print(f"\nGenerating {config.N_TEST_BULK_SAMPLES:,} test bulk samples...")

    bulk_test, props_test = create_simulated_bulk(
        adata_test,
        n_samples=config.N_TEST_BULK_SAMPLES,
        cells_per_sample=config.CELLS_PER_SAMPLE,
        alpha=config.DIRICHLET_ALPHA,
        seed=config.TEST_SEED
    )

    print(f"✓ Test bulk data: {bulk_test.shape}")
    print(f"✓ Test proportions: {props_test.shape}")

    return bulk_test, props_test


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_deconvolution(model, bulk_test, props_test, device='cuda'):
    """Evaluate deconvolution performance."""
    print("\n" + "=" * 80)
    print("EVALUATING DECONVOLUTION PERFORMANCE")
    print("=" * 80)

    # Convert to tensors
    bulk_tensor = torch.FloatTensor(bulk_test)

    if device == 'cuda' and torch.cuda.is_available():
        bulk_tensor = bulk_tensor.cuda()

    # Predict proportions
    print("Predicting cell type proportions...")
    with torch.no_grad():
        props_pred = model(bulk_tensor)
        props_pred = props_pred.cpu().numpy()

    # Compute metrics
    print("\nComputing metrics...")

    # Overall metrics
    mae = mean_absolute_error(props_test, props_pred)
    rmse = np.sqrt(mean_squared_error(props_test, props_pred))

    # Per-sample correlation
    correlations = []
    for i in range(len(props_test)):
        if props_test[i].std() > 0 and props_pred[i].std() > 0:
            corr, _ = pearsonr(props_test[i], props_pred[i])
            correlations.append(corr)

    mean_corr = np.mean(correlations)

    # Per-cell-type metrics
    n_cell_types = props_test.shape[1]
    per_type_mae = []
    per_type_corr = []

    for ct in range(n_cell_types):
        ct_mae = mean_absolute_error(props_test[:, ct], props_pred[:, ct])
        per_type_mae.append(ct_mae)

        if props_test[:, ct].std() > 0 and props_pred[:, ct].std() > 0:
            ct_corr, _ = pearsonr(props_test[:, ct], props_pred[:, ct])
            per_type_corr.append(ct_corr)
        else:
            per_type_corr.append(0.0)

    # Print results
    print("\n" + "-" * 80)
    print("OVERALL METRICS")
    print("-" * 80)
    print(f"Mean Absolute Error (MAE):        {mae:.6f}")
    print(f"Root Mean Squared Error (RMSE):   {rmse:.6f}")
    print(f"Mean Per-Sample Correlation:      {mean_corr:.6f}")

    print("\n" + "-" * 80)
    print("PER-CELL-TYPE METRICS")
    print("-" * 80)
    for ct in range(n_cell_types):
        print(f"Cell Type {ct}:")
        print(f"  MAE:         {per_type_mae[ct]:.6f}")
        print(f"  Correlation: {per_type_corr[ct]:.6f}")

    results = {
        'props_test': props_test,
        'props_pred': props_pred,
        'overall_mae': mae,
        'overall_rmse': rmse,
        'mean_correlation': mean_corr,
        'per_type_mae': per_type_mae,
        'per_type_corr': per_type_corr,
        'all_correlations': correlations
    }

    return results


def evaluate_reconstruction(model, adata_test, device='cuda', n_samples=1000):
    """Evaluate single-cell reconstruction quality."""
    print("\n" + "=" * 80)
    print("EVALUATING SINGLE-CELL RECONSTRUCTION")
    print("=" * 80)

    # Sample random cells for evaluation
    n_test_cells = min(n_samples, adata_test.n_obs)
    indices = np.random.choice(adata_test.n_obs, n_test_cells, replace=False)

    # Get data
    sc_data = adata_test.X[indices]
    if hasattr(sc_data, 'toarray'):
        sc_data = sc_data.toarray()

    sc_tensor = torch.FloatTensor(sc_data)
    if device == 'cuda' and torch.cuda.is_available():
        sc_tensor = sc_tensor.cuda()

    # Encode and reconstruct
    print(f"Reconstructing {n_test_cells:,} cells...")
    with torch.no_grad():
        mu, logvar = model.encode(sc_tensor)
        z = mu  # Use mean for reconstruction
        sc_rec = model.decode(z)
        sc_rec = sc_rec.cpu().numpy()

    sc_data = sc_tensor.cpu().numpy()

    # Compute metrics
    recon_mse = mean_squared_error(sc_data, sc_rec)
    recon_mae = mean_absolute_error(sc_data, sc_rec)

    # Per-cell correlation
    per_cell_corr = []
    for i in range(n_test_cells):
        if sc_data[i].std() > 0 and sc_rec[i].std() > 0:
            corr, _ = pearsonr(sc_data[i], sc_rec[i])
            per_cell_corr.append(corr)

    mean_cell_corr = np.mean(per_cell_corr)

    print("\n" + "-" * 80)
    print("RECONSTRUCTION METRICS")
    print("-" * 80)
    print(f"Mean Squared Error (MSE):     {recon_mse:.6f}")
    print(f"Mean Absolute Error (MAE):    {recon_mae:.6f}")
    print(f"Mean Per-Cell Correlation:    {mean_cell_corr:.6f}")

    recon_results = {
        'recon_mse': recon_mse,
        'recon_mae': recon_mae,
        'mean_cell_corr': mean_cell_corr,
        'per_cell_corr': per_cell_corr
    }

    return recon_results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_scatter_by_celltype(results, cell_type_names, config):
    """Create scatter plots of predicted vs true proportions for each cell type."""
    print("\nGenerating per-cell-type scatter plots...")

    props_test = results['props_test']
    props_pred = results['props_pred']
    n_cell_types = props_test.shape[1]

    # Create figure
    n_cols = 4
    n_rows = (n_cell_types + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_cell_types > 1 else [axes]

    for ct in range(n_cell_types):
        ax = axes[ct]

        # Scatter plot
        ax.scatter(props_test[:, ct], props_pred[:, ct], alpha=0.3, s=10)

        # Perfect prediction line
        max_val = max(props_test[:, ct].max(), props_pred[:, ct].max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect')

        # Labels and title
        ct_name = cell_type_names[ct] if ct < len(cell_type_names) else f'Type {ct}'
        ax.set_xlabel('True Proportion', fontsize=10)
        ax.set_ylabel('Predicted Proportion', fontsize=10)
        ax.set_title(f'{ct_name}\nMAE={results["per_type_mae"][ct]:.4f}, r={results["per_type_corr"][ct]:.3f}',
                    fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_cell_types, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    save_path = Path(config.PLOTS_DIR) / 'scatter_by_celltype.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_correlation_heatmap(results, cell_type_names, config):
    """Create heatmap of predicted vs true proportions."""
    print("\nGenerating correlation heatmap...")

    props_test = results['props_test']
    props_pred = results['props_pred']
    n_cell_types = props_test.shape[1]

    # Compute correlation matrix between predicted and true for each cell type
    corr_matrix = np.zeros((n_cell_types, n_cell_types))

    for i in range(n_cell_types):
        for j in range(n_cell_types):
            if props_test[:, i].std() > 0 and props_pred[:, j].std() > 0:
                corr_matrix[i, j], _ = pearsonr(props_test[:, i], props_pred[:, j])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    labels = [cell_type_names[i] if i < len(cell_type_names) else f'Type {i}'
              for i in range(n_cell_types)]

    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                xticklabels=labels, yticklabels=labels, ax=ax,
                vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation'})

    ax.set_xlabel('Predicted Cell Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Cell Type', fontsize=12, fontweight='bold')
    ax.set_title('Correlation Matrix: True vs Predicted Proportions',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    save_path = Path(config.PLOTS_DIR) / 'correlation_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_overall_scatter(results, config):
    """Create overall scatter plot of all predictions."""
    print("\nGenerating overall scatter plot...")

    props_test = results['props_test']
    props_pred = results['props_pred']

    fig, ax = plt.subplots(figsize=(8, 8))

    # Flatten all proportions
    true_flat = props_test.flatten()
    pred_flat = props_pred.flatten()

    # Hexbin plot for better visualization of density
    hb = ax.hexbin(true_flat, pred_flat, gridsize=50, cmap='viridis', mincnt=1)

    # Perfect prediction line
    max_val = max(true_flat.max(), pred_flat.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('True Proportion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Proportion', fontsize=12, fontweight='bold')
    ax.set_title(f'Overall Deconvolution Performance\nMAE={results["overall_mae"]:.4f}, RMSE={results["overall_rmse"]:.4f}',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label('Count', fontsize=10)

    plt.tight_layout()
    save_path = Path(config.PLOTS_DIR) / 'overall_scatter.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_mae_by_celltype(results, cell_type_names, config):
    """Bar plot of MAE by cell type."""
    print("\nGenerating MAE bar plot...")

    per_type_mae = results['per_type_mae']
    n_cell_types = len(per_type_mae)

    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [cell_type_names[i] if i < len(cell_type_names) else f'Type {i}'
              for i in range(n_cell_types)]

    colors = plt.cm.viridis(np.linspace(0, 1, n_cell_types))
    bars = ax.bar(range(n_cell_types), per_type_mae, color=colors, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Cell Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    ax.set_title('Deconvolution Error by Cell Type', fontsize=14, fontweight='bold')
    ax.set_xticks(range(n_cell_types))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, mae) in enumerate(zip(bars, per_type_mae)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = Path(config.PLOTS_DIR) / 'mae_by_celltype.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_correlation_distribution(results, config):
    """Histogram of per-sample correlations."""
    print("\nGenerating correlation distribution plot...")

    correlations = results['all_correlations']

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(correlations, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(correlations), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(correlations):.3f}')
    ax.axvline(np.median(correlations), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(correlations):.3f}')

    ax.set_xlabel('Pearson Correlation (per sample)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Per-Sample Correlations', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = Path(config.PLOTS_DIR) / 'correlation_distribution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(results, recon_results, cell_type_names, config):
    """Save numerical results to files."""
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    results_dir = Path(config.RESULTS_DIR)

    # Save predictions
    np.save(results_dir / 'props_test_true.npy', results['props_test'])
    np.save(results_dir / 'props_test_pred.npy', results['props_pred'])
    print(f"✓ Saved: {results_dir / 'props_test_true.npy'}")
    print(f"✓ Saved: {results_dir / 'props_test_pred.npy'}")

    # Save metrics as CSV
    metrics_df = pd.DataFrame({
        'cell_type': [cell_type_names[i] if i < len(cell_type_names) else f'Type_{i}'
                     for i in range(len(results['per_type_mae']))],
        'mae': results['per_type_mae'],
        'correlation': results['per_type_corr']
    })
    metrics_df.to_csv(results_dir / 'per_celltype_metrics.csv', index=False)
    print(f"✓ Saved: {results_dir / 'per_celltype_metrics.csv'}")

    # Save summary
    summary_path = results_dir / 'test_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HP-VADE TEST RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("DECONVOLUTION PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Overall MAE:              {results['overall_mae']:.6f}\n")
        f.write(f"Overall RMSE:             {results['overall_rmse']:.6f}\n")
        f.write(f"Mean Sample Correlation:  {results['mean_correlation']:.6f}\n")
        f.write(f"Median Sample Correlation: {np.median(results['all_correlations']):.6f}\n")
        f.write(f"Min Sample Correlation:   {np.min(results['all_correlations']):.6f}\n")
        f.write(f"Max Sample Correlation:   {np.max(results['all_correlations']):.6f}\n\n")

        f.write("PER-CELL-TYPE PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        for i, (mae, corr) in enumerate(zip(results['per_type_mae'], results['per_type_corr'])):
            ct_name = cell_type_names[i] if i < len(cell_type_names) else f'Type_{i}'
            f.write(f"{ct_name:30s}  MAE: {mae:.6f}  Corr: {corr:.6f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("RECONSTRUCTION PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Reconstruction MSE:       {recon_results['recon_mse']:.6f}\n")
        f.write(f"Reconstruction MAE:       {recon_results['recon_mae']:.6f}\n")
        f.write(f"Mean Cell Correlation:    {recon_results['mean_cell_corr']:.6f}\n")
        f.write(f"Median Cell Correlation:  {np.median(recon_results['per_cell_corr']):.6f}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"✓ Saved: {summary_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(args):
    """Main testing workflow."""

    print("\n" + "=" * 80)
    print("HP-VADE MODEL TESTING")
    print("=" * 80)

    # Configuration
    config = TestConfig()

    if args.data_dir:
        config.DATA_DIR = args.data_dir
        config.TEST_DATA_PATH = f'{args.data_dir}/adata_test.h5ad'
        config.TRAIN_DATA_PATH = f'{args.data_dir}/adata_train.h5ad'

    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
        config.RESULTS_DIR = f'{args.output_dir}/results'
        config.PLOTS_DIR = f'{args.output_dir}/plots'

    # Setup directories
    setup_directories(config)

    # Find or load checkpoint
    if args.auto:
        print("\nAuto-detecting best checkpoint...")
        checkpoint_path = find_best_checkpoint(args.experiment)
        if not checkpoint_path:
            print("ERROR: No checkpoint found!")
            print("Please specify checkpoint with --checkpoint or train a model first.")
            return
        print(f"✓ Found checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = args.checkpoint

    config.CHECKPOINT_PATH = checkpoint_path

    # Determine device
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load model
    model = load_model(checkpoint_path, device)

    # Load test data
    adata_test = load_test_data(config)

    # Get cell type names
    cell_type_names = get_cell_type_names(config)
    print(f"\nCell types: {cell_type_names}")

    # Generate test bulk samples
    bulk_test, props_test = generate_test_bulk(adata_test, config)

    # Evaluate deconvolution
    results = evaluate_deconvolution(model, bulk_test, props_test, device)

    # Evaluate reconstruction
    recon_results = evaluate_reconstruction(model, adata_test, device, n_samples=1000)

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_overall_scatter(results, config)
    plot_scatter_by_celltype(results, cell_type_names, config)
    plot_correlation_heatmap(results, cell_type_names, config)
    plot_mae_by_celltype(results, cell_type_names, config)
    plot_correlation_distribution(results, config)

    # Save results
    save_results(results, recon_results, cell_type_names, config)

    print("\n" + "=" * 80)
    print("✅ TESTING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print(f"Plots saved to: {config.PLOTS_DIR}")
    print(f"\nView summary: cat {config.RESULTS_DIR}/test_summary.txt")
    print("=" * 80 + "\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Test HP-VADE model on held-out test data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect latest checkpoint
  python test_hp_vade.py --auto

  # Specify checkpoint
  python test_hp_vade.py --checkpoint ./hp_vade_training/checkpoints/.../best.ckpt

  # Custom data directory
  python test_hp_vade.py --auto --data-dir /path/to/data

  # Run on CPU
  python test_hp_vade.py --auto --cpu
        """
    )

    # Checkpoint arguments
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--auto', action='store_true',
                       help='Automatically find best checkpoint')
    parser.add_argument('--experiment', type=str,
                       help='Experiment name to search for (with --auto)')

    # Data arguments
    parser.add_argument('--data-dir', type=str,
                       help='Directory containing test data')
    parser.add_argument('--output-dir', type=str,
                       help='Directory for outputs (default: ./hp_vade_testing)')

    # Hardware
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')

    args = parser.parse_args()

    # Validation
    if not args.auto and not args.checkpoint:
        parser.error("Either --auto or --checkpoint must be specified")

    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
