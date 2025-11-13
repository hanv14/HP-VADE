"""
HP-VADE Training Monitor
========================
Monitor and visualize training progress for remote server execution.
Generates plots and summaries from TensorBoard logs without GUI.

Usage:
    python monitor_training.py --log-dir ./hp_vade_training/logs
    python monitor_training.py --watch  # Auto-refresh every 30 seconds

Author: HP-VADE Development Team
"""

import os
import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for remote servers
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("WARNING: TensorBoard not installed. Install with: pip install tensorboard")


def find_latest_experiment(log_dir):
    """Find the most recent experiment directory."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return None

    # Look for experiment directories
    experiment_dirs = [d for d in log_path.iterdir() if d.is_dir()]
    if not experiment_dirs:
        return None

    # Return the most recently modified
    latest = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
    return latest


def load_tensorboard_logs(log_dir):
    """Load metrics from TensorBoard event files."""
    if not TENSORBOARD_AVAILABLE:
        return None

    # Find version directory
    version_dirs = list(Path(log_dir).glob('version_*'))
    if not version_dirs:
        return None

    latest_version = max(version_dirs, key=lambda x: x.stat().st_mtime)

    # Load events
    event_acc = EventAccumulator(str(latest_version))
    event_acc.Reload()

    # Extract scalars
    metrics = defaultdict(list)
    for tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        metrics[tag] = [(e.step, e.value) for e in events]

    return metrics


def plot_training_curves(metrics, output_dir):
    """Generate training curve plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define metric groups
    metric_groups = {
        'Total Loss': ['train_loss', 'val_loss'],
        'Single-Cell Losses': [
            'sc/loss_recon', 'sc/loss_kl', 'sc/loss_proto', 'sc/loss_total',
            'val_sc/loss_recon', 'val_sc/loss_kl', 'val_sc/loss_proto'
        ],
        'Bulk Losses': [
            'bulk/loss_prop', 'bulk/loss_recon', 'bulk/loss_total',
            'val_bulk/loss_prop', 'val_bulk/loss_recon'
        ],
        'Latent Metrics': [
            'metrics/mu_mean', 'metrics/mu_std', 'metrics/logvar_mean', 'metrics/S_norm'
        ]
    }

    # Plot each group
    for group_name, metric_names in metric_groups.items():
        fig, ax = plt.subplots(figsize=(12, 6))

        plotted_any = False
        for metric_name in metric_names:
            if metric_name in metrics:
                steps, values = zip(*metrics[metric_name])
                label = metric_name.replace('/', '_')
                ax.plot(steps, values, label=label, alpha=0.8, linewidth=2)
                plotted_any = True

        if plotted_any:
            ax.set_xlabel('Step', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title(f'{group_name}', fontsize=14, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save figure
            filename = group_name.lower().replace(' ', '_') + '.png'
            save_path = output_path / filename
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {save_path}")


def generate_summary_report(metrics, output_dir):
    """Generate a text summary report."""
    output_path = Path(output_dir) / 'training_summary.txt'

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HP-VADE TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Get final values for key metrics
        key_metrics = {
            'Training Loss': 'train_loss',
            'Validation Loss': 'val_loss',
            'SC Reconstruction': 'sc/loss_recon',
            'SC KL Divergence': 'sc/loss_kl',
            'SC Prototype Loss': 'sc/loss_proto',
            'Bulk Proportion Loss': 'bulk/loss_prop',
            'Bulk Reconstruction': 'bulk/loss_recon',
            'Val SC Reconstruction': 'val_sc/loss_recon',
            'Val Bulk Proportion': 'val_bulk/loss_prop'
        }

        f.write("Final Metric Values:\n")
        f.write("-" * 80 + "\n")

        for name, metric_key in key_metrics.items():
            if metric_key in metrics and metrics[metric_key]:
                final_value = metrics[metric_key][-1][1]
                f.write(f"  {name:30s}: {final_value:.6f}\n")

        f.write("\n" + "=" * 80 + "\n")

        # Training progress
        if 'train_loss' in metrics:
            steps = [s for s, v in metrics['train_loss']]
            f.write(f"\nTraining Progress:\n")
            f.write(f"  Total steps: {steps[-1] if steps else 0}\n")
            f.write(f"  Number of updates: {len(steps)}\n")

        # Best validation loss
        if 'val_loss' in metrics:
            val_losses = [v for s, v in metrics['val_loss']]
            if val_losses:
                best_val_loss = min(val_losses)
                best_val_step = [s for s, v in metrics['val_loss'] if v == best_val_loss][0]
                f.write(f"\nBest Validation Performance:\n")
                f.write(f"  Best val_loss: {best_val_loss:.6f}\n")
                f.write(f"  At step: {best_val_step}\n")

    print(f"  ✓ Saved: {output_path}")


def create_comparison_table(metrics, output_dir):
    """Create a comparison table of train vs validation metrics."""
    train_val_pairs = [
        ('train_loss', 'val_loss', 'Total Loss'),
        ('sc/loss_recon', 'val_sc/loss_recon', 'SC Reconstruction'),
        ('sc/loss_kl', 'val_sc/loss_kl', 'SC KL Divergence'),
        ('sc/loss_proto', 'val_sc/loss_proto', 'SC Prototype'),
        ('bulk/loss_prop', 'val_bulk/loss_prop', 'Bulk Proportion'),
        ('bulk/loss_recon', 'val_bulk/loss_recon', 'Bulk Reconstruction')
    ]

    data = []
    for train_key, val_key, name in train_val_pairs:
        row = {'Metric': name}

        if train_key in metrics and metrics[train_key]:
            row['Train (Final)'] = metrics[train_key][-1][1]
            row['Train (Min)'] = min(v for s, v in metrics[train_key])
        else:
            row['Train (Final)'] = 'N/A'
            row['Train (Min)'] = 'N/A'

        if val_key in metrics and metrics[val_key]:
            row['Val (Final)'] = metrics[val_key][-1][1]
            row['Val (Min)'] = min(v for s, v in metrics[val_key])
        else:
            row['Val (Final)'] = 'N/A'
            row['Val (Min)'] = 'N/A'

        data.append(row)

    df = pd.DataFrame(data)

    # Save as CSV
    csv_path = Path(output_dir) / 'metrics_comparison.csv'
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"  ✓ Saved: {csv_path}")

    # Also save as formatted text table
    txt_path = Path(output_dir) / 'metrics_comparison.txt'
    with open(txt_path, 'w') as f:
        f.write(df.to_string(index=False))
    print(f"  ✓ Saved: {txt_path}")


def monitor_training(log_dir, output_dir, watch=False, interval=30):
    """Main monitoring function."""
    print("\n" + "=" * 80)
    print("HP-VADE TRAINING MONITOR")
    print("=" * 80)

    if watch:
        print(f"\nWatching directory: {log_dir}")
        print(f"Refresh interval: {interval} seconds")
        print("Press Ctrl+C to stop\n")

    try:
        while True:
            # Find experiment directory
            if not os.path.exists(log_dir):
                print(f"Log directory not found: {log_dir}")
                if watch:
                    print(f"Waiting for logs... (checking again in {interval}s)")
                    time.sleep(interval)
                    continue
                else:
                    return

            # Load metrics
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loading metrics...")
            metrics = load_tensorboard_logs(log_dir)

            if not metrics:
                print("No metrics found yet.")
                if watch:
                    print(f"Waiting... (checking again in {interval}s)")
                    time.sleep(interval)
                    continue
                else:
                    return

            print(f"Found {len(metrics)} metric types")

            # Generate plots
            print("\nGenerating plots...")
            plot_training_curves(metrics, output_dir)

            # Generate reports
            print("\nGenerating reports...")
            generate_summary_report(metrics, output_dir)
            create_comparison_table(metrics, output_dir)

            print("\n" + "=" * 80)
            print("✅ Monitoring update complete!")
            print("=" * 80)
            print(f"\nOutputs saved to: {output_dir}")

            # Quick summary to console
            if 'train_loss' in metrics and metrics['train_loss']:
                latest_train = metrics['train_loss'][-1][1]
                print(f"\nLatest train_loss: {latest_train:.6f}")

            if 'val_loss' in metrics and metrics['val_loss']:
                latest_val = metrics['val_loss'][-1][1]
                best_val = min(v for s, v in metrics['val_loss'])
                print(f"Latest val_loss:   {latest_val:.6f}")
                print(f"Best val_loss:     {best_val:.6f}")

            if not watch:
                break

            print(f"\nNext update in {interval} seconds... (Ctrl+C to stop)")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")


def main():
    parser = argparse.ArgumentParser(
        description='Monitor HP-VADE training progress',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--log-dir', type=str, default='./hp_vade_training/logs',
                        help='TensorBoard log directory (default: ./hp_vade_training/logs)')
    parser.add_argument('--output-dir', type=str, default='./hp_vade_training/monitoring',
                        help='Output directory for plots and reports (default: ./hp_vade_training/monitoring)')
    parser.add_argument('--watch', action='store_true',
                        help='Continuously monitor and update (default: single update)')
    parser.add_argument('--interval', type=int, default=30,
                        help='Update interval in seconds for watch mode (default: 30)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run monitoring
    monitor_training(args.log_dir, args.output_dir, args.watch, args.interval)


if __name__ == '__main__':
    main()
