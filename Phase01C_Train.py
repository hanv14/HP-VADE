import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import os
from Phase01C_Model import HP_VADE, create_model


def train_hp_vade(
    train_dataloader,
    val_dataloader,
    adata_train=None,  # Training AnnData for signature matrix initialization
    input_dim=2000,
    latent_dim=32,
    n_cell_types=8,
    n_hidden=128,
    # Loss weights
    lambda_proto=0.1,  # Updated default
    lambda_bulk_recon=10.0,  # Updated default
    lambda_bulk=5.0,  # Updated default
    lambda_kl=0.01,  # Updated default
    # Training parameters
    learning_rate=1e-3,
    max_epochs=100,
    patience=20,
    use_gpu=True,
    experiment_name="hp_vade_experiment",
):
    """
    Fixed training function compatible with PyTorch Lightning 2.0+
    
    This version automatically detects your PyTorch Lightning version
    and uses the correct API for GPU configuration.
    """
    
    print("=" * 80)
    print("HP-VADE Training Configuration")
    print("=" * 80)
    print(f"Architecture Parameters:")
    print(f"  - Input dimension: {input_dim}")
    print(f"  - Latent dimension: {latent_dim}")
    print(f"  - Number of cell types: {n_cell_types}")
    print(f"  - Hidden layer size: {n_hidden}")
    print(f"\nLoss Weights:")
    print(f"  - λ_proto: {lambda_proto}")
    print(f"  - λ_bulk_recon: {lambda_bulk_recon}")
    print(f"  - λ_bulk: {lambda_bulk}")
    print(f"  - λ_kl: {lambda_kl}")
    print(f"\nTraining Parameters:")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Max epochs: {max_epochs}")
    print(f"  - Early stopping patience: {patience}")
    
    # Check PyTorch Lightning version
    import pytorch_lightning
    pl_version_str = pytorch_lightning.__version__
    print(f"  - PyTorch Lightning version: {pl_version_str}")
    
    # Parse version
    try:
        pl_version = tuple(map(int, pl_version_str.split('.')[:2]))
    except:
        # If version parsing fails, assume newer version
        pl_version = (2, 0)
    
    # Check for GPU
    has_gpu = torch.cuda.is_available()
    print(f"  - GPU available: {has_gpu}")
    if has_gpu and use_gpu:
        print(f"  - GPU device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 1. Create the model
    # -------------------------------------------------------------------------
    model = create_model(
        input_dim=input_dim,
        latent_dim=latent_dim,
        n_cell_types=n_cell_types,
        n_hidden=n_hidden,
        lambda_proto=lambda_proto,
        lambda_bulk_recon=lambda_bulk_recon,
        lambda_bulk=lambda_bulk,
        lambda_kl=lambda_kl,
        learning_rate=learning_rate
    )

    # Initialize signature matrix from cell type means (CRITICAL FOR DECONVOLUTION!)
    if adata_train is not None:
        print("\n" + "=" * 80)
        print("INITIALIZING SIGNATURE MATRIX FROM CELL TYPE MEANS")
        print("=" * 80)
        model.init_signature_from_celltype_means(adata_train)
        print("=" * 80 + "\n")
    else:
        print("\n⚠️  WARNING: adata_train not provided - signature matrix will be random!")
        print("⚠️  This may lead to poor deconvolution performance!\n")

    # -------------------------------------------------------------------------
    # 2. Setup callbacks
    # -------------------------------------------------------------------------
    os.makedirs(f'checkpoints/{experiment_name}', exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{experiment_name}',
        filename='hp_vade-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0001,
        patience=patience,
        verbose=True,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]
    
    # -------------------------------------------------------------------------
    # 3. Setup logger
    # -------------------------------------------------------------------------
    logger = TensorBoardLogger(
        save_dir='logs',
        name=experiment_name,
        default_hp_metric=False
    )
    
    # -------------------------------------------------------------------------
    # 4. Create trainer with version-appropriate GPU settings
    # -------------------------------------------------------------------------
    trainer_kwargs = {
        'max_epochs': max_epochs,
        'callbacks': callbacks,
        'logger': logger,
        'gradient_clip_val': 1.0,
        'log_every_n_steps': 10,
        'check_val_every_n_epoch': 1,
        'enable_progress_bar': True,
        'enable_model_summary': True,
    }
    
    # Configure GPU/CPU based on PyTorch Lightning version
    if pl_version >= (2, 0):
        # PyTorch Lightning 2.0+ uses 'accelerator' and 'devices'
        if has_gpu and use_gpu:
            trainer_kwargs['accelerator'] = 'gpu'
            trainer_kwargs['devices'] = 1
            print("\n✓ Using GPU acceleration (PL 2.0+ API)")
        else:
            trainer_kwargs['accelerator'] = 'cpu'
            trainer_kwargs['devices'] = 'auto'
            print("\n✓ Using CPU (PL 2.0+ API)")
    else:
        # PyTorch Lightning < 2.0 uses 'gpus'
        if has_gpu and use_gpu:
            trainer_kwargs['gpus'] = 1
            print("\n✓ Using GPU acceleration (PL 1.x API)")
        else:
            trainer_kwargs['gpus'] = 0
            print("\n✓ Using CPU (PL 1.x API)")
    
    # Create trainer
    trainer = pl.Trainer(**trainer_kwargs)
    
    # -------------------------------------------------------------------------
    # 5. Train the model
    # -------------------------------------------------------------------------
    print("\nStarting training...")
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # -------------------------------------------------------------------------
    # 6. Load best checkpoint
    # -------------------------------------------------------------------------
    best_model_path = checkpoint_callback.best_model_path
    print(f"\nBest model saved at: {best_model_path}")
    
    # Load the best model
    model = HP_VADE.load_from_checkpoint(best_model_path)
    
    # Save signature matrix
    import numpy as np
    S = model.get_signature_matrix()
    signature_path = f'checkpoints/{experiment_name}/signature_matrix.npy'
    np.save(signature_path, S)
    print(f"Signature matrix saved at: {signature_path}")
    
    return model, trainer


# ============================================================================
# QUICK TEST FUNCTION
# ============================================================================

def quick_test_training(train_dataloader, val_dataloader, max_epochs=2):
    """
    Quick test to ensure training works with your data.
    Runs only 2 epochs by default.
    """
    print("\n" + "=" * 80)
    print("QUICK TEST - Running 2 epochs to verify setup")
    print("=" * 80)
    
    model, trainer = train_hp_vade(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        max_epochs=max_epochs,
        patience=5,
        experiment_name="hp_vade_quick_test"
    )
    
    print("\n✅ Quick test completed successfully!")
    print("If training ran without errors, you can now run full training.")
    
    return model, trainer
