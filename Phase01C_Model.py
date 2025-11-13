"""
HP-VADE Model Architecture: PyTorch/PyTorch Lightning Implementation
=====================================================================
This file contains the complete architecture for the HP-VADE (Hierarchical Prototype
Variational Autoencoder for Deconvolution) method.

Architecture Overview:
- Phase I-C: Component Modules (Encoder, Decoder, DeconvolutionNetwork)
- Phase I-D: Orchestration Module (HP_VADE LightningModule)

Author: HP-VADE Development Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Tuple, Optional


# ============================================================================
# PHASE I-C: COMPONENT ARCHITECTURE (nn.Module)
# ============================================================================

class Encoder(nn.Module):
    """
    Encoder network for the VAE component.
    
    Maps gene expression vectors to latent distributions (mu, logvar).
    
    Args:
        in_features: Number of input genes (e.g., 2000)
        latent_dim: Dimensionality of the latent space (e.g., 32)
        n_hidden: Size of the intermediate layer (default: 128)
    """
    
    def __init__(self, in_features: int, latent_dim: int, n_hidden: int = 128):
        super().__init__()
        
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.n_hidden = n_hidden
        
        # Define the MLP architecture
        # Output is latent_dim * 2 to hold both mu and logvar
        self.mlp = nn.Sequential(
            nn.Linear(in_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, latent_dim * 2)
        )
        
        # Initialize weights for better convergence
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            mu: Mean of the latent distribution (batch_size, latent_dim)
            logvar: Log variance of the latent distribution (batch_size, latent_dim)
        """
        # Pass through MLP
        output = self.mlp(x)  # Shape: (batch_size, latent_dim * 2)
        
        # Split into mu and logvar
        mu = output[:, :self.latent_dim]      # First half
        logvar = output[:, self.latent_dim:]  # Second half
        
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder network for the VAE component.
    
    Maps latent vectors back to gene expression space.
    
    Args:
        latent_dim: Dimensionality of the latent space (must match Encoder)
        out_features: Number of output genes (must match Encoder's in_features)
        n_hidden: Size of the intermediate layer (default: 128)
    """
    
    def __init__(self, latent_dim: int, out_features: int, n_hidden: int = 128):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.out_features = out_features
        self.n_hidden = n_hidden
        
        # Define the MLP architecture (symmetric to encoder)
        # CRITICAL: No activation after final layer (reconstructing log-normalized data)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, out_features)  # Linear output, no activation
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            x_rec: Reconstructed expression vector (batch_size, out_features)
        """
        x_rec = self.mlp(z)
        return x_rec


class DeconvolutionNetwork(nn.Module):
    """
    Deconvolution network for predicting cell type proportions from bulk data.
    
    Args:
        in_features: Number of input genes (e.g., 2000)
        n_cell_types: Number of cell types to predict (e.g., 8 PBMC types)
        n_hidden: Size of the intermediate layer (default: 128)
    """
    
    def __init__(self, in_features: int, n_cell_types: int, n_hidden: int = 128):
        super().__init__()
        
        self.in_features = in_features
        self.n_cell_types = n_cell_types
        self.n_hidden = n_hidden
        
        # Define the MLP architecture
        # CRITICAL: Softmax at the end to ensure proportions sum to 1
        self.mlp = nn.Sequential(
            nn.Linear(in_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_cell_types),
            nn.Softmax(dim=1)  # dim=1 for batch dimension
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the deconvolution network.
        
        Args:
            b: Bulk expression vector of shape (batch_size, in_features)
            
        Returns:
            p_pred: Predicted cell type proportions (batch_size, n_cell_types)
                   Each row sums to 1.0
        """
        p_pred = self.mlp(b)
        return p_pred


# ============================================================================
# PHASE I-D: ORCHESTRATION MODULE (pl.LightningModule)
# ============================================================================

class HP_VADE(pl.LightningModule):
    """
    Hierarchical Prototype Variational Autoencoder for Deconvolution.
    
    This is the main orchestration module that combines:
    1. VAE for single-cell data (Encoder + Decoder)
    2. Deconvolution network for bulk data
    3. Learnable Signature Matrix S connecting both paths
    
    Args:
        input_dim: Number of genes (e.g., 2000)
        latent_dim: Dimensionality of VAE latent space (e.g., 32)
        n_cell_types: Number of cell types (e.g., 8)
        n_hidden: Hidden layer size for all networks (default: 128)
        lambda_proto: Weight for prototype loss (default: 1.0)
        lambda_bulk_recon: Weight for bulk reconstruction loss (default: 0.5)
        lambda_bulk: Weight for entire bulk path (default: 1.0)
        lambda_kl: Weight for KL divergence loss (default: 0.1)
        learning_rate: Learning rate for Adam optimizer (default: 1e-3)
    """
    
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 n_cell_types: int,
                 n_hidden: int = 128,
                 lambda_proto: float = 1.0,
                 lambda_bulk_recon: float = 0.5,
                 lambda_bulk: float = 1.0,
                 lambda_kl: float = 0.1,
                 learning_rate: float = 1e-3):
        
        super().__init__()
        
        # Save all hyperparameters for logging and checkpointing
        self.save_hyperparameters()
        
        # --- 1. Instantiate Component Modules ---
        print(f"[HP-VADE] Initializing components:")
        print(f"  - Input dimension: {input_dim}")
        print(f"  - Latent dimension: {latent_dim}")
        print(f"  - Number of cell types: {n_cell_types}")
        print(f"  - Hidden layer size: {n_hidden}")
        
        self.encoder = Encoder(input_dim, latent_dim, n_hidden)
        self.decoder = Decoder(latent_dim, input_dim, n_hidden)
        self.deconv_net = DeconvolutionNetwork(input_dim, n_cell_types, n_hidden)
        
        # --- 2. Define the Learnable Signature Matrix (S) ---
        # This is the central parameter connecting the two paths
        # Shape: (input_dim, n_cell_types) - each column is a cell type prototype
        self.S = nn.Parameter(torch.empty(input_dim, n_cell_types))
        
        # Initialize S using Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.S)
        
        print(f"[HP-VADE] Signature matrix S initialized with shape: {self.S.shape}")
        print(f"[HP-VADE] Loss weights:")
        print(f"  - λ_proto: {lambda_proto}")
        print(f"  - λ_bulk_recon: {lambda_bulk_recon}")
        print(f"  - λ_bulk: {lambda_bulk}")
        print(f"  - λ_kl: {lambda_kl}")
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Implements the reparameterization trick for the VAE.
        
        During training: samples from the latent distribution
        During inference: returns the mean
        
        Args:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        if not self.training:
            # During inference, use the mean
            return mu
        
        # During training, sample using reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample from N(0,1)
        return mu + eps * std
    
    def forward(self, bulk_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference (deconvolution only).
        
        This is used when the model is in eval mode and we only want
        to predict cell type proportions from bulk data.
        
        Args:
            bulk_data: Bulk expression data (batch_size, input_dim)
            
        Returns:
            p_pred: Predicted cell type proportions (batch_size, n_cell_types)
        """
        return self.deconv_net(bulk_data)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Defines the training logic for one batch.
        
        This is the core method implementing the joint training of:
        1. VAE path for single-cell data
        2. Deconvolution path for bulk data
        3. Shared signature matrix S
        
        Args:
            batch: Dictionary containing:
                - 'sc_data': Single-cell expression data (batch_size, input_dim)
                - 'sc_label': Cell type labels (batch_size,) as integers
                - 'bulk_data': Bulk expression data (batch_size, input_dim)
                - 'bulk_prop': True cell type proportions (batch_size, n_cell_types)
            batch_idx: Index of the current batch
            
        Returns:
            total_loss: Combined loss for backpropagation
        """
        
        # ===================================================================
        # 1. THE SINGLE-CELL PATH (VAE)
        # ===================================================================
        
        sc_x = batch['sc_data']        # (batch_size, input_dim)
        sc_y = batch['sc_label']       # (batch_size,) - integer labels
        
        # VAE forward pass
        mu, logvar = self.encoder(sc_x)           # Encode to latent distribution
        z = self.reparameterize(mu, logvar)       # Sample from distribution
        sc_rec = self.decoder(z)                  # Reconstruct expression
        
        # --- Calculate Single-Cell Losses ---
        
        # L_recon: Reconstruction loss (compare to original cell)
        loss_recon = F.mse_loss(sc_rec, sc_x)
        
        # L_KL: KL divergence from N(0,1)
        # KL(q(z|x) || p(z)) where p(z) = N(0,1)
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        loss_kl = loss_kl.mean()  # Average over batch
        
        # L_proto: Novel prototype loss
        # Compare reconstruction to its corresponding prototype in S
        # self.S shape: (input_dim, n_cell_types)
        # We need to select the right prototype for each cell
        s_y_prototypes = self.S.T[sc_y]  # (batch_size, input_dim)
        loss_proto = F.mse_loss(sc_rec, s_y_prototypes)
        
        # Total single-cell loss
        loss_sc_total = (loss_recon + 
                        (self.hparams.lambda_kl * loss_kl) + 
                        (self.hparams.lambda_proto * loss_proto))
        
        # ===================================================================
        # 2. THE BULK PATH (DECONVOLUTION)
        # ===================================================================
        
        b_sim = batch['bulk_data']     # (batch_size, input_dim)
        p_true = batch['bulk_prop']    # (batch_size, n_cell_types)
        
        # Deconvolution forward pass
        p_pred = self.deconv_net(b_sim)  # Predict proportions
        
        # --- Calculate Bulk Losses ---

        # L_prop: Proportion prediction loss
        # Using KL divergence for comparing distributions
        # CRITICAL: Ensure p_true is properly normalized and stable
        p_true_safe = p_true.clamp(min=1e-10)  # Avoid zeros
        p_true_safe = p_true_safe / p_true_safe.sum(dim=1, keepdim=True)  # Renormalize to sum to 1

        # F.kl_div expects: input = log-probabilities, target = probabilities
        loss_prop = F.kl_div(p_pred.log(), p_true_safe, reduction='batchmean')

        # Alternative (simpler and more stable):
        # loss_prop = F.mse_loss(p_pred, p_true_safe)
        
        # L_bulk_recon: Reconstruct bulk data from S and predicted proportions
        # b_rec = S @ p_pred^T
        # Math: (input_dim, n_cell_types) @ (batch_size, n_cell_types)^T
        # We want: (batch_size, input_dim), so compute: p_pred @ S^T
        b_rec = torch.matmul(p_pred, self.S.T)  # (batch_size, input_dim)
        loss_bulk_recon = F.mse_loss(b_rec, b_sim)
        
        # Total bulk loss
        loss_bulk_total = loss_prop + (self.hparams.lambda_bulk_recon * loss_bulk_recon)
        
        # ===================================================================
        # 3. FINAL TOTAL LOSS
        # ===================================================================

        total_loss = loss_sc_total + (self.hparams.lambda_bulk * loss_bulk_total)

        # --- NaN Detection (for debugging) ---
        if torch.isnan(total_loss):
            print("\n⚠ WARNING: NaN detected in training!")
            print(f"  loss_recon: {loss_recon.item()}")
            print(f"  loss_kl: {loss_kl.item()}")
            print(f"  loss_proto: {loss_proto.item()}")
            print(f"  loss_prop: {loss_prop.item()}")
            print(f"  loss_bulk_recon: {loss_bulk_recon.item()}")
            print(f"  sc_rec range: [{sc_rec.min().item():.2f}, {sc_rec.max().item():.2f}]")
            print(f"  b_rec range: [{b_rec.min().item():.2f}, {b_rec.max().item():.2f}]")
            print(f"  p_pred range: [{p_pred.min().item():.4f}, {p_pred.max().item():.4f}]")
            print(f"  S range: [{self.S.min().item():.2f}, {self.S.max().item():.2f}]")

        # --- Comprehensive Logging for Debugging ---
        self.log('train_loss', total_loss, prog_bar=True)
        
        # Single-cell losses
        self.log('sc/loss_recon', loss_recon)
        self.log('sc/loss_kl', loss_kl)
        self.log('sc/loss_proto', loss_proto)
        self.log('sc/loss_total', loss_sc_total)
        
        # Bulk losses
        self.log('bulk/loss_prop', loss_prop)
        self.log('bulk/loss_recon', loss_bulk_recon)
        self.log('bulk/loss_total', loss_bulk_total)
        
        # Additional metrics for monitoring
        self.log('metrics/mu_mean', mu.mean())
        self.log('metrics/mu_std', mu.std())
        self.log('metrics/logvar_mean', logvar.mean())
        self.log('metrics/S_norm', torch.norm(self.S))
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step - same as training but with val_ prefix for metrics.
        
        This helps monitor overfitting and model performance on held-out data.
        """
        
        # --- Single-Cell Path ---
        sc_x = batch['sc_data']
        sc_y = batch['sc_label']
        
        mu, logvar = self.encoder(sc_x)
        z = self.reparameterize(mu, logvar)
        sc_rec = self.decoder(z)
        
        # SC Losses
        loss_recon = F.mse_loss(sc_rec, sc_x)
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        s_y_prototypes = self.S.T[sc_y]
        loss_proto = F.mse_loss(sc_rec, s_y_prototypes)
        
        loss_sc_total = (loss_recon + 
                        (self.hparams.lambda_kl * loss_kl) + 
                        (self.hparams.lambda_proto * loss_proto))
        
        # --- Bulk Path ---
        b_sim = batch['bulk_data']
        p_true = batch['bulk_prop']
        
        p_pred = self.deconv_net(b_sim)

        # Bulk Losses
        p_true_safe = p_true.clamp(min=1e-10)
        p_true_safe = p_true_safe / p_true_safe.sum(dim=1, keepdim=True)
        loss_prop = F.kl_div(p_pred.log(), p_true_safe, reduction='batchmean')
        
        b_rec = torch.matmul(p_pred, self.S.T)
        loss_bulk_recon = F.mse_loss(b_rec, b_sim)
        
        loss_bulk_total = loss_prop + (self.hparams.lambda_bulk_recon * loss_bulk_recon)
        
        # --- Total Loss ---
        total_loss = loss_sc_total + (self.hparams.lambda_bulk * loss_bulk_total)
        
        # --- Logging with val_ prefix ---
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_sc/loss_recon', loss_recon)
        self.log('val_sc/loss_kl', loss_kl)
        self.log('val_sc/loss_proto', loss_proto)
        self.log('val_bulk/loss_prop', loss_prop)
        self.log('val_bulk/loss_recon', loss_bulk_recon)
        
        # Additional validation metrics
        # Calculate proportion prediction accuracy (MAE)
        prop_mae = torch.mean(torch.abs(p_pred - p_true))
        self.log('val_metrics/prop_mae', prop_mae)
        
        return total_loss
    
    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        
        Uses Adam optimizer with the specified learning rate.
        All parameters (encoder, decoder, deconv_net, and S) are optimized jointly.
        
        Returns:
            optimizer: Configured Adam optimizer
        """
        # self.parameters() automatically includes:
        # 1. self.encoder parameters
        # 2. self.decoder parameters
        # 3. self.deconv_net parameters
        # 4. self.S (because it's nn.Parameter)
        
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate
        )
        
        # Optional: Add learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode='min', factor=0.5, patience=10
        # )
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler,
        #         'monitor': 'val_loss'
        #     }
        # }
        
        return optimizer
    
    def get_signature_matrix(self) -> torch.Tensor:
        """
        Utility method to retrieve the learned signature matrix.
        
        Returns:
            S: The signature matrix (input_dim, n_cell_types) as a numpy array
        """
        return self.S.detach().cpu().numpy()
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode single-cell data to latent space.
        
        Args:
            x: Single-cell expression data
            
        Returns:
            mu, logvar: Parameters of the latent distribution
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to expression space.
        
        Args:
            z: Latent vectors
            
        Returns:
            x_rec: Reconstructed expression
        """
        return self.decoder(z)
    
    def deconvolve(self, bulk_data: torch.Tensor) -> torch.Tensor:
        """
        Predict cell type proportions from bulk data.
        
        Args:
            bulk_data: Bulk expression data
            
        Returns:
            proportions: Predicted cell type proportions
        """
        return self.deconv_net(bulk_data)


# ============================================================================
# UTILITY FUNCTIONS FOR MODEL INSTANTIATION AND TRAINING
# ============================================================================

def create_model(input_dim: int = 2000,
                 latent_dim: int = 32,
                 n_cell_types: int = 8,
                 n_hidden: int = 128,
                 **kwargs) -> HP_VADE:
    """
    Factory function to create an HP-VADE model with default parameters.
    
    Args:
        input_dim: Number of genes
        latent_dim: Latent space dimension
        n_cell_types: Number of cell types
        n_hidden: Hidden layer size
        **kwargs: Additional hyperparameters (lambda values, learning rate)
    
    Returns:
        model: Initialized HP-VADE model
    """
    
    default_params = {
        'lambda_proto': 1.0,
        'lambda_bulk_recon': 0.5,
        'lambda_bulk': 1.0,
        'lambda_kl': 0.1,
        'learning_rate': 1e-3
    }
    
    # Update defaults with any provided kwargs
    default_params.update(kwargs)
    
    model = HP_VADE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        n_cell_types=n_cell_types,
        n_hidden=n_hidden,
        **default_params
    )
    
    return model


if __name__ == "__main__":
    """
    Quick test to ensure the model can be instantiated and run a forward pass.
    """
    
    print("=" * 80)
    print("HP-VADE Model Architecture Test")
    print("=" * 80)
    
    # Create a model
    model = create_model(
        input_dim=2000,
        latent_dim=32,
        n_cell_types=8,
        n_hidden=128
    )
    
    print("\nModel created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test with dummy data
    batch_size = 4
    dummy_batch = {
        'sc_data': torch.randn(batch_size, 2000),
        'sc_label': torch.randint(0, 8, (batch_size,)),
        'bulk_data': torch.randn(batch_size, 2000),
        'bulk_prop': torch.softmax(torch.randn(batch_size, 8), dim=1)
    }
    
    # Test training step
    print("\nTesting training step...")
    loss = model.training_step(dummy_batch, 0)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test inference
    print("\nTesting inference (deconvolution)...")
    model.eval()
    with torch.no_grad():
        bulk_test = torch.randn(2, 2000)
        proportions = model(bulk_test)
        print(f"Predicted proportions shape: {proportions.shape}")
        print(f"Proportions sum to 1: {proportions.sum(dim=1)}")
    
    print("\n✓ All tests passed!")