"""
Sparse Autoencoder (SAE) implementation with BatchTopK architecture.
Based on the methodology from the LessWrong diff-SAE paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import einops


class BatchTopKSAE(nn.Module):
    """
    Batch TopK Sparse Autoencoder.

    This SAE uses a TopK activation function to enforce sparsity,
    keeping only the top k activations per sample.
    """

    def __init__(
        self,
        d_model: int,
        n_latents: int,
        k: int = 64,
        init_scale: float = 0.1,
    ):
        """
        Initialize the BatchTopK SAE.

        Args:
            d_model: Dimension of the input activations
            n_latents: Number of latent features (e.g., 18432)
            k: Number of top activations to keep per sample
            init_scale: Scale for weight initialization
        """
        super().__init__()

        self.d_model = d_model
        self.n_latents = n_latents
        self.k = k

        # Encoder: maps from d_model to n_latents
        self.encoder = nn.Linear(d_model, n_latents, bias=True)

        # Decoder: maps from n_latents back to d_model
        self.decoder = nn.Linear(n_latents, d_model, bias=True)

        # Initialize weights
        self._init_weights(init_scale)

        # Normalize decoder columns to unit norm (standard practice in SAEs)
        self._normalize_decoder()

    def _init_weights(self, scale: float):
        """Initialize weights with small random values."""
        nn.init.normal_(self.encoder.weight, mean=0.0, std=scale)
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=scale)
        nn.init.zeros_(self.decoder.bias)

    def _normalize_decoder(self):
        """Normalize decoder weight columns to unit norm."""
        with torch.no_grad():
            # Decoder weight shape: (d_model, n_latents)
            # Normalize each column (each latent's decoder vector)
            norms = torch.norm(self.decoder.weight, dim=0, keepdim=True)
            self.decoder.weight.div_(norms + 1e-8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space with TopK sparsity.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Sparse latent activations of shape (..., n_latents)
        """
        # Linear projection
        pre_activations = self.encoder(x)

        # Apply ReLU
        activations = F.relu(pre_activations)

        # Apply TopK sparsity
        # Keep only top k values per sample, set others to zero
        if self.k < self.n_latents:
            # Get topk values and indices
            topk_values, topk_indices = torch.topk(
                activations,
                k=self.k,
                dim=-1,
                sorted=False
            )

            # Create sparse tensor with only topk values
            sparse_activations = torch.zeros_like(activations)
            sparse_activations.scatter_(-1, topk_indices, topk_values)

            return sparse_activations
        else:
            return activations

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent activations back to input space.

        Args:
            latents: Latent activations of shape (..., n_latents)

        Returns:
            Reconstructed activations of shape (..., d_model)
        """
        return self.decoder(latents)

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the SAE.

        Args:
            x: Input tensor of shape (..., d_model)
            return_components: Whether to return intermediate components

        Returns:
            Dictionary with reconstruction and optional components
        """
        # Encode
        latents = self.encode(x)

        # Decode
        reconstruction = self.decode(latents)

        result = {
            'reconstruction': reconstruction,
            'latents': latents,
        }

        if return_components:
            # Compute sparsity metrics
            l0 = (latents != 0).float().sum(dim=-1).mean()
            result['l0'] = l0
            result['active_latents'] = latents != 0

        return result

    def compute_loss(
        self,
        x: torch.Tensor,
        reduction: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction loss.

        Args:
            x: Input tensor of shape (..., d_model)
            reduction: How to reduce the loss ('mean' or 'sum')

        Returns:
            Dictionary with loss components
        """
        # Forward pass
        output = self.forward(x, return_components=True)
        reconstruction = output['reconstruction']

        # MSE reconstruction loss
        mse_loss = F.mse_loss(reconstruction, x, reduction=reduction)

        # L0 sparsity (number of active features)
        l0 = output['l0']

        return {
            'loss': mse_loss,  # Total loss is just reconstruction for TopK
            'mse_loss': mse_loss,
            'l0': l0,
        }

    def renormalize_decoder(self):
        """Renormalize decoder weights (call periodically during training)."""
        self._normalize_decoder()

    @torch.no_grad()
    def get_feature_magnitudes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the magnitude of each feature's activation.

        Args:
            x: Input tensor

        Returns:
            Feature magnitudes of shape (n_latents,)
        """
        latents = self.encode(x)
        return latents.abs().sum(dim=0)

    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            'd_model': self.d_model,
            'n_latents': self.n_latents,
            'k': self.k,
            'architecture': 'BatchTopK'
        }

    @classmethod
    def from_config(cls, config: Dict) -> 'BatchTopKSAE':
        """Create SAE from configuration dictionary."""
        return cls(
            d_model=config['d_model'],
            n_latents=config['n_latents'],
            k=config['k']
        )


class SAETrainingBuffer:
    """Buffer for storing and sampling activation data for SAE training."""

    def __init__(self, buffer_size: int = 100000, device: str = 'cuda'):
        """
        Initialize the training buffer.

        Args:
            buffer_size: Maximum number of samples to store
            device: Device to store buffer on
        """
        self.buffer_size = buffer_size
        self.device = device
        self.buffer = []
        self.current_size = 0

    def add(self, activations: torch.Tensor):
        """Add activations to the buffer."""
        activations = activations.to(self.device)

        if self.current_size < self.buffer_size:
            self.buffer.append(activations)
            self.current_size += activations.shape[0]
        else:
            # Buffer is full, replace random samples
            if len(self.buffer) > 0:
                self.buffer = [torch.cat(self.buffer, dim=0)]
                # Keep only buffer_size samples
                if self.buffer[0].shape[0] > self.buffer_size:
                    indices = torch.randperm(self.buffer[0].shape[0])[:self.buffer_size]
                    self.buffer[0] = self.buffer[0][indices]

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample a batch from the buffer."""
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        # Concatenate all buffered data
        all_data = torch.cat(self.buffer, dim=0)

        # Sample random indices
        indices = torch.randint(0, all_data.shape[0], (batch_size,))
        return all_data[indices]

    def __len__(self):
        """Get current buffer size."""
        return self.current_size


if __name__ == "__main__":
    # Test SAE
    print("Testing BatchTopK SAE...")
    d_model = 2304  # Gemma 2 2B hidden dim
    n_latents = 18432
    k = 64

    sae = BatchTopKSAE(d_model=d_model, n_latents=n_latents, k=k)

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, d_model)

    output = sae(x, return_components=True)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {output['reconstruction'].shape}")
    print(f"Latents shape: {output['latents'].shape}")
    print(f"L0 (active features): {output['l0']:.2f}")

    # Test loss computation
    loss_dict = sae.compute_loss(x)
    print(f"\nLoss: {loss_dict['loss']:.4f}")
    print(f"MSE Loss: {loss_dict['mse_loss']:.4f}")

    print("\nBatchTopK SAE test completed successfully!")
