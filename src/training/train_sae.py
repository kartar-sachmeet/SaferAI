"""
Training script for the Diff-SAE on activation differences.
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Optional, Dict
import argparse

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models import BatchTopKSAE
from src.utils import prepare_training_data


class ActivationDataset(Dataset):
    """Dataset for activation differences."""

    def __init__(self, activations: torch.Tensor):
        """
        Initialize dataset.

        Args:
            activations: Tensor of shape (n_samples, d_model)
        """
        self.activations = activations

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx]


class SAETrainer:
    """Trainer for BatchTopK SAE on activation differences."""

    def __init__(
        self,
        sae: BatchTopKSAE,
        config: Dict,
        device: str = 'cuda'
    ):
        """
        Initialize the SAE trainer.

        Args:
            sae: The SAE model to train
            config: Training configuration dictionary
            device: Device to train on
        """
        self.sae = sae.to(device)
        self.config = config
        self.device = device

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.sae.parameters(),
            lr=config['sae']['learning_rate']
        )

        # Setup learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['sae']['learning_rate'],
            total_steps=config['sae'].get('total_steps', 10000),
            pct_start=0.1  # 10% warmup
        )

        # Logging
        self.use_wandb = config['logging']['use_wandb']
        if self.use_wandb:
            wandb.init(
                project=config['logging']['wandb_project'],
                entity=config['logging']['wandb_entity'],
                config=config
            )

        # Checkpointing
        self.checkpoint_dir = Path(config['sae']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = config['sae']['save_every']

        self.step = 0

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch: Batch of activations

        Returns:
            Dictionary of metrics
        """
        batch = batch.to(self.device)

        # Forward pass
        loss_dict = self.sae.compute_loss(batch)

        # Backward pass
        self.optimizer.zero_grad()
        loss_dict['loss'].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.sae.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        # Renormalize decoder periodically
        if self.step % 100 == 0:
            self.sae.renormalize_decoder()

        self.step += 1

        # Convert to float for logging
        metrics = {k: v.item() for k, v in loss_dict.items()}
        metrics['lr'] = self.scheduler.get_last_lr()[0]

        return metrics

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader for training data
            epoch: Current epoch number
        """
        self.sae.train()

        total_loss = 0.0
        total_mse = 0.0
        total_l0 = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            metrics = self.train_step(batch)

            total_loss += metrics['loss']
            total_mse += metrics['mse_loss']
            total_l0 += metrics['l0']

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'mse': f"{metrics['mse_loss']:.4f}",
                'l0': f"{metrics['l0']:.1f}",
                'lr': f"{metrics['lr']:.2e}"
            })

            # Log to wandb
            if self.use_wandb and batch_idx % self.config['logging']['log_every'] == 0:
                wandb.log({
                    'train/loss': metrics['loss'],
                    'train/mse_loss': metrics['mse_loss'],
                    'train/l0': metrics['l0'],
                    'train/lr': metrics['lr'],
                    'step': self.step
                })

            # Save checkpoint
            if self.step % self.save_every == 0:
                self.save_checkpoint(f'checkpoint_step_{self.step}.pt')

        # Epoch statistics
        n_batches = len(dataloader)
        avg_loss = total_loss / n_batches
        avg_mse = total_mse / n_batches
        avg_l0 = total_l0 / n_batches

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg MSE: {avg_mse:.4f}")
        print(f"  Avg L0: {avg_l0:.1f}")

        return {
            'loss': avg_loss,
            'mse_loss': avg_mse,
            'l0': avg_l0
        }

    def train(
        self,
        train_data: torch.Tensor,
        num_epochs: Optional[int] = None
    ):
        """
        Train the SAE.

        Args:
            train_data: Training data tensor
            num_epochs: Number of epochs (uses config if not specified)
        """
        if num_epochs is None:
            num_epochs = self.config['sae']['num_epochs']

        # Create dataset and dataloader
        dataset = ActivationDataset(train_data)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['sae']['batch_size'],
            shuffle=True,
            num_workers=self.config['compute']['num_workers'],
            pin_memory=True
        )

        # Update total steps for scheduler
        total_steps = len(dataloader) * num_epochs
        self.config['sae']['total_steps'] = total_steps
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config['sae']['learning_rate'],
            total_steps=total_steps,
            pct_start=0.1
        )

        print(f"Training SAE for {num_epochs} epochs ({total_steps} steps)")
        print(f"Training samples: {len(dataset)}")
        print(f"Batch size: {self.config['sae']['batch_size']}")
        print(f"Steps per epoch: {len(dataloader)}")

        # Training loop
        for epoch in range(1, num_epochs + 1):
            epoch_metrics = self.train_epoch(dataloader, epoch)

            # Log epoch metrics
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'epoch/loss': epoch_metrics['loss'],
                    'epoch/mse_loss': epoch_metrics['mse_loss'],
                    'epoch/l0': epoch_metrics['l0']
                })

            # Save checkpoint at end of epoch
            self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

        # Save final model
        self.save_checkpoint('final_model.pt')
        print("\nTraining complete!")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'step': self.step,
            'model_state_dict': self.sae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.sae.get_config()
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.sae.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']

        print(f"Checkpoint loaded from {checkpoint_path} (step {self.step})")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Diff-SAE')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--activations',
        type=str,
        required=True,
        help='Path to saved diff activations'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load activations
    print(f"Loading activations from {args.activations}...")
    from src.utils import DiffActivationCollector
    activations = DiffActivationCollector.load_activations(args.activations)

    # Prepare training data
    train_data = prepare_training_data(
        activations['diff'],
        flatten_sequence=True
    )
    print(f"Training data shape: {train_data.shape}")

    # Create SAE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sae = BatchTopKSAE(
        d_model=config['sae']['d_model'],
        n_latents=config['sae']['n_latents'],
        k=config['sae']['k']
    )

    # Create trainer
    trainer = SAETrainer(sae, config, device=device)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(train_data)


if __name__ == "__main__":
    main()
