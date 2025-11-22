"""
Utilities for extracting and computing activation differences between models.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from pathlib import Path
import pickle


class ActivationExtractor:
    """Extract activations from specific layers of transformer models."""

    def __init__(self, model, layer_idx: int):
        """
        Initialize the activation extractor.

        Args:
            model: The transformer model
            layer_idx: Index of the layer to extract activations from
        """
        self.model = model
        self.layer_idx = layer_idx
        self.activations = {}
        self.hooks = []

    def _get_activation_hook(self, name: str):
        """Create a hook function to capture activations."""
        def hook(module, input, output):
            # For transformer blocks, output is typically a tuple (hidden_states, ...)
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook

    def register_hooks(self):
        """Register forward hooks to capture activations."""
        # Access the specific layer (assuming Gemma structure)
        target_layer = self.model.model.layers[self.layer_idx]

        # Register hook on the layer output
        hook = target_layer.register_forward_hook(
            self._get_activation_hook(f'layer_{self.layer_idx}')
        )
        self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def extract_activations(self, input_ids, attention_mask=None):
        """
        Extract activations for given inputs.

        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask for padding

        Returns:
            Activations tensor of shape (batch_size, seq_len, d_model)
        """
        self.activations = {}

        with torch.no_grad():
            _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False
            )

        # Return the activations from the target layer
        return self.activations[f'layer_{self.layer_idx}']

    def __del__(self):
        """Clean up hooks when object is deleted."""
        self.remove_hooks()


class DiffActivationCollector:
    """Collect activation differences between base and instruction-tuned models."""

    def __init__(
        self,
        base_model,
        instruct_model,
        tokenizer,
        layer_idx: int,
        device: str = 'cuda'
    ):
        """
        Initialize the diff activation collector.

        Args:
            base_model: Base model
            instruct_model: Instruction-tuned model
            tokenizer: Tokenizer for both models
            layer_idx: Layer index to extract activations from
            device: Device to run on
        """
        self.base_model = base_model
        self.instruct_model = instruct_model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = device

        # Initialize extractors
        self.base_extractor = ActivationExtractor(base_model, layer_idx)
        self.instruct_extractor = ActivationExtractor(instruct_model, layer_idx)

    def collect_diff_activations(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 512,
        return_individual: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Collect activation differences for a list of texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            return_individual: Whether to return individual activations as well

        Returns:
            Dictionary with 'diff' activations and optionally 'base' and 'instruct'
        """
        # Register hooks
        self.base_extractor.register_hooks()
        self.instruct_extractor.register_hooks()

        all_diff_acts = []
        all_base_acts = []
        all_instruct_acts = []

        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)

            # Extract activations from both models
            base_acts = self.base_extractor.extract_activations(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            instruct_acts = self.instruct_extractor.extract_activations(
                inputs['input_ids'],
                inputs['attention_mask']
            )

            # Compute difference (IT - base)
            diff_acts = instruct_acts - base_acts

            # Store activations
            all_diff_acts.append(diff_acts.cpu())
            if return_individual:
                all_base_acts.append(base_acts.cpu())
                all_instruct_acts.append(instruct_acts.cpu())

        # Concatenate all batches
        result = {
            'diff': torch.cat(all_diff_acts, dim=0)
        }

        if return_individual:
            result['base'] = torch.cat(all_base_acts, dim=0)
            result['instruct'] = torch.cat(all_instruct_acts, dim=0)

        # Clean up hooks
        self.base_extractor.remove_hooks()
        self.instruct_extractor.remove_hooks()

        return result

    def save_activations(
        self,
        activations: Dict[str, torch.Tensor],
        save_path: str
    ):
        """Save activations to disk."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to numpy for efficient storage
        activations_np = {
            k: v.numpy() for k, v in activations.items()
        }

        with open(save_path, 'wb') as f:
            pickle.dump(activations_np, f)

        print(f"Activations saved to {save_path}")

    @staticmethod
    def load_activations(load_path: str) -> Dict[str, torch.Tensor]:
        """Load activations from disk."""
        with open(load_path, 'rb') as f:
            activations_np = pickle.load(f)

        # Convert back to torch
        activations = {
            k: torch.from_numpy(v) for k, v in activations_np.items()
        }

        print(f"Activations loaded from {load_path}")
        return activations


def prepare_training_data(
    diff_activations: torch.Tensor,
    flatten_sequence: bool = True
) -> torch.Tensor:
    """
    Prepare activation differences for SAE training.

    Args:
        diff_activations: Activation differences of shape (batch, seq_len, d_model)
        flatten_sequence: Whether to flatten batch and sequence dimensions

    Returns:
        Prepared data for SAE training
    """
    if flatten_sequence:
        # Flatten to (batch * seq_len, d_model)
        batch_size, seq_len, d_model = diff_activations.shape
        return diff_activations.reshape(-1, d_model)
    return diff_activations


if __name__ == "__main__":
    # Test activation extraction
    print("Activation extraction utilities loaded successfully.")
