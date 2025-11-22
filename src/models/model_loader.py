"""
Model loading utilities for Gemma 2 base and instruction-tuned models.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional
import yaml
from pathlib import Path


class GemmaModelPair:
    """Manages loading and access to both Gemma 2 base and IT models."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the model pair.

        Args:
            config_path: Path to configuration YAML file
        """
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default config
            self.config = {
                'models': {
                    'base': {
                        'name': 'google/gemma-2-2b',
                        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                        'dtype': 'bfloat16'
                    },
                    'instruct': {
                        'name': 'google/gemma-2-2b-it',
                        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                        'dtype': 'bfloat16'
                    },
                    'target_layer': 13
                }
            }

        self.base_model = None
        self.instruct_model = None
        self.tokenizer = None
        self.device = self.config['models']['base']['device']
        self.dtype = self._get_dtype(self.config['models']['base']['dtype'])

    def _get_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert dtype string to torch dtype."""
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.float32)

    def load_models(self, load_base: bool = True, load_instruct: bool = True):
        """
        Load the Gemma 2 models.

        Args:
            load_base: Whether to load the base model
            load_instruct: Whether to load the instruction-tuned model
        """
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['models']['base']['name'],
            trust_remote_code=True
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if load_base:
            print(f"Loading base model: {self.config['models']['base']['name']}...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.config['models']['base']['name'],
                torch_dtype=self.dtype,
                device_map=self.device,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.base_model.eval()
            print("Base model loaded successfully.")

        if load_instruct:
            print(f"Loading instruction-tuned model: {self.config['models']['instruct']['name']}...")
            self.instruct_model = AutoModelForCausalLM.from_pretrained(
                self.config['models']['instruct']['name'],
                torch_dtype=self.dtype,
                device_map=self.device,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.instruct_model.eval()
            print("Instruction-tuned model loaded successfully.")

    def get_model_config(self):
        """Get the model configuration (hidden size, num layers, etc.)."""
        if self.base_model is None:
            raise ValueError("Models not loaded. Call load_models() first.")

        config = self.base_model.config
        return {
            'd_model': config.hidden_size,
            'n_layers': config.num_hidden_layers,
            'n_heads': config.num_attention_heads,
            'vocab_size': config.vocab_size,
            'target_layer': self.config['models']['target_layer']
        }

    def tokenize(self, texts, **kwargs):
        """Tokenize input texts."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_models() first.")

        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            **kwargs
        )

    def to(self, device: str):
        """Move models to specified device."""
        self.device = device
        if self.base_model is not None:
            self.base_model = self.base_model.to(device)
        if self.instruct_model is not None:
            self.instruct_model = self.instruct_model.to(device)
        return self

    def __repr__(self):
        base_status = "loaded" if self.base_model is not None else "not loaded"
        instruct_status = "loaded" if self.instruct_model is not None else "not loaded"
        return f"GemmaModelPair(base={base_status}, instruct={instruct_status}, device={self.device})"


def load_gemma_models(config_path: str = "configs/config.yaml") -> GemmaModelPair:
    """
    Convenience function to load Gemma model pair.

    Args:
        config_path: Path to configuration file

    Returns:
        Initialized GemmaModelPair with models loaded
    """
    model_pair = GemmaModelPair(config_path)
    model_pair.load_models()
    return model_pair


if __name__ == "__main__":
    # Test model loading
    print("Testing model loading...")
    model_pair = load_gemma_models()
    print(model_pair)
    print("\nModel config:")
    print(model_pair.get_model_config())
