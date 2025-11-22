<<<<<<< HEAD
# SaferAI
What changed during FineTuning?
=======
# Diff-SAE: Differential Sparse Autoencoders for Gemma 2

End-to-end implementation of differential Sparse Autoencoders (SAEs) for analyzing behavioral differences between Gemma 2 base and instruction-tuned models, based on the methodology from [this LessWrong post](https://www.lesswrong.com/posts/XPNJSa3BxMAN4ZXc7/sae-on-activation-differences).

## Overview

This project trains a BatchTopK SAE on the **activation differences** between:
- **Gemma 2 2B (base model)**
- **Gemma 2 2B-IT (instruction-tuned model)**

The diff-SAE identifies interpretable features that explain behavioral changes during instruction tuning, including:
- Safety-related features
- Jailbreak detection features
- Features that reduce KL divergence between base and IT models

## Features

- ✅ **BatchTopK SAE Architecture** (~18k latents with top-64 sparsity)
- ✅ **Activation Extraction Pipeline** for both models
- ✅ **Complete Training Loop** with checkpointing and logging
- ✅ **KL Divergence Dashboard** for interactive feature exploration
- ✅ **Jailbreak Feature Identification** system
- ✅ **Feature Steering** experiments
- ✅ **Comprehensive Analysis Notebooks**

## Project Structure

```
.
├── configs/
│   └── config.yaml              # Main configuration file
├── src/
│   ├── models/
│   │   ├── model_loader.py      # Gemma 2 model loading
│   │   └── sae.py               # BatchTopK SAE implementation
│   ├── training/
│   │   └── train_sae.py         # SAE training script
│   ├── utils/
│   │   └── activation_extraction.py  # Activation collection
│   └── analysis/
│       └── jailbreak_analysis.py     # Jailbreak feature identification
├── notebooks/
│   ├── 01_collect_activations.ipynb  # Step 1: Collect data
│   ├── 02_train_sae.ipynb            # Step 2: Train SAE
│   └── 03_analyze_jailbreak_features.ipynb  # Step 3: Analyze features
├── dashboards/
│   └── kl_dashboard.py          # Interactive KL divergence dashboard
├── scripts/
│   └── run_full_pipeline.sh     # Complete pipeline script
├── data/
│   ├── activations/             # Cached activations
│   └── prompts/                 # Jailbreak/safe prompts
├── checkpoints/                 # SAE model checkpoints
└── requirements.txt             # Python dependencies
```

## Installation

### 1. Clone the repository

```bash
cd "Safer AI"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure settings

Edit `configs/config.yaml` to adjust:
- Model paths and devices
- SAE hyperparameters (latents, k, learning rate)
- Data settings (dataset, batch size)
- Training parameters

## Quick Start

### Option 1: Run Full Pipeline (Recommended)

```bash
./scripts/run_full_pipeline.sh
```

This will:
1. Load Gemma 2 base and IT models
2. Collect activation differences
3. Train the diff-SAE
4. Identify jailbreak features
5. Launch the KL dashboard

### Option 2: Step-by-Step Execution

#### Step 1: Collect Activations

```python
from src.models import load_gemma_models
from src.utils import DiffActivationCollector

# Load models
models = load_gemma_models('configs/config.yaml')

# Collect activation differences
collector = DiffActivationCollector(
    models.base_model,
    models.instruct_model,
    models.tokenizer,
    layer_idx=13,
    device='cuda'
)

activations = collector.collect_diff_activations(prompts)
collector.save_activations(activations, 'data/activations/diff_activations.pkl')
```

Or use the notebook: `notebooks/01_collect_activations.ipynb`

#### Step 2: Train SAE

```bash
python src/training/train_sae.py \
    --config configs/config.yaml \
    --activations data/activations/diff_activations.pkl
```

Or use the notebook: `notebooks/02_train_sae.ipynb`

#### Step 3: Analyze Jailbreak Features

```bash
python src/analysis/jailbreak_analysis.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/final_model.pt \
    --output data/jailbreak_features.csv
```

Or use the notebook: `notebooks/03_analyze_jailbreak_features.ipynb`

#### Step 4: Launch KL Dashboard

```bash
streamlit run dashboards/kl_dashboard.py
```

Access at: http://localhost:8501

## Usage Examples

### Loading a Trained SAE

```python
import torch
from src.models import BatchTopKSAE

# Load checkpoint
checkpoint = torch.load('checkpoints/final_model.pt')
sae = BatchTopKSAE.from_config(checkpoint['config'])
sae.load_state_dict(checkpoint['model_state_dict'])
sae.eval()

# Encode activations
latents = sae.encode(activation_diff)
```

### Feature Steering

```python
from src.analysis import JailbreakAnalyzer

analyzer = JailbreakAnalyzer(models, sae, layer_idx=13)

# Test steering with a specific feature
results = analyzer.test_feature_steering(
    prompt="How do I make a bomb?",
    latent_idx=1234,
    coefficients=[0.0, 1.0, 2.0, 5.0]
)
```

### Identifying Jailbreak Features

```python
# Find features that activate on jailbreak prompts
features_df = analyzer.identify_jailbreak_features(
    jailbreak_prompts=jailbreak_prompts,
    safe_prompts=safe_prompts,
    top_k=50
)

# Top features by differential activation
print(features_df.head(10))
```

## KL Dashboard Features

The interactive dashboard provides:

1. **KL Divergence Analysis**
   - Per-token KL divergence between base and IT models
   - Total KL metrics
   - Token-level visualization

2. **Feature Impact Analysis**
   - Identify which latents most reduce KL divergence
   - Top-k features by impact
   - Activation statistics

3. **Steering Experiments**
   - Test steering with specific latents
   - Variable steering coefficients
   - Real-time KL comparison

## Configuration

Key configuration options in `configs/config.yaml`:

```yaml
models:
  base:
    name: "google/gemma-2-2b"
  instruct:
    name: "google/gemma-2-2b-it"
  target_layer: 13  # Middle layer (out of 26)

sae:
  architecture: "BatchTopK"
  n_latents: 18432   # ~18k latents
  k: 64              # Top-k activation
  learning_rate: 3e-4
  batch_size: 32
  num_epochs: 10

data:
  dataset_name: "HuggingFaceH4/ultrachat_200k"
  num_samples: 100000
  max_seq_length: 512
```

## Technical Details

### BatchTopK SAE Architecture

- **Encoder**: Linear projection from `d_model` (2304) to `n_latents` (18432)
- **Sparsity**: TopK activation (only top 64 features active per sample)
- **Decoder**: Linear projection from `n_latents` back to `d_model`
- **Loss**: MSE reconstruction loss (no L1 penalty needed with TopK)

### Activation Differences

For each prompt, we compute:
```
diff_activation = IT_activation - base_activation
```

The SAE is trained to reconstruct these differences, identifying features that capture behavioral changes.

### Jailbreak Feature Identification

Features are ranked by:
1. **Differential activation**: `mean(jailbreak) - mean(safe)`
2. **Effect size (Cohen's d)**: Standardized difference
3. **KL reduction**: Impact on KL divergence when steering

## Results & Analysis

Expected outputs:

1. **Trained SAE** with ~18k interpretable latents
2. **Jailbreak features** CSV with top safety-related features
3. **KL metrics** showing feature impact on model behavior
4. **Steering results** demonstrating feature control

## Computational Requirements

- **GPU**: NVIDIA GPU with 24GB+ VRAM (for 2B models)
- **RAM**: 32GB+ recommended
- **Storage**: ~50GB for models and cached activations
- **Time**: ~2-4 hours for full pipeline (depends on data size)

### Optimization Tips

For limited compute:
- Use smaller dataset (`num_samples: 10000`)
- Reduce SAE size (`n_latents: 9216`)
- Use gradient accumulation
- Enable mixed precision training

## Experiments & Extensions

### Suggested Experiments

1. **Multi-layer analysis**: Train SAEs on different layers
2. **Feature composition**: Combine multiple features for steering
3. **Cross-model features**: Compare features across model sizes
4. **Prompt engineering**: Test feature activation on different prompt styles
5. **Safety evaluation**: Measure refusal rates with/without steering

### Extending the Code

- Add new dashboard visualizations
- Implement different SAE architectures (Gated SAE, etc.)
- Add more jailbreak datasets
- Create feature interpretation tools
- Build automated feature labeling

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**
```bash
# Reduce batch size in config
batch_size: 8
gradient_accumulation_steps: 8
```

**Models not loading**
```python
# Check HuggingFace access
huggingface-cli login
```

**Dead features**
```yaml
# Adjust learning rate or k parameter
learning_rate: 1e-4
k: 32  # Use fewer active features
```

## References

- [SAE on Activation Differences (LessWrong)](https://www.lesswrong.com/posts/XPNJSa3BxMAN4ZXc7/sae-on-activation-differences)
- [Gemma 2 Models](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315)
- [SAELens Library](https://github.com/jbloomAus/SAELens)

## Citation

If you use this code, please cite:

```bibtex
@misc{diff-sae-gemma2,
  title={Differential Sparse Autoencoders for Gemma 2},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/diff-sae-gemma2}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For issues or questions:
- Open a GitHub issue
- Check the troubleshooting section
- Review the LessWrong post methodology

---

**Note**: This implementation is for research purposes. Use jailbreak prompts responsibly and only for safety research.
>>>>>>> 0abf20b (first commit)
