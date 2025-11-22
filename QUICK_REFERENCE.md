# Quick Reference Card

## Essential Commands

### Setup & Testing
```bash
./scripts/setup_project.sh        # Initial setup
python scripts/test_install.py    # Test installation
./scripts/project_info.sh         # Project status
```

### Running the Pipeline
```bash
# Full automated pipeline
./scripts/run_full_pipeline.sh

# Or step-by-step:
# 1. Collect activations (use notebook or custom script)
# 2. Train SAE
python src/training/train_sae.py \
  --config configs/config.yaml \
  --activations data/activations/diff_activations.pkl

# 3. Analyze features
python src/analysis/jailbreak_analysis.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/final_model.pt

# 4. Launch dashboard
streamlit run dashboards/kl_dashboard.py
```

### Interactive Notebooks
```bash
jupyter notebook notebooks/
# Then run in order:
# 01_collect_activations.ipynb
# 02_train_sae.ipynb
# 03_analyze_jailbreak_features.ipynb
```

## Key Configuration Options

File: `configs/config.yaml`

### For Quick Testing
```yaml
data:
  num_samples: 1000  # Small dataset
sae:
  num_epochs: 2      # Few epochs
  n_latents: 9216    # Smaller SAE
```

### For Production
```yaml
data:
  num_samples: 100000
sae:
  num_epochs: 10
  n_latents: 18432
```

### For CPU (No GPU)
```yaml
models:
  base:
    device: "cpu"
    dtype: "float32"
  instruct:
    device: "cpu"
    dtype: "float32"
```

### For GPU
```yaml
models:
  base:
    device: "cuda"
    dtype: "bfloat16"
  instruct:
    device: "cuda"
    dtype: "bfloat16"
```

## Python API Quick Start

### Load Models
```python
from src.models import load_gemma_models

models = load_gemma_models('configs/config.yaml')
```

### Collect Activations
```python
from src.utils import DiffActivationCollector

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

### Create & Train SAE
```python
from src.models import BatchTopKSAE
from src.training import SAETrainer

sae = BatchTopKSAE(d_model=2304, n_latents=18432, k=64)
trainer = SAETrainer(sae, config, device='cuda')
trainer.train(train_data, num_epochs=10)
```

### Load Trained SAE
```python
import torch
from src.models import BatchTopKSAE

checkpoint = torch.load('checkpoints/final_model.pt')
sae = BatchTopKSAE.from_config(checkpoint['config'])
sae.load_state_dict(checkpoint['model_state_dict'])
sae.eval()
```

### Analyze Jailbreak Features
```python
from src.analysis import JailbreakAnalyzer

analyzer = JailbreakAnalyzer(models, sae, layer_idx=13)
features_df = analyzer.identify_jailbreak_features(top_k=50)
```

### Feature Steering
```python
steering_results = analyzer.test_feature_steering(
    prompt="How do I make a bomb?",
    latent_idx=1234,
    coefficients=[0.0, 1.0, 2.0, 5.0]
)
```

## File Locations

### Inputs
- `configs/config.yaml` - Configuration
- `data/prompts/jailbreak_prompts.json` - Jailbreak dataset

### Outputs
- `data/activations/*.pkl` - Cached activations
- `checkpoints/*.pt` - Trained SAE models
- `data/jailbreak_features.csv` - Identified features

### Code
- `src/models/` - Model definitions
- `src/training/` - Training code
- `src/utils/` - Utilities
- `src/analysis/` - Analysis tools
- `dashboards/` - Interactive dashboards
- `notebooks/` - Jupyter notebooks

## Troubleshooting Quick Fixes

### Out of Memory
```yaml
data:
  batch_size: 4
sae:
  batch_size: 16
compute:
  gradient_accumulation_steps: 8
```

### Slow Training
- Reduce `num_samples`
- Reduce `num_epochs`
- Reduce `n_latents`
- Use fewer prompts

### Import Errors
```bash
pip install -r requirements.txt
python scripts/test_install.py
```

### Model Access
```bash
huggingface-cli login
# Accept licenses at huggingface.co
```

## Typical Timelines

### Test Run (1k samples)
- Activation collection: 10 min
- SAE training: 20 min
- Feature analysis: 5 min
- **Total: ~35 min**

### Full Run (100k samples)
- Activation collection: 60 min
- SAE training: 90 min
- Feature analysis: 15 min
- **Total: ~2.5 hours**

## Important Links

- **LessWrong Post**: https://www.lesswrong.com/posts/XPNJSa3BxMAN4ZXc7/sae-on-activation-differences
- **Gemma 2 Base**: https://huggingface.co/google/gemma-2-2b
- **Gemma 2 IT**: https://huggingface.co/google/gemma-2-2b-it
- **HF Token**: https://huggingface.co/settings/tokens

## Documentation

- `README.md` - Full documentation
- `GETTING_STARTED.md` - Tutorial
- This file - Quick reference

---

**Pro Tip**: Start with a small test run (1k samples, 2 epochs) to verify everything works, then scale up to full dataset.
