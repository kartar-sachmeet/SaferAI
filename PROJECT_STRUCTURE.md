# Project Structure

Complete directory and file organization for the Diff-SAE project.

```
Safer AI/
│
├── README.md                    # Main documentation (start here!)
├── GETTING_STARTED.md           # Step-by-step tutorial
├── QUICK_REFERENCE.md           # Command cheat sheet
├── PROJECT_STRUCTURE.md         # This file
├── requirements.txt             # Python dependencies
│
├── configs/
│   └── config.yaml              # Main configuration file
│                                # - Model settings
│                                # - SAE hyperparameters
│                                # - Training parameters
│                                # - Data settings
│
├── src/                         # Source code
│   ├── __init__.py
│   │
│   ├── models/                  # Model definitions
│   │   ├── __init__.py
│   │   ├── model_loader.py      # Gemma 2 model loading
│   │   │                        # - GemmaModelPair class
│   │   │                        # - Dual model management
│   │   │                        # - Tokenizer handling
│   │   └── sae.py               # BatchTopK SAE architecture
│   │                            # - Encoder/decoder
│   │                            # - TopK sparsity
│   │                            # - Training utilities
│   │
│   ├── training/                # Training code
│   │   ├── __init__.py
│   │   └── train_sae.py         # SAE training pipeline
│   │                            # - SAETrainer class
│   │                            # - Training loop
│   │                            # - Checkpointing
│   │                            # - WandB logging
│   │
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   └── activation_extraction.py
│   │                            # - ActivationExtractor
│   │                            # - DiffActivationCollector
│   │                            # - Activation caching
│   │
│   └── analysis/                # Analysis tools
│       ├── __init__.py
│       └── jailbreak_analysis.py
│                                # - JailbreakAnalyzer
│                                # - Feature identification
│                                # - Statistical analysis
│                                # - Steering experiments
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_collect_activations.ipynb
│   │                            # - Load models
│   │                            # - Collect activations
│   │                            # - Save for training
│   │
│   ├── 02_train_sae.ipynb
│   │                            # - Load activations
│   │                            # - Train SAE
│   │                            # - Evaluate results
│   │
│   └── 03_analyze_jailbreak_features.ipynb
│                                # - Identify features
│                                # - Test steering
│                                # - Visualize results
│
├── dashboards/                  # Interactive dashboards
│   └── kl_dashboard.py          # Streamlit KL divergence dashboard
│                                # - Text input
│                                # - KL visualization
│                                # - Feature impact analysis
│                                # - Steering experiments
│
├── scripts/                     # Utility scripts
│   ├── setup_project.sh         # Environment setup
│   ├── run_full_pipeline.sh     # Full automation
│   ├── project_info.sh          # Project statistics
│   └── test_install.py          # Installation verification
│
├── data/                        # Data storage
│   ├── activations/             # Cached activations
│   │   └── *.pkl                # Activation files
│   │
│   └── prompts/                 # Prompt datasets
│       ├── jailbreak_prompts.json
│       └── *.json               # Custom prompts
│
└── checkpoints/                 # Model checkpoints
    ├── checkpoint_step_*.pt     # Training checkpoints
    ├── checkpoint_epoch_*.pt    # Epoch checkpoints
    └── final_model.pt           # Final trained model

```

## File Descriptions

### Documentation Files

| File | Purpose | When to Read |
|------|---------|-------------|
| `README.md` | Complete project documentation | First thing to read |
| `GETTING_STARTED.md` | Step-by-step tutorial | Before first run |
| `QUICK_REFERENCE.md` | Command cheat sheet | Quick lookup |
| `PROJECT_STRUCTURE.md` | This file | Understanding organization |

### Configuration

| File | Purpose | When to Edit |
|------|---------|-------------|
| `configs/config.yaml` | All settings and hyperparameters | Before each run |
| `requirements.txt` | Python dependencies | When adding packages |

### Core Source Code

#### src/models/

| File | Classes | Purpose |
|------|---------|---------|
| `model_loader.py` | `GemmaModelPair` | Load and manage Gemma 2 models |
| `sae.py` | `BatchTopKSAE`, `SAETrainingBuffer` | SAE architecture |

**Key Functions:**
- `load_gemma_models()` - Load both models
- `BatchTopKSAE.encode()` - Encode to latent space
- `BatchTopKSAE.decode()` - Decode back to activations

#### src/training/

| File | Classes | Purpose |
|------|---------|---------|
| `train_sae.py` | `SAETrainer`, `ActivationDataset` | Train SAE on diff activations |

**Key Functions:**
- `SAETrainer.train()` - Main training loop
- `SAETrainer.save_checkpoint()` - Save model

#### src/utils/

| File | Classes | Purpose |
|------|---------|---------|
| `activation_extraction.py` | `ActivationExtractor`, `DiffActivationCollector` | Extract and compute activation diffs |

**Key Functions:**
- `DiffActivationCollector.collect_diff_activations()` - Main collection function
- `save_activations()` - Cache to disk
- `load_activations()` - Load from disk

#### src/analysis/

| File | Classes | Purpose |
|------|---------|---------|
| `jailbreak_analysis.py` | `JailbreakAnalyzer` | Identify and analyze safety features |

**Key Functions:**
- `identify_jailbreak_features()` - Find safety-related features
- `test_feature_steering()` - Test steering effects
- `find_top_activating_examples()` - Find top examples per feature

### Notebooks

| Notebook | Purpose | Estimated Time |
|----------|---------|---------------|
| `01_collect_activations.ipynb` | Data collection | 30-60 min |
| `02_train_sae.ipynb` | SAE training | 1-2 hours |
| `03_analyze_jailbreak_features.ipynb` | Feature analysis | 15-30 min |

### Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup_project.sh` | Environment setup | Run once at start |
| `run_full_pipeline.sh` | Automated pipeline | Run for full automation |
| `project_info.sh` | Display stats | Anytime for info |
| `test_install.py` | Verify installation | After setup |

### Data Directories

```
data/
├── activations/          # Cached activation files (~10-50 GB)
│   └── diff_activations.pkl
│
└── prompts/             # Prompt datasets (~1 MB)
    └── jailbreak_prompts.json
```

### Checkpoint Files

```
checkpoints/
├── checkpoint_step_1000.pt    # Intermediate checkpoints
├── checkpoint_step_2000.pt
├── checkpoint_epoch_1.pt      # End of epoch checkpoints
├── checkpoint_epoch_2.pt
└── final_model.pt             # Final trained model
```

**Checkpoint Contents:**
- Model state dict (weights)
- Optimizer state
- Scheduler state
- Training step counter
- Model configuration

## Workflow Paths

### Path 1: Automated (Fastest)
```
./scripts/run_full_pipeline.sh
↓
Activations → Training → Analysis → Dashboard
```

### Path 2: Interactive (Learning)
```
notebooks/01_collect_activations.ipynb
↓
notebooks/02_train_sae.ipynb
↓
notebooks/03_analyze_jailbreak_features.ipynb
↓
streamlit run dashboards/kl_dashboard.py
```

### Path 3: Manual (Maximum Control)
```
1. Custom activation collection
2. python src/training/train_sae.py
3. python src/analysis/jailbreak_analysis.py
4. streamlit run dashboards/kl_dashboard.py
```

## Import Paths

### From notebooks:
```python
import sys
sys.path.append('..')

from src.models import load_gemma_models, BatchTopKSAE
from src.utils import DiffActivationCollector
from src.training import SAETrainer
from src.analysis import JailbreakAnalyzer
```

### From scripts:
```python
import sys
sys.path.append('.')

from src.models import load_gemma_models, BatchTopKSAE
# etc.
```

## Data Flow

```
Input Dataset (HuggingFace)
  ↓
Gemma 2 Base + IT Models
  ↓
Activation Differences (data/activations/)
  ↓
BatchTopK SAE Training
  ↓
Trained SAE (checkpoints/)
  ↓
Feature Analysis
  ↓
Jailbreak Features (data/jailbreak_features.csv)
  ↓
Dashboard Visualization
```

## Output Files

| File | Size | Description |
|------|------|-------------|
| `data/activations/diff_activations.pkl` | 10-50 GB | Cached activation differences |
| `checkpoints/final_model.pt` | ~500 MB | Trained SAE model |
| `data/jailbreak_features.csv` | <1 MB | Identified safety features |
| WandB logs (if enabled) | Varies | Training metrics and plots |

## Navigation Tips

1. **Start here**: `README.md` or `GETTING_STARTED.md`
2. **Configure**: Edit `configs/config.yaml`
3. **Learn by doing**: Use notebooks in order
4. **Automate**: Use `scripts/run_full_pipeline.sh`
5. **Explore results**: Launch dashboard
6. **Deep dive**: Read source code in `src/`

## Quick File Finder

Need to find...
- **Model loading code**: `src/models/model_loader.py`
- **SAE architecture**: `src/models/sae.py`
- **Training logic**: `src/training/train_sae.py`
- **Activation extraction**: `src/utils/activation_extraction.py`
- **Feature analysis**: `src/analysis/jailbreak_analysis.py`
- **Dashboard**: `dashboards/kl_dashboard.py`
- **Configuration**: `configs/config.yaml`
- **Tutorial**: `GETTING_STARTED.md`
- **Command reference**: `QUICK_REFERENCE.md`

---

For detailed information about any component, see the inline documentation in the source files.
