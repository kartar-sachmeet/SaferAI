#!/bin/bash

# Project setup script for Diff-SAE
# This script initializes the environment and verifies installation

set -e

echo "======================================"
echo "Diff-SAE Project Setup"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if python -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "✓ Python version is compatible"
else
    echo "✗ Python 3.9+ required"
    exit 1
fi
echo ""

# Check for GPU
echo "Checking for CUDA/GPU..."
if python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    echo "✓ PyTorch is installed"
    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
else
    echo "⚠ PyTorch not installed yet (will be installed with requirements)"
fi
echo ""

# Install requirements
echo "Installing requirements..."
if [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt
    echo "✓ Requirements installed"
else
    echo "✗ requirements.txt not found"
    exit 1
fi
echo ""

# Verify installations
echo "Verifying installations..."
python -c "
import torch
import transformers
import streamlit
import plotly
print('✓ PyTorch:', torch.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ Streamlit:', streamlit.__version__)
print('✓ Plotly:', plotly.__version__)
"
echo ""

# Check directory structure
echo "Verifying directory structure..."
for dir in src configs data checkpoints notebooks dashboards scripts; do
    if [ -d "$dir" ]; then
        echo "✓ $dir/"
    else
        echo "✗ $dir/ missing"
    fi
done
echo ""

# Create __init__.py files if missing
echo "Creating __init__.py files..."
touch src/__init__.py
echo "✓ src/__init__.py"
echo ""

# Check CUDA availability
echo "CUDA/GPU Status:"
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA is available')
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('⚠ CUDA not available - will run on CPU (slower)')
"
echo ""

# HuggingFace login check
echo "Checking HuggingFace authentication..."
if python -c "from huggingface_hub import HfFolder; token = HfFolder.get_token(); exit(0 if token else 1)" 2>/dev/null; then
    echo "✓ HuggingFace token found"
else
    echo "⚠ HuggingFace token not found"
    echo "  To access Gemma models, run: huggingface-cli login"
    echo "  Get your token from: https://huggingface.co/settings/tokens"
fi
echo ""

# Test imports
echo "Testing project imports..."
python -c "
import sys
sys.path.append('.')
try:
    from src.models import GemmaModelPair, BatchTopKSAE
    from src.utils import DiffActivationCollector
    from src.training import SAETrainer
    from src.analysis import JailbreakAnalyzer
    print('✓ All project modules can be imported')
except Exception as e:
    print(f'✗ Import error: {e}')
    exit(1)
"
echo ""

# Summary
echo "======================================"
echo "Setup Summary"
echo "======================================"
echo ""
echo "✓ Python environment configured"
echo "✓ Dependencies installed"
echo "✓ Project structure verified"
echo "✓ Modules can be imported"
echo ""
echo "Next steps:"
echo "1. Login to HuggingFace (if not done): huggingface-cli login"
echo "2. Review configuration: configs/config.yaml"
echo "3. Run the pipeline: ./scripts/run_full_pipeline.sh"
echo "   OR use the notebooks in: notebooks/"
echo ""
echo "For help, see: README.md"
echo ""
