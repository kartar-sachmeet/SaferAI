#!/bin/bash

# Display project information and status

echo "======================================"
echo "Diff-SAE Project Information"
echo "======================================"
echo ""

# Project structure
echo "📁 Project Structure:"
echo "---"
find . -maxdepth 2 -type d -not -path '*/.*' | grep -v __pycache__ | sort | sed 's|^\./||' | sed 's|^|  |'
echo ""

# File counts
echo "📊 File Statistics:"
echo "---"
echo "  Python files: $(find . -name '*.py' -not -path '*/.*' | wc -l | tr -d ' ')"
echo "  Notebooks: $(find . -name '*.ipynb' | wc -l | tr -d ' ')"
echo "  Config files: $(find . -name '*.yaml' -o -name '*.yml' | wc -l | tr -d ' ')"
echo "  Shell scripts: $(find . -name '*.sh' | wc -l | tr -d ' ')"
echo "  Documentation: $(find . -name '*.md' | wc -l | tr -d ' ')"
echo ""

# Code statistics
echo "💻 Code Statistics:"
echo "---"
python_lines=$(find . -name '*.py' -not -path '*/.*' -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
echo "  Total Python lines: ${python_lines:-0}"
echo ""

# Check for key files
echo "✓ Key Files:"
echo "---"
for file in "README.md" "GETTING_STARTED.md" "requirements.txt" "configs/config.yaml"; do
    if [ -f "$file" ]; then
        size=$(wc -l < "$file" 2>/dev/null || echo "0")
        echo "  ✓ $file ($size lines)"
    else
        echo "  ✗ $file (missing)"
    fi
done
echo ""

# Check for data/checkpoints
echo "📦 Data & Checkpoints:"
echo "---"
if [ -d "data/activations" ]; then
    act_count=$(ls -1 data/activations/*.pkl 2>/dev/null | wc -l | tr -d ' ')
    echo "  Activation files: $act_count"
else
    echo "  Activation files: 0 (directory not found)"
fi

if [ -d "checkpoints" ]; then
    ckpt_count=$(ls -1 checkpoints/*.pt 2>/dev/null | wc -l | tr -d ' ')
    echo "  Checkpoint files: $ckpt_count"
    if [ "$ckpt_count" -gt 0 ]; then
        echo "  Latest checkpoint:"
        ls -lt checkpoints/*.pt 2>/dev/null | head -1 | awk '{print "    " $9 " (" $6 " " $7 " " $8 ")"}'
    fi
else
    echo "  Checkpoint files: 0 (directory not found)"
fi
echo ""

# Module summary
echo "🔧 Implemented Modules:"
echo "---"
echo "  src/models/"
echo "    - model_loader.py (Gemma 2 loading)"
echo "    - sae.py (BatchTopK SAE)"
echo "  src/training/"
echo "    - train_sae.py (Training pipeline)"
echo "  src/utils/"
echo "    - activation_extraction.py (Activation collection)"
echo "  src/analysis/"
echo "    - jailbreak_analysis.py (Feature identification)"
echo "  dashboards/"
echo "    - kl_dashboard.py (Interactive dashboard)"
echo ""

# Workflow summary
echo "🚀 Quick Start Commands:"
echo "---"
echo "  Setup:         ./scripts/setup_project.sh"
echo "  Full pipeline: ./scripts/run_full_pipeline.sh"
echo "  Dashboard:     streamlit run dashboards/kl_dashboard.py"
echo "  Notebooks:     jupyter notebook notebooks/"
echo ""

# Documentation
echo "📚 Documentation:"
echo "---"
echo "  README.md           - Full documentation"
echo "  GETTING_STARTED.md  - Step-by-step tutorial"
echo "  configs/config.yaml - Configuration reference"
echo ""

# Reference
echo "🔗 Reference:"
echo "---"
echo "  LessWrong Post: https://www.lesswrong.com/posts/XPNJSa3BxMAN4ZXc7/sae-on-activation-differences"
echo "  Gemma 2 Base:   https://huggingface.co/google/gemma-2-2b"
echo "  Gemma 2 IT:     https://huggingface.co/google/gemma-2-2b-it"
echo ""

echo "======================================"
echo "For detailed instructions, see:"
echo "  • GETTING_STARTED.md - Quick tutorial"
echo "  • README.md - Full documentation"
echo "======================================"
