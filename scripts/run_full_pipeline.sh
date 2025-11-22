#!/bin/bash

# Full pipeline for Diff-SAE analysis
# This script runs the complete workflow from data collection to analysis

set -e  # Exit on error

echo "======================================"
echo "Diff-SAE Full Pipeline"
echo "======================================"
echo ""

# Configuration
CONFIG_PATH="configs/config.yaml"
ACTIVATIONS_PATH="data/activations/diff_activations.pkl"
CHECKPOINT_PATH="checkpoints/final_model.pt"
JAILBREAK_FEATURES_PATH="data/jailbreak_features.csv"

# Step 1: Collect Activations
echo "Step 1: Collecting activation differences..."
echo "--------------------------------------"
python -c "
import sys
import torch
from datasets import load_dataset
import yaml

sys.path.append('.')
from src.models import load_gemma_models
from src.utils import DiffActivationCollector

# Load config
with open('$CONFIG_PATH', 'r') as f:
    config = yaml.safe_load(f)

# Load models
print('Loading models...')
models = load_gemma_models('$CONFIG_PATH')

# Load dataset
print('Loading dataset...')
dataset = load_dataset(
    config['data']['dataset_name'],
    split=f'train[:{config[\"data\"][\"num_samples\"]}]'
)

# Extract prompts (adjust based on dataset structure)
prompts = []
for item in dataset:
    if 'messages' in item and item['messages']:
        prompts.append(item['messages'][0]['content'])

print(f'Collected {len(prompts)} prompts')

# Collect activations
device = 'cuda' if torch.cuda.is_available() else 'cpu'
collector = DiffActivationCollector(
    models.base_model,
    models.instruct_model,
    models.tokenizer,
    config['models']['target_layer'],
    device
)

print('Collecting activations...')
activations = collector.collect_diff_activations(
    prompts,
    batch_size=config['data']['batch_size'],
    max_length=config['data']['max_seq_length'],
    return_individual=True
)

# Save
collector.save_activations(activations, '$ACTIVATIONS_PATH')
print('Activations saved!')
"

echo ""
echo "Step 1 complete!"
echo ""

# Step 2: Train SAE
echo "Step 2: Training Diff-SAE..."
echo "--------------------------------------"
python src/training/train_sae.py \
    --config $CONFIG_PATH \
    --activations $ACTIVATIONS_PATH

echo ""
echo "Step 2 complete!"
echo ""

# Step 3: Analyze Jailbreak Features
echo "Step 3: Identifying jailbreak features..."
echo "--------------------------------------"
python src/analysis/jailbreak_analysis.py \
    --config $CONFIG_PATH \
    --checkpoint $CHECKPOINT_PATH \
    --output $JAILBREAK_FEATURES_PATH

echo ""
echo "Step 3 complete!"
echo ""

# Step 4: Launch Dashboard
echo "Step 4: Launching KL Dashboard..."
echo "--------------------------------------"
echo "Starting Streamlit dashboard..."
echo "Access it at: http://localhost:8501"
echo ""
streamlit run dashboards/kl_dashboard.py

echo ""
echo "======================================"
echo "Pipeline complete!"
echo "======================================"
