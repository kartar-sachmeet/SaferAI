#!/usr/bin/env python
"""
Quick test script to verify installation and basic functionality.
Run this after setup to ensure everything is working.
"""

import sys
import torch

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import transformers
        import streamlit
        import plotly
        import pandas
        import numpy
        import yaml
        import einops
        print("✓ All core packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_project_modules():
    """Test that project modules can be imported."""
    print("\nTesting project modules...")
    try:
        sys.path.append('.')
        from src.models import GemmaModelPair, BatchTopKSAE
        from src.utils import DiffActivationCollector
        from src.training import SAETrainer
        from src.analysis import JailbreakAnalyzer
        print("✓ All project modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_sae_creation():
    """Test that we can create an SAE instance."""
    print("\nTesting SAE creation...")
    try:
        from src.models import BatchTopKSAE

        sae = BatchTopKSAE(
            d_model=2304,
            n_latents=18432,
            k=64
        )

        # Test forward pass
        x = torch.randn(2, 2304)
        output = sae(x)

        assert output['reconstruction'].shape == (2, 2304)
        assert output['latents'].shape == (2, 18432)

        print("✓ SAE creation and forward pass successful")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output shape: {output['reconstruction'].shape}")
        print(f"  - Latents shape: {output['latents'].shape}")
        return True
    except Exception as e:
        print(f"✗ SAE test failed: {e}")
        return False

def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    if torch.cuda.is_available():
        print("✓ CUDA is available")
        print(f"  - Device: {torch.cuda.get_device_name(0)}")
        print(f"  - VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("⚠ CUDA not available - will use CPU (slower)")
        print("  This is okay, but training will be slower")
        return True

def test_config():
    """Test that config file can be loaded."""
    print("\nTesting configuration...")
    try:
        import yaml
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        required_keys = ['models', 'sae', 'data', 'dashboard']
        for key in required_keys:
            assert key in config, f"Missing key: {key}"

        print("✓ Configuration file loaded successfully")
        print(f"  - Target layer: {config['models']['target_layer']}")
        print(f"  - SAE latents: {config['sae']['n_latents']}")
        print(f"  - SAE k: {config['sae']['k']}")
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_file_structure():
    """Test that expected directories and files exist."""
    print("\nTesting file structure...")
    from pathlib import Path

    required_dirs = [
        'src/models',
        'src/training',
        'src/utils',
        'src/analysis',
        'configs',
        'notebooks',
        'dashboards',
        'scripts',
        'data/prompts'
    ]

    required_files = [
        'README.md',
        'GETTING_STARTED.md',
        'requirements.txt',
        'configs/config.yaml',
        'src/models/sae.py',
        'src/training/train_sae.py',
        'dashboards/kl_dashboard.py'
    ]

    all_good = True

    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ (missing)")
            all_good = False

    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
            all_good = False

    if all_good:
        print("✓ All required files and directories exist")
    return all_good

def main():
    """Run all tests."""
    print("=" * 50)
    print("Diff-SAE Installation Test")
    print("=" * 50)
    print()

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Project Modules", test_project_modules()))
    results.append(("SAE Creation", test_sae_creation()))
    results.append(("CUDA", test_cuda()))
    results.append(("Configuration", test_config()))
    results.append(("File Structure", test_file_structure()))

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("  1. Login to HuggingFace: huggingface-cli login")
        print("  2. Review GETTING_STARTED.md for usage")
        print("  3. Run your first experiment!")
    else:
        print("\n⚠ Some tests failed. Please check the output above.")
        print("  - Ensure all dependencies are installed: pip install -r requirements.txt")
        print("  - Run setup script: ./scripts/setup_project.sh")

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
