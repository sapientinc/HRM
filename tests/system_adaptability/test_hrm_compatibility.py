#!/usr/bin/env python3
"""
Isolated HRM compatibility test for cross-platform support.
Tests model loading and basic inference without modifying core functionality.
"""

import torch
import os
import sys
from pathlib import Path

def test_device_detection():
    """Test universal device detection"""
    print("=== DEVICE DETECTION TEST ===")
    
    def get_device():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    device = get_device()
    print(f"✅ Detected device: {device}")
    
    # Test basic tensor ops
    try:
        x = torch.randn(10, 10).to(device)
        y = torch.matmul(x, x.T)
        print(f"✅ Basic tensor operations work on {device}")
        return device
    except Exception as e:
        print(f"❌ Tensor operations failed: {e}")
        return None

def test_flash_attn_fallback():
    """Test flash attention conditional fallback"""
    print("\\n=== FLASH ATTENTION FALLBACK TEST ===")
    
    try:
        from models.layers import flash_attn_func
        if flash_attn_func is not None:
            print("✅ Flash attention available - using optimized path")
            return "flash_attn"
        else:
            print("✅ Flash attention not available - using PyTorch fallback")
            return "pytorch_fallback"
    except Exception as e:
        print(f"❌ Flash attention test failed: {e}")
        return None

def test_model_import():
    """Test model import and basic initialization"""
    print("\\n=== MODEL IMPORT TEST ===")
    
    try:
        # Test if we can import the model components
        from models.layers import Attention, rms_norm
        print("✅ Core model components import successfully")
        
        # Test basic attention layer creation
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        attn = Attention(hidden_size=64, num_heads=8, causal=False)
        print("✅ Attention layer creates successfully")
        
        # Test basic forward pass
        x = torch.randn(1, 10, 64)
        with torch.no_grad():
            out = attn(x)
        print(f"✅ Attention forward pass works - output shape: {out.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model import/creation failed: {e}")
        return False

def test_checkpoint_loading():
    """Test checkpoint loading capability"""
    print("\\n=== CHECKPOINT LOADING TEST ===")
    
    checkpoint_path = "checkpoints/HRM-ARC-2/checkpoint"
    config_path = "checkpoints/HRM-ARC-2/all_config.yaml"
    
    if not os.path.exists(checkpoint_path):
        print("⚠️  No checkpoint found - download first with HuggingFace")
        return False
    
    try:
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        
        # Test checkpoint loading
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"✅ Checkpoint loads successfully - {len(checkpoint)} parameters")
        
        # Test config loading
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Config loads successfully - model: {config['arch']['name']}")
        
        return True
    except Exception as e:
        print(f"❌ Checkpoint loading failed: {e}")
        return False

def test_dataset_compatibility():
    """Test dataset format compatibility"""
    print("\\n=== DATASET COMPATIBILITY TEST ===")
    
    arc_data_path = "data/arc-2-aug-1000"
    
    if not os.path.exists(arc_data_path):
        print("⚠️  ARC dataset not found - build with:")
        print("   python dataset/build_arc_dataset.py --output-dir data/arc-2-aug-1000 --num-aug 1000")
        return False
    
    try:
        train_path = os.path.join(arc_data_path, "train", "dataset.json")
        test_path = os.path.join(arc_data_path, "test", "dataset.json")
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            print("✅ ARC dataset structure is correct")
            
            import json
            with open(train_path, 'r') as f:
                train_data = json.load(f)
            print(f"✅ Train dataset: {len(train_data)} examples")
            
            return True
        else:
            print("❌ Missing dataset.json files")
            return False
            
    except Exception as e:
        print(f"❌ Dataset compatibility test failed: {e}")
        return False

def run_full_compatibility_test():
    """Run complete HRM compatibility test suite"""
    print("🧪 HRM CROSS-PLATFORM COMPATIBILITY TEST")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Device detection
    device = test_device_detection()
    results['device'] = device is not None
    
    # Test 2: Flash attention fallback
    attn_mode = test_flash_attn_fallback()
    results['attention'] = attn_mode is not None
    
    # Test 3: Model import
    results['model_import'] = test_model_import()
    
    # Test 4: Checkpoint loading
    results['checkpoint'] = test_checkpoint_loading()
    
    # Test 5: Dataset compatibility
    results['dataset'] = test_dataset_compatibility()
    
    # Summary
    print("\\n" + "=" * 60)
    print("🏁 COMPATIBILITY TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {test.replace('_', ' ').title()}: {'PASS' if status else 'FAIL'}")
    
    print(f"\\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 HRM is fully compatible with this system!")
        print("\\n📋 TO RUN CLASSIFICATION:")
        print("python evaluate.py checkpoint=checkpoints/HRM-ARC-2/checkpoint")
    else:
        print("⚠️  Some compatibility issues detected")
        if not results['dataset']:
            print("\\n🔧 TO FIX: Build the ARC dataset first")
        if not results['checkpoint']:
            print("\\n🔧 TO FIX: Download checkpoint from HuggingFace")
    
    return passed == total

if __name__ == "__main__":
    success = run_full_compatibility_test()
    sys.exit(0 if success else 1)