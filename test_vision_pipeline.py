#!/usr/bin/env python3
"""
Test script to verify the vision HRM pipeline works correctly.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from models.hrm.hrm_vision_v1 import HierarchicalReasoningModel_VisionV1
        from models.vision_losses import VisionClassificationLossHead
        from dataset.build_cifar_dataset import DataProcessConfig
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_model_creation():
    """Test that the vision model can be created."""
    print("Testing model creation...")
    
    try:
        from models.hrm.hrm_vision_v1 import HierarchicalReasoningModel_VisionV1
        
        # Create a minimal config
        config = {
            "hidden_size": 256,
            "num_heads": 4,
            "expansion": 2,
            "H_layers": 2,
            "L_layers": 2,
            "H_cycles": 1,
            "L_cycles": 1,
            "vocab_size": 256,
            "seq_len": 3072,  # 32x32 image as patches
            "num_classes": 10,
            "patch_size": 4,
            "image_size": 32,
            "num_puzzle_identifiers": 1,
            "puzzle_emb_ndim": 64,
            "halt_exploration_prob": 0.0,
            "halt_max_steps": 1,
            "batch_size": 2,
            "forward_dtype": "float32"
        }
        
        model = HierarchicalReasoningModel_VisionV1(config)
        print("✓ Model creation successful")
        
        # Test forward pass
        batch_size = 2
        seq_len = 3072
        batch = {
            "inputs": torch.randint(0, 256, (batch_size, seq_len)),
            "labels": torch.randint(0, 10, (batch_size, seq_len))
        }
        
        carry = model.initial_carry(batch)
        carry, metrics, preds, _, all_finish = model(carry, batch)
        
        print(f"✓ Forward pass successful")
        print(f"  - Loss: {metrics['loss']:.4f}")
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - All finish: {all_finish}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation/forward pass error: {e}")
        return False

def test_dataset_config():
    """Test dataset configuration."""
    print("Testing dataset configuration...")
    
    try:
        from dataset.build_cifar_dataset import DataProcessConfig
        
        config = DataProcessConfig(
            dataset_name="CIFAR10",
            output_dir="test_data",
            seed=42,
            num_aug=10,
            image_size=32,
            patch_size=4,
            num_channels=3
        )
        
        print("✓ Dataset configuration successful")
        print(f"  - Dataset: {config.dataset_name}")
        print(f"  - Output dir: {config.output_dir}")
        print(f"  - Augmentations: {config.num_aug}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset configuration error: {e}")
        return False

def test_patch_processing():
    """Test image patch processing."""
    print("Testing patch processing...")
    
    try:
        from dataset.build_cifar_dataset import image_to_patches, patches_to_sequence
        
        # Create a dummy 32x32 RGB image
        image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        
        # Convert to patches
        patches = image_to_patches(image, patch_size=4)
        print(f"✓ Image to patches: {image.shape} -> {patches.shape}")
        
        # Convert to sequence
        seq = patches_to_sequence(patches, max_seq_len=3072)
        print(f"✓ Patches to sequence: {patches.shape} -> {seq.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Patch processing error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Vision HRM Pipeline")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_dataset_config,
        test_patch_processing,
        test_model_creation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The vision HRM pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Build the CIFAR dataset: python dataset/build_cifar_dataset.py")
        print("2. Train the model: python train_cifar.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
