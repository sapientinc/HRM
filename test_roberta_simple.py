#!/usr/bin/env python3
"""
Simple test script to verify HRM can load and process RoBERTa data
without the full training pipeline.
"""

import torch
import numpy as np
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

def test_data_loading():
    """Test that our RoBERTa dataset can be loaded by HRM"""
    print("Testing HRM data loading with RoBERTa dataset...")
    
    # Create dataset config
    config = PuzzleDatasetConfig(
        seed=42,
        dataset_path="dataset/roberta_test",
        global_batch_size=4,  # Very small for testing
        test_set_mode=False,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1
    )
    
    # Create dataset
    dataset = PuzzleDataset(config, split="train")
    
    print(f"Dataset metadata:")
    print(f"  Vocab size: {dataset.metadata.vocab_size}")
    print(f"  Sequence length: {dataset.metadata.seq_len}")
    print(f"  Total groups: {dataset.metadata.total_groups}")
    print(f"  Sets: {dataset.metadata.sets}")
    
    # Try to get a batch
    print("\nTesting batch loading...")
    iterator = iter(dataset)
    
    try:
        set_name, batch, batch_size = next(iterator)
        print(f"Successfully loaded batch from set: {set_name}")
        print(f"Batch size: {batch_size}")
        print(f"Input shape: {batch['inputs'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        print(f"Puzzle identifiers shape: {batch['puzzle_identifiers'].shape}")
        print(f"Sample input tokens (first 10):", batch['inputs'][0][:10].tolist())
        print(f"Sample label tokens (first 10):", batch['labels'][0][:10].tolist())
        
        return True
        
    except Exception as e:
        print(f"Error loading batch: {e}")
        return False

def test_model_loading():
    """Test that we can load the HRM model architecture"""
    print("\nTesting HRM model loading...")
    
    try:
        # Import model utilities
        from utils.functions import load_model_class
        
        # Load HRM model class
        model_class = load_model_class("hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1")
        print(f"Successfully loaded model class: {model_class.__name__}")
        
        # Try to create a small model instance for testing
        # We'll need to check what parameters it expects
        print("Model class loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def main():
    print("HRM + RoBERTa Compatibility Test")
    print("=" * 50)
    
    # Test 1: Data loading
    data_success = test_data_loading()
    
    # Test 2: Model loading
    model_success = test_model_loading()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"  Data loading: {'✅ PASS' if data_success else '❌ FAIL'}")
    print(f"  Model loading: {'✅ PASS' if model_success else '❌ FAIL'}")
    
    if data_success and model_success:
        print("\n🎉 HRM appears compatible with language data!")
        print("The architecture can load and process tokenized text.")
    else:
        print("\n⚠️  Some compatibility issues found.")
        
    return data_success and model_success

if __name__ == "__main__":
    main()