#!/usr/bin/env python3
"""
Classification test script for HRM using holon tags data.
Tests the model's ability to classify text using the tag taxonomy.
"""

import pandas as pd
import torch
import os
import sys
from pathlib import Path

def get_device():
    """Universal device detection for HRM testing"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_classification_data():
    """Load holon tags and tags master for classification testing"""
    dataset_path = Path("dataset")
    
    # Load holon tags (the data to classify)
    holon_tags = pd.read_csv(dataset_path / "holon_tags.csv")
    print(f"Loaded {len(holon_tags)} holon entries")
    
    # Load tags master (the classification taxonomy)
    tags_master = pd.read_csv(dataset_path / "tags_master.csv")
    print(f"Loaded {len(tags_master)} tag definitions")
    
    return holon_tags, tags_master

def prepare_classification_examples():
    """Prepare text examples for classification testing"""
    holon_tags, tags_master = load_classification_data()
    
    examples = []
    for _, row in holon_tags.iterrows():
        example = {
            'id': row['holon_id'],
            'title': row['title'],
            'description': row['description'],
            'true_tags': row['tags'].split('; ') if pd.notna(row['tags']) else [],
            'full_text': f"{row['title']}: {row['description']}"
        }
        examples.append(example)
    
    print(f"Prepared {len(examples)} classification examples")
    return examples, tags_master

def test_device_compatibility():
    """Test basic tensor operations on the detected device"""
    device = get_device()
    print(f"Testing device compatibility: {device}")
    
    try:
        # Test tensor creation and operations
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = torch.matmul(x, y)
        
        print(f"✅ Device test passed - tensor operations work on {device}")
        return True
    except Exception as e:
        print(f"❌ Device test failed: {e}")
        return False

def run_classification_test():
    """Main classification test runner"""
    print("=" * 60)
    print("HRM CLASSIFICATION TEST")
    print("=" * 60)
    
    # Test device compatibility
    if not test_device_compatibility():
        return False
    
    # Load and prepare data
    try:
        examples, tags_master = prepare_classification_examples()
        
        print(f"\\nClassification Test Data Summary:")
        print(f"- Examples to classify: {len(examples)}")
        print(f"- Available tags: {len(tags_master)}")
        print(f"- Device: {get_device()}")
        
        # Show sample data
        print(f"\\nSample classification example:")
        sample = examples[0]
        print(f"ID: {sample['id']}")
        print(f"Title: {sample['title']}")
        print(f"Description: {sample['description'][:100]}...")
        print(f"True tags: {sample['true_tags']}")
        
        print(f"\\nAvailable tag categories:")
        for _, tag in tags_master.head(10).iterrows():
            print(f"- {tag['tag']}: {tag['description']}")
        
        print(f"\\n✅ Classification test data prepared successfully!")
        print(f"\\n📋 NEXT STEPS:")
        print(f"1. Load a pretrained HRM model checkpoint")
        print(f"2. Run inference on the prepared examples") 
        print(f"3. Compare predicted tags vs true tags")
        
        return True
        
    except Exception as e:
        print(f"❌ Classification test failed: {e}")
        return False

if __name__ == "__main__":
    success = run_classification_test()
    sys.exit(0 if success else 1)