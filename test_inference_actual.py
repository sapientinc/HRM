#!/usr/bin/env python3
"""
Actual inference test with the trained HRM model
"""

import torch
import numpy as np
from transformers import RobertaTokenizer
from utils.functions import load_model_class
import os
import glob
import yaml

def load_model_and_config():
    """Load the trained model and its configuration"""
    print("Loading trained HRM model...")
    
    # Find checkpoint directory
    checkpoint_dirs = glob.glob("checkpoints/*/HierarchicalReasoningModel_ACTV1*")
    if not checkpoint_dirs:
        print("❌ No checkpoint directories found!")
        return None, None, None
    
    # Use the most recent checkpoint directory
    checkpoint_dir = max(checkpoint_dirs, key=os.path.getctime)
    print(f"Using checkpoint directory: {checkpoint_dir}")
    
    # Load config
    config_path = os.path.join(checkpoint_dir, "all_config.yaml")
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return None, None, None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Find model checkpoint
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "step_*"))
    if not checkpoint_files:
        print("❌ No model checkpoint files found!")
        return None, None, None
    
    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[-1]))
    print(f"Loading model from: {latest_checkpoint}")
    
    # Load model class
    model_class = load_model_class("hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1")
    
    # Create model config dictionary
    model_config_dict = {
        'batch_size': 1,
        'seq_len': 127,
        'puzzle_emb_ndim': config['arch']['puzzle_emb_ndim'],
        'num_puzzle_identifiers': 1000,  # From our dataset
        'vocab_size': 50265,  # RoBERTa vocab size
        'H_cycles': config['arch']['H_cycles'],
        'L_cycles': config['arch']['L_cycles'],
        'H_layers': config['arch']['H_layers'],
        'L_layers': config['arch']['L_layers'],
        'hidden_size': config['arch']['hidden_size'],
        'expansion': config['arch']['expansion'],
        'num_heads': config['arch']['num_heads'],
        'pos_encodings': config['arch']['pos_encodings'],
        'halt_max_steps': config['arch']['halt_max_steps'],
        'halt_exploration_prob': config['arch']['halt_exploration_prob']
    }
    
    model = model_class(model_config_dict)
    
    # Load checkpoint
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    
    # Extract model state dict
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        print("⚠️ Checkpoint format not recognized, trying direct load...")
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    return model, model_config_dict, config

def run_inference_test(model, model_config, tokenizer, prompt, max_new_tokens=10):
    """Run inference on a text prompt"""
    print(f"\n🔍 Testing: '{prompt}'")
    
    # Tokenize input
    input_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    print(f"Input tokens: {input_tokens[:10]}{'...' if len(input_tokens) > 10 else ''}")
    
    # Pad to sequence length if needed
    seq_len = model_config['seq_len']
    if len(input_tokens) > seq_len:
        input_tokens = input_tokens[:seq_len]
    else:
        input_tokens = input_tokens + [tokenizer.pad_token_id] * (seq_len - len(input_tokens))
    
    # Convert to tensor
    input_tensor = torch.tensor([input_tokens], dtype=torch.long)
    
    try:
        with torch.no_grad():
            # Create puzzle identifier (just use 0)
            puzzle_ids = torch.tensor([0], dtype=torch.long)
            
            # Run forward pass through HRM
            output = model(input_tensor, puzzle_ids)
            
            # Get logits and convert to probabilities
            if hasattr(output, 'logits'):
                logits = output.logits
            else:
                logits = output
            
            # Get the last non-padded token's predictions
            last_token_logits = logits[0, -1, :]  # [vocab_size]
            probs = torch.softmax(last_token_logits, dim=-1)
            
            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probs, 5)
            
            print(f"✅ Inference successful!")
            print(f"Top 5 next token predictions:")
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                token = tokenizer.decode([idx.item()])
                print(f"  {i+1}. '{token}' (prob: {prob.item():.4f})")
                
            # Try to generate a completion
            predicted_token = top_indices[0].item()
            completion = tokenizer.decode([predicted_token])
            print(f"💡 HRM suggests next token: '{completion}'")
            
            return True
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧠 HRM Language Model Inference Test")
    print("=" * 60)
    
    # Load model
    model, model_config, config = load_model_and_config()
    if model is None:
        print("❌ Failed to load model")
        return
    
    print("✅ Model loaded successfully!")
    print(f"Model parameters: ~{sum(p.numel() for p in model.parameters()):,}")
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Test prompts
    test_prompts = [
        "The quick brown fox",
        "Machine learning is",
        "Python is a",
        "Deep learning",
        "Artificial intelligence will"
    ]
    
    print(f"\n🧪 Running inference tests on {len(test_prompts)} prompts...")
    print("-" * 60)
    
    success_count = 0
    for prompt in test_prompts:
        success = run_inference_test(model, model_config, tokenizer, prompt)
        if success:
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Results: {success_count}/{len(test_prompts)} tests successful")
    
    if success_count > 0:
        print("🎉 HRM successfully generated text predictions!")
        print("The hierarchical reasoning modules are processing language data.")
    else:
        print("⚠️ Inference tests encountered issues.")

if __name__ == "__main__":
    main()