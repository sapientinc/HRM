#!/usr/bin/env python3
"""
Fixed inference script that properly handles the HRM ACT carry structure
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

def run_actual_inference(model, model_config, tokenizer, prompt):
    """Run actual inference with proper carry structure"""
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
            # Create proper batch structure that HRM expects
            batch = {
                'inputs': input_tensor,
                'labels': input_tensor,  # For inference, labels same as inputs
                'puzzle_identifiers': torch.tensor([0], dtype=torch.long)
            }
            
            # Initialize proper carry structure
            carry = model.initial_carry(batch)
            
            print("✅ Carry structure initialized properly")
            
            # Run forward pass through HRM with proper carry
            carry, outputs = model(carry, batch)
            
            # Extract logits
            logits = outputs['logits']  # [batch_size, seq_len, vocab_size]
            
            # Get the last non-padded token's predictions
            # Find the last non-pad token
            pad_token_id = tokenizer.pad_token_id
            last_real_token_idx = -1
            for i in range(len(input_tokens) - 1, -1, -1):
                if input_tokens[i] != pad_token_id:
                    last_real_token_idx = i
                    break
            
            if last_real_token_idx == -1:
                last_real_token_idx = 0
            
            last_token_logits = logits[0, last_real_token_idx, :]  # [vocab_size]
            probs = torch.softmax(last_token_logits, dim=-1)
            
            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probs, 5)
            
            print(f"✅ Inference successful!")
            print(f"Last real token position: {last_real_token_idx}")
            print(f"Top 5 next token predictions:")
            
            actual_predictions = []
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                token = tokenizer.decode([idx.item()]).strip()
                prob_val = prob.item()
                actual_predictions.append((token, prob_val))
                print(f"  {i+1}. '{token}' (prob: {prob_val:.4f})")
            
            # Show the most likely completion
            best_token = actual_predictions[0][0]
            best_prob = actual_predictions[0][1]
            print(f"💡 HRM's top prediction: '{best_token}' (confidence: {best_prob:.4f})")
            
            # Try to generate a few more tokens
            print(f"\n🔄 Generating longer completion...")
            generated_text = prompt
            current_tokens = input_tokens.copy()
            
            for step in range(3):  # Generate 3 more tokens
                # Get next token
                if len(actual_predictions) > 0:
                    next_token_text = actual_predictions[0][0]
                    next_token_id = top_indices[0].item()
                    
                    # Add to sequence
                    generated_text += next_token_text
                    current_tokens = current_tokens[:-sum(1 for t in current_tokens[::-1] if t == pad_token_id)] + [next_token_id]
                    
                    # Pad again if needed
                    if len(current_tokens) < seq_len:
                        current_tokens = current_tokens + [pad_token_id] * (seq_len - len(current_tokens))
                    elif len(current_tokens) > seq_len:
                        current_tokens = current_tokens[:seq_len]
                    
                    # Prepare for next iteration
                    input_tensor = torch.tensor([current_tokens], dtype=torch.long)
                    batch['inputs'] = input_tensor
                    batch['labels'] = input_tensor
                    
                    # Get next prediction
                    carry = model.initial_carry(batch)
                    carry, outputs = model(carry, batch)
                    logits = outputs['logits']
                    
                    # Find last real token position again
                    last_real_token_idx = -1
                    for i in range(len(current_tokens) - 1, -1, -1):
                        if current_tokens[i] != pad_token_id:
                            last_real_token_idx = i
                            break
                    
                    if last_real_token_idx >= 0:
                        last_token_logits = logits[0, last_real_token_idx, :]
                        probs = torch.softmax(last_token_logits, dim=-1)
                        top_probs, top_indices = torch.topk(probs, 5)
                        actual_predictions = [(tokenizer.decode([idx.item()]).strip(), prob.item()) 
                                           for prob, idx in zip(top_probs, top_indices)]
            
            print(f"📝 Generated completion: '{generated_text}'")
            
            return True, actual_predictions
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    print("🧠 HRM Language Model - REAL Inference Test")
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
    
    print(f"\n🧪 Running REAL inference tests on {len(test_prompts)} prompts...")
    print("-" * 60)
    
    success_count = 0
    all_results = []
    
    for prompt in test_prompts:
        success, predictions = run_actual_inference(model, model_config, tokenizer, prompt)
        if success:
            success_count += 1
            all_results.append((prompt, predictions))
    
    print("\n" + "=" * 60)
    print(f"📊 REAL Results: {success_count}/{len(test_prompts)} tests successful")
    
    if success_count > 0:
        print("\n🎉 HRM successfully generated ACTUAL text predictions!")
        print("These are the real outputs from the trained hierarchical reasoning model:")
        print("-" * 40)
        for prompt, predictions in all_results:
            print(f"\nPrompt: '{prompt}'")
            print(f"HRM's actual top prediction: '{predictions[0][0]}' ({predictions[0][1]:.4f})")
    else:
        print("❌ All inference tests failed.")

if __name__ == "__main__":
    main()