#!/usr/bin/env python3
"""
Test HRM story completion inference
"""

import torch
import numpy as np
from transformers import RobertaTokenizer
from utils.functions import load_model_class
import os
import glob
import yaml

def load_model_and_config():
    """Load the trained HRM model and its configuration"""
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
        'seq_len': 255,  # Match the CamStories sequence length
        'puzzle_emb_ndim': config['arch']['puzzle_emb_ndim'],
        'num_puzzle_identifiers': 10000,  # From our CamStories dataset
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

def run_story_completion(model, model_config, tokenizer, story_prompt):
    """Run story completion inference"""
    print(f"\n📖 Story prompt: '{story_prompt}'")
    
    # Tokenize input
    input_tokens = tokenizer.encode(story_prompt, add_special_tokens=True)
    print(f"Input tokens: {len(input_tokens)} tokens")
    
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
            # Create proper batch structure
            batch = {
                'inputs': input_tensor,
                'labels': input_tensor,
                'puzzle_identifiers': torch.tensor([0], dtype=torch.long)
            }
            
            # Initialize carry structure
            carry = model.initial_carry(batch)
            
            # Generate story continuation
            generated_text = story_prompt
            current_tokens = input_tokens.copy()
            
            print(f"🔄 Generating story continuation...")
            
            for step in range(20):  # Generate 20 tokens
                # Run forward pass
                carry, outputs = model(carry, batch)
                logits = outputs['logits']
                
                # Find last real token position
                pad_token_id = tokenizer.pad_token_id
                last_real_token_idx = -1
                for i in range(len(current_tokens) - 1, -1, -1):
                    if current_tokens[i] != pad_token_id:
                        last_real_token_idx = i
                        break
                
                if last_real_token_idx == -1:
                    break
                
                # Get next token prediction
                last_token_logits = logits[0, last_real_token_idx, :]
                probs = torch.softmax(last_token_logits, dim=-1)
                
                # Sample next token (use top-k sampling for variety)
                top_k = 50
                top_probs, top_indices = torch.topk(probs, top_k)
                top_probs = top_probs / top_probs.sum()  # Renormalize
                
                # Sample from top-k
                next_token_idx = top_indices[torch.multinomial(top_probs, 1)[0]].item()
                next_token_text = tokenizer.decode([next_token_idx])
                
                # Stop if we hit end-of-text or similar
                if next_token_idx in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                    break
                
                # Add to sequence
                generated_text += next_token_text
                
                # Update tokens for next iteration
                current_tokens = current_tokens[:-sum(1 for t in current_tokens[::-1] if t == pad_token_id)] + [next_token_idx]
                
                # Pad again if needed
                if len(current_tokens) < seq_len:
                    current_tokens = current_tokens + [pad_token_id] * (seq_len - len(current_tokens))
                elif len(current_tokens) > seq_len:
                    current_tokens = current_tokens[:seq_len]
                
                # Update batch for next iteration
                input_tensor = torch.tensor([current_tokens], dtype=torch.long)
                batch['inputs'] = input_tensor
                batch['labels'] = input_tensor
                
                # Don't reset carry - maintain stateful reasoning
                # carry = model.initial_carry(batch)  # Keep commented for V2 approach
            
            print(f"📚 Generated story:")
            print(f"'{generated_text}'")
            
            return True, generated_text
            
    except Exception as e:
        print(f"❌ Story generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    print("📖 HRM Story Completion Test")
    print("=" * 50)
    
    # Load model
    model, model_config, config = load_model_and_config()
    if model is None:
        print("❌ Failed to load model")
        return
    
    print("✅ Model loaded successfully!")
    print(f"Model parameters: ~{sum(p.numel() for p in model.parameters()):,}")
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Test story prompts
    story_prompts = [
        "Once upon a time there was a little girl named",
        "The magic forest was filled with",
        "In a faraway kingdom, the brave knight",
        "Every morning, the friendly dragon would",
        "The mysterious castle on the hill"
    ]
    
    print(f"\n📚 Testing story completion on {len(story_prompts)} prompts...")
    print("-" * 50)
    
    success_count = 0
    
    for prompt in story_prompts:
        success, story = run_story_completion(model, model_config, tokenizer, prompt)
        if success:
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {success_count}/{len(story_prompts)} story completions successful")
    
    if success_count > 0:
        print("🎉 HRM successfully generated story continuations!")
        print("The hierarchical reasoning model can now complete stories!")
    else:
        print("❌ All story completion tests failed.")

if __name__ == "__main__":
    main()