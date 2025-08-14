#!/usr/bin/env python3
"""
Script to create a RoBERTa-compatible dataset in HRM format for testing
language modeling with the Hierarchical Reasoning Model.
"""

import sys
import numpy as np
import os
import json

def main():
    print('Python version:', sys.version)

    # Try to import required libraries
    try:
        import torch
        print('Torch version:', torch.__version__)
    except ImportError:
        print('Error: Torch not available. Install with: pip install torch')
        sys.exit(1)

    try:
        from transformers import RobertaTokenizer
        print('Transformers available')
    except ImportError:
        print('Error: Transformers not available. Install with: pip install transformers')
        sys.exit(1)

    # Download some sample text data for testing
    print('Setting up RoBERTa tokenizer...')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Create sample text dataset
    texts = [
        'The quick brown fox jumps over the lazy dog.',
        'Machine learning is transforming the world.',
        'Natural language processing helps computers understand text.',
        'Deep learning models require large amounts of data.',
        'Artificial intelligence will change many industries.',
        'Python is a popular programming language for AI.',
        'Neural networks learn patterns from training data.',
        'Computer vision enables machines to see and understand images.',
        'Reinforcement learning trains agents through trial and error.',
        'Large language models can generate human-like text.'
    ] * 100  # Repeat to get 1000 samples

    print(f'Created {len(texts)} text samples')

    # Tokenize texts
    max_length = 128
    inputs = []
    labels = []

    print('Tokenizing texts...')
    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f'  Progress: {i}/{len(texts)}')
            
        # Tokenize and create input/label pairs for language modeling
        tokens = tokenizer.encode(text, max_length=max_length, truncation=True, padding='max_length')
        
        # For language modeling, input is tokens[:-1] and label is tokens[1:]
        input_tokens = tokens[:-1]
        label_tokens = tokens[1:]
        
        inputs.append(input_tokens)
        labels.append(label_tokens)

    # Convert to numpy arrays
    inputs = np.array(inputs, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)

    print(f'Input shape: {inputs.shape}')
    print(f'Labels shape: {labels.shape}')
    print(f'Vocab size: {tokenizer.vocab_size}')

    # Create directories and save in HRM format
    output_dir = 'dataset/roberta_test/train'
    os.makedirs(output_dir, exist_ok=True)

    # Create dataset metadata (following HRM's expected format)
    metadata = {
        'pad_id': tokenizer.pad_token_id,
        'ignore_label_id': -100,
        'blank_identifier_id': 0,
        'vocab_size': tokenizer.vocab_size,
        'seq_len': max_length - 1,
        'num_puzzle_identifiers': len(texts),
        'total_groups': len(texts),
        'mean_puzzle_examples': 1.0,
        'sets': ['roberta_set']
    }

    # Save metadata
    print('Saving dataset metadata...')
    with open(os.path.join(output_dir, 'dataset.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create puzzle identifiers and indices (one per text sample)
    puzzle_identifiers = np.arange(len(texts), dtype=np.int32)
    puzzle_indices = np.arange(len(texts) + 1, dtype=np.int32)
    group_indices = np.arange(len(texts) + 1, dtype=np.int32)

    # Save data files in HRM format
    print('Saving dataset files...')
    np.save(os.path.join(output_dir, 'roberta_set__inputs.npy'), inputs)
    np.save(os.path.join(output_dir, 'roberta_set__labels.npy'), labels)
    np.save(os.path.join(output_dir, 'roberta_set__puzzle_identifiers.npy'), puzzle_identifiers)
    np.save(os.path.join(output_dir, 'roberta_set__puzzle_indices.npy'), puzzle_indices)
    np.save(os.path.join(output_dir, 'roberta_set__group_indices.npy'), group_indices)

    print('Dataset files created successfully!')
    print('Files created:')
    for file in sorted(os.listdir(output_dir)):
        file_path = os.path.join(output_dir, file)
        size = os.path.getsize(file_path)
        print(f'  - {file} ({size:,} bytes)')
        
    print(f'\nDataset ready at: {output_dir}')
    print('You can now test HRM with this language dataset!')

if __name__ == '__main__':
    main()