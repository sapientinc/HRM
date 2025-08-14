#!/usr/bin/env python3
"""
Script to create a CamStories-10k dataset in HRM format for testing
story completion with the Hierarchical Reasoning Model.
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

    try:
        from datasets import load_dataset
        print('Datasets library available')
    except ImportError:
        print('Error: Datasets not available. Install with: pip install datasets')
        sys.exit(1)

    # Load CamStories-10k dataset (just the stories file)
    print('Loading CamStories-10k dataset from HuggingFace...')
    try:
        dataset = load_dataset("Piros/CamStories-10k", data_files="camstories_10000.parquet")
        print(f'Dataset loaded successfully!')
        print(f'Available splits: {list(dataset.keys())}')
        
        # Use train split
        train_data = dataset['train']
        print(f'Train split size: {len(train_data)}')
        print(f'Available columns: {train_data.column_names}')
        
        # Show first example
        print(f'First story: {train_data[0]["story"][:100]}...')
        
    except Exception as e:
        print(f'Error loading dataset: {e}')
        sys.exit(1)

    # Setup tokenizer
    print('Setting up RoBERTa tokenizer...')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # Extract stories from the 'story' column
    print('Extracting stories from dataset...')
    stories = []
    
    # Take first 10000 stories for training (manageable size)
    sample_size = min(10000, len(train_data))
    print(f'Using first {sample_size} stories from dataset...')
    
    for i in range(sample_size):
        story = train_data[i]['story'].strip()
        if len(story) > 20:  # Only include substantial stories
            stories.append(story)
    
    print(f'Extracted {len(stories)} valid stories for training')

    # Show some examples
    print('\nExample stories:')
    for i in range(min(3, len(stories))):
        print(f'{i+1}. {stories[i][:100]}...')

    # Tokenize stories
    max_length = 256  # Longer for stories
    inputs = []
    labels = []

    print('Tokenizing stories...')
    for i, story in enumerate(stories):
        if i % 500 == 0:
            print(f'  Progress: {i}/{len(stories)}')
            
        # Tokenize and create input/label pairs for language modeling
        tokens = tokenizer.encode(story, max_length=max_length, truncation=True, padding='max_length')
        
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
    output_dir = 'dataset/camstories_test/train'
    os.makedirs(output_dir, exist_ok=True)

    # Create dataset metadata (following HRM's expected format)
    metadata = {
        'pad_id': tokenizer.pad_token_id,
        'ignore_label_id': -100,
        'blank_identifier_id': 0,
        'vocab_size': tokenizer.vocab_size,
        'seq_len': max_length - 1,
        'num_puzzle_identifiers': len(stories),
        'total_groups': len(stories),
        'mean_puzzle_examples': 1.0,
        'sets': ['camstories_set']
    }

    # Save metadata
    print('Saving dataset metadata...')
    with open(os.path.join(output_dir, 'dataset.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create puzzle identifiers and indices (one per story)
    puzzle_identifiers = np.arange(len(stories), dtype=np.int32)
    puzzle_indices = np.arange(len(stories) + 1, dtype=np.int32)
    group_indices = np.arange(len(stories) + 1, dtype=np.int32)

    # Save data files in HRM format
    print('Saving dataset files...')
    np.save(os.path.join(output_dir, 'camstories_set__inputs.npy'), inputs)
    np.save(os.path.join(output_dir, 'camstories_set__labels.npy'), labels)
    np.save(os.path.join(output_dir, 'camstories_set__puzzle_identifiers.npy'), puzzle_identifiers)
    np.save(os.path.join(output_dir, 'camstories_set__puzzle_indices.npy'), puzzle_indices)
    np.save(os.path.join(output_dir, 'camstories_set__group_indices.npy'), group_indices)

    print('Dataset files created successfully!')
    print('Files created:')
    for file in sorted(os.listdir(output_dir)):
        file_path = os.path.join(output_dir, file)
        size = os.path.getsize(file_path)
        print(f'  - {file} ({size:,} bytes)')
        
    print(f'\nCamStories dataset ready at: {output_dir}')
    print('You can now train HRM with real story data!')

if __name__ == '__main__':
    main()