# Research Scripts

This directory contains scripts used for the original research but not part of the core `hierarchical_reasoning_model` package.

## Overview

The core package (`hierarchical_reasoning_model`) provides only the model architecture components. These scripts provide reference implementations for dataset processing, training, and evaluation.

## Directory Structure

```
scripts/
├── data/                    # Dataset loading utilities
│   ├── dataset.py          # PuzzleDataset for loading preprocessed data
│   ├── metadata.py         # Dataset metadata and transformations
│   └── __init__.py
├── build_arc_dataset.py    # Preprocess ARC-AGI puzzles
├── build_sudoku_dataset.py # Preprocess Sudoku datasets
├── build_maze_dataset.py   # Preprocess maze datasets
├── train.py                # Full training loop with Hydra/wandb
├── evaluate.py             # Distributed evaluation script
└── README.md               # This file
```

## Dataset Builders

### ARC Dataset Builder

Preprocesses ARC-AGI and ConceptARC puzzles with data augmentation:

```bash
python scripts/build_arc_dataset.py \
    --dataset-dirs dataset/raw-data/ARC-AGI/data dataset/raw-data/ConceptARC/corpus \
    --output-dir data/arc-aug-1000 \
    --seed 42 \
    --num-aug 1000
```

**Features**:
- Dihedral transformations (8 orientations) for augmentation
- Color permutations for invariance
- JSON metadata generation
- Memory-mapped numpy arrays for efficient loading

### Sudoku Dataset Builder

Downloads and preprocesses Sudoku puzzles from HuggingFace:

```bash
python scripts/build_sudoku_dataset.py \
    --source-repo sapientinc/sudoku-extreme \
    --output-dir data/sudoku-extreme-full \
    --subsample-size 1000 \
    --min-difficulty 5 \
    --num-aug 0
```

**Features**:
- Downloads from HuggingFace datasets
- Difficulty filtering
- Optional subsampling
- Data augmentation support

### Maze Dataset Builder

Preprocesses maze navigation datasets:

```bash
python scripts/build_maze_dataset.py \
    --source-repo sapientinc/maze-30x30-hard-1k \
    --output-dir data/maze-30x30-hard-1k \
    --aug true
```

**Features**:
- Dihedral transformations for augmentation
- Character-to-ID mapping
- Puzzle grouping for batch sampling

## Training & Evaluation

### Training Script

Full training loop with distributed training, Hydra configuration, and wandb logging:

```bash
# Single GPU training
python scripts/train.py --config-path configs --config-name train.yaml

# Multi-GPU distributed training
torchrun --nproc_per_node=4 scripts/train.py \
    --config-path configs \
    --config-name train.yaml
```

**Features**:
- Distributed training with `torch.distributed`
- Hydra configuration management
- Weights & Biases experiment tracking
- Gradient accumulation
- Learning rate scheduling
- Checkpoint saving/loading
- Sparse puzzle embedding optimization (SignSGD)

**Configuration**:
The training script uses Hydra for configuration. See example configs in the original research repository.

### Evaluation Script

Distributed evaluation with output saving:

```bash
python scripts/evaluate.py \
    --checkpoint path/to/checkpoint.pt \
    --save-outputs inputs labels logits q_halt_logits
```

**Features**:
- Distributed evaluation across GPUs
- Configurable output saving
- Metrics computation
- Inference mode optimization

## Dataset Loading

The `scripts/data/` module provides utilities for loading preprocessed datasets:

```python
from scripts.data import PuzzleDataset, PuzzleDatasetConfig

# Configure dataset
config = PuzzleDatasetConfig(
    dataset_path="data/arc-aug-1000/train",
    batch_size=32,
    epochs_per_iteration=1,
    test_mode=False,  # False for training (shuffled), True for evaluation
)

# Create dataset
dataset = PuzzleDataset(config)

# Use with PyTorch DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=None, num_workers=0)

for batch in loader:
    inputs = batch["inputs"]           # [batch_size, seq_len]
    labels = batch["labels"]           # [batch_size, seq_len]
    puzzle_ids = batch["puzzle_identifiers"]  # [batch_size]
    # ... train model ...
```

**Features**:
- Memory-mapped file I/O for large datasets
- Distributed training support (automatic data splitting)
- Efficient puzzle-based batching
- Automatic padding to sequence length
- Train/test mode (shuffled vs sequential)

## Using with the Package

These scripts depend on the `hierarchical_reasoning_model` package being installed:

```bash
# Install the package
uv sync

# Then use scripts
python scripts/build_arc_dataset.py --help
python scripts/train.py --help
```

## Integration Example

Here's how to integrate the scripts with the core package:

```python
# Use the package for the model
from hierarchical_reasoning_model import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config,
    ACTLossHead,
)

# Use scripts for data loading
from scripts.data import PuzzleDataset, PuzzleDatasetConfig

# Create model
config = HierarchicalReasoningModel_ACTV1Config(
    batch_size=32,
    seq_len=81,
    vocab_size=11,
    num_puzzle_identifiers=1,
    H_cycles=2,
    L_cycles=2,
    H_layers=4,
    L_layers=4,
    hidden_size=512,
    num_heads=8,
)
model = HierarchicalReasoningModel_ACTV1(config.model_dump())
loss_head = ACTLossHead(model, loss_type="softmax")

# Load dataset
dataset_config = PuzzleDatasetConfig(
    dataset_path="data/arc-aug-1000/train",
    batch_size=32,
)
dataset = PuzzleDataset(dataset_config)

# Train
import torch
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in dataset:
    optimizer.zero_grad()
    carry = loss_head.initial_carry(batch)
    _, loss, metrics, _, _ = loss_head.forward(
        return_keys=["logits"],
        carry=carry,
        batch=batch,
    )
    loss.backward()
    optimizer.step()
```

## Notes

- These scripts are provided as reference implementations from the research
- You can modify them for your own use cases
- The core `hierarchical_reasoning_model` package does not depend on these scripts
- For custom training loops, you can write your own data loading using the model package directly

## Dependencies

Additional dependencies needed for scripts (beyond core package):

- `hydra-core` - Configuration management for training
- `wandb` - Experiment tracking
- `adam-atan2` - Custom optimizer
- `coolname` - Run naming
- `argdantic` - CLI argument parsing for builders
- `huggingface-hub` - Dataset downloading
- `tqdm` - Progress bars

Install all dependencies:

```bash
uv sync
```
