# HRM Examples

This directory contains usage examples for the Hierarchical Reasoning Model package.

## Prerequisites

- Python >=3.10
- CUDA-capable GPU
- FlashAttention installed (see main README for installation instructions)

## Examples

### 01_basic_model_usage.py

Demonstrates basic model usage including:
- Creating an HRM model configuration
- Initializing the model
- Preparing input data
- Running forward pass
- Computing losses

**Run:**
```bash
uv run python examples/01_basic_model_usage.py
```

### 02_train_sudoku_extreme.py

Complete training example following the paper specifications (arXiv:2506.21734v3):
- Trains HRM on `sapientinc/sudoku-extreme-1k` from HuggingFace
- Uses paper-specified hyperparameters (27M params, H_cycles=2, L_cycles=2)
- Implements Adaptive Computation Time (ACT) with Q-learning
- Uses deep supervision and one-step gradient approximation
- Supports both single-GPU and multi-GPU distributed training
- Expected performance: ~99%+ exact accuracy after 20k epochs

**Features:**
- Automatic dataset download from HuggingFace
- SignSGD optimizer for sparse puzzle embeddings
- Periodic evaluation and checkpoint saving
- Progress tracking with tqdm
- Weights & Biases integration for experiment tracking (optional)

**Setup (for Weights & Biases tracking):**
```bash
# Create .env file with your wandb API key
echo "WANDB_API_KEY=your_api_key_here" > .env

# Or login via wandb CLI
wandb login
```

**Run (single GPU):**
```bash
uv run python examples/02_train_sudoku_extreme.py
```

**Run (multi-GPU distributed):**
```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 examples/02_train_sudoku_extreme.py
```

**Note:** The script will automatically log metrics to Weights & Biases if wandb is installed and configured. If not installed, training will proceed without experiment tracking.

**Expected runtime:**
- Single GPU: ~1-2 hours (depending on hardware)
- 8 GPUs: ~10 minutes (as reported in paper)

## Coming Soon

Additional examples will be added for:
- Loading and using pretrained checkpoints
- Building custom datasets
- Evaluating model performance on ARC-AGI
- Inference-time scaling demonstrations

## Note on Dependencies

These examples assume you have installed the full package with training dependencies:

```bash
uv sync --extra train --extra data
```

For GPU support with FlashAttention, follow the installation instructions in the main README.
