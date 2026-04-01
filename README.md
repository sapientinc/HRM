# Hierarchical Reasoning Model

![](./assets/hrm.png)

Reasoning, the process of devising and executing complex goal-oriented action sequences, remains a critical challenge in AI.
Current large language models (LLMs) primarily employ Chain-of-Thought (CoT) techniques, which suffer from brittle task decomposition, extensive data requirements, and high latency. Inspired by the hierarchical and multi-timescale processing in the human brain, we propose the Hierarchical Reasoning Model (HRM), a novel recurrent architecture that attains significant computational depth while maintaining both training stability and efficiency.
HRM executes sequential reasoning tasks in a single forward pass without explicit supervision of the intermediate process, through two interdependent recurrent modules: a high-level module responsible for slow, abstract planning, and a low-level module handling rapid, detailed computations. With only 27 million parameters, HRM achieves exceptional performance on complex reasoning tasks using only 1000 training samples. The model operates without pre-training or CoT data, yet achieves nearly perfect performance on challenging tasks including complex Sudoku puzzles and optimal path finding in large mazes.
Furthermore, HRM outperforms much larger models with significantly longer context windows on the Abstraction and Reasoning Corpus (ARC), a key benchmark for measuring artificial general intelligence capabilities.
These results underscore HRM’s potential as a transformative advancement toward universal computation and general-purpose reasoning systems.

Read Our Paper: [https://arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734)

**Join Our Discord Community: [https://discord.gg/sapient](https://discord.gg/sapient)**

## Package Structure 📁

This repository contains:

- **`hierarchical_reasoning_model/`** - Core model package (install with `uv sync`)
  - Pure model architecture components
  - No training/evaluation utilities
  - Minimal dependencies

- **`scripts/`** - Research scripts (not part of package)
  - Dataset builders for ARC, Sudoku, Maze
  - Training and evaluation scripts
  - Data loading utilities
  - See `scripts/README.md` for details

## Installation 📦

### Core Model Package (Recommended for Users)

Install just the model architecture:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/HRM.git
cd HRM

# Install core package
uv sync
```

### With Research Scripts

For dataset processing, training, and evaluation:

```bash
# Install all dependencies
uv sync --all-extras

# Initialize submodules (for raw datasets)
git submodule update --init --recursive
```

### Using pip

```bash
# Clone and install
git clone https://github.com/yourusername/HRM.git
cd HRM
pip install -e .
```

### Docker (GPU Support)

For a containerized environment with all dependencies:

```bash
# Build for FlashAttention 2 (Ampere GPUs: RTX 30xx, A100, etc.)
docker-compose build hrm-train-fa2

# Or build for FlashAttention 3 (Hopper GPUs: H100, etc.)
docker-compose build hrm-train-fa3

# Run training
docker-compose run hrm-train-fa2
```

### FlashAttention Installation

HRM requires FlashAttention for efficient attention computation. The package attempts to install it automatically, but for manual installation:

**For Hopper GPUs (H100, etc.) - FlashAttention 3:**
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

**For Ampere/Ada GPUs (RTX 30xx/40xx, A100, etc.) - FlashAttention 2:**
```bash
pip install flash-attn --no-build-isolation
```

### Verify Installation

```bash
# Run basic usage example
uv run python examples/01_basic_model_usage.py
```

## Python API Usage 🐍

The core package provides the HRM model architecture for easy integration:

```python
from hierarchical_reasoning_model import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config,
    ACTLossHead,
)
import torch

# Create model configuration
config = HierarchicalReasoningModel_ACTV1Config(
    batch_size=4,
    seq_len=81,  # Sudoku grid size
    vocab_size=11,  # 0-9 digits + padding
    num_puzzle_identifiers=1,
    H_cycles=2,  # High-level reasoning cycles
    L_cycles=2,  # Low-level computation cycles
    H_layers=4,
    L_layers=4,
    hidden_size=512,
    num_heads=8,
    expansion=4.0,
    pos_encodings="rope",
    halt_max_steps=16,
    halt_exploration_prob=0.1,
    puzzle_emb_ndim=512,
)

# Initialize model
model = HierarchicalReasoningModel_ACTV1(config.model_dump())
loss_head = ACTLossHead(model, loss_type="stablemax_cross_entropy")

# Prepare batch (you provide your own data loading)
batch = {
    "inputs": torch.randint(0, 11, (4, 81)),
    "labels": torch.randint(0, 11, (4, 81)),
    "puzzle_identifiers": torch.zeros(4, dtype=torch.long),
}

# Forward pass
carry = loss_head.initial_carry(batch)
carry, loss, metrics, predictions, all_halted = loss_head.forward(
    return_keys=["logits"],
    carry=carry,
    batch=batch,
)

# Training loop (your own optimizer and data)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in your_dataloader:  # Bring your own data loader
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

### Using Research Scripts

For complete training pipelines with data loading, see `scripts/README.md`:

```bash
# Use dataset builders
python scripts/build_sudoku_dataset.py --output-dir data/sudoku

# Use training script
python scripts/train.py --config-path configs --config-name train.yaml
```

See `examples/01_basic_model_usage.py` for a complete working example.

## Quick Start Guide 🚀

### Prerequisites ⚙️

Ensure PyTorch and CUDA are installed. The repo needs CUDA extensions to be built. If not present, run the following commands:

```bash
# Install CUDA 12.6
CUDA_URL=https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run

wget -q --show-progress --progress=bar:force:noscroll -O cuda_installer.run $CUDA_URL
sudo sh cuda_installer.run --silent --toolkit --override

export CUDA_HOME=/usr/local/cuda-12.6

# Install PyTorch with CUDA 12.6
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

pip3 install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL

# Additional packages for building extensions
pip3 install packaging ninja wheel setuptools setuptools-scm
```

Then install FlashAttention. For Hopper GPUs, install FlashAttention 3

```bash
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

For Ampere or earlier GPUs, install FlashAttention 2

```bash
pip3 install flash-attn
```

## Install Python Dependencies 🐍

```bash
pip install -r requirements.txt
```

## W&B Integration 📈

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking and metric visualization. Ensure you're logged in:

```bash
wandb login
```

## Run Experiments

### Quick Demo: Sudoku Solver 💻🗲

Train a master-level Sudoku AI capable of solving extremely difficult puzzles on a modern laptop GPU. 🧩

```bash
# Download and build Sudoku dataset (using research scripts)
uv run python scripts/build_sudoku_dataset.py \
    --output-dir data/sudoku-extreme-1k-aug-1000 \
    --subsample-size 1000 \
    --num-aug 1000

# Start training (single GPU, smaller batch size)
OMP_NUM_THREADS=8 uv run python scripts/train.py \
    data_path=data/sudoku-extreme-1k-aug-1000 \
    epochs=20000 \
    eval_interval=2000 \
    global_batch_size=384 \
    lr=7e-5 \
    puzzle_emb_lr=7e-5 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0
```

Runtime: ~10 hours on a RTX 4070 laptop GPU

## Trained Checkpoints 🚧

 - [ARC-AGI-2](https://huggingface.co/sapientinc/HRM-checkpoint-ARC-2)
 - [Sudoku 9x9 Extreme (1000 examples)](https://huggingface.co/sapientinc/HRM-checkpoint-sudoku-extreme)
 - [Maze 30x30 Hard (1000 examples)](https://huggingface.co/sapientinc/HRM-checkpoint-maze-30x30-hard)

To use the checkpoints, see Evaluation section below.

## Full-scale Experiments 🔵

Experiments below assume an 8-GPU setup.

### Dataset Preparation

```bash
# Initialize submodules (for raw datasets)
git submodule update --init --recursive

# ARC-1 (using scripts/)
python scripts/build_arc_dataset.py \
    --output-dir data/arc-aug-1000 \
    --num-aug 1000

# ARC-2
python scripts/build_arc_dataset.py \
    --dataset-dirs dataset/raw-data/ARC-AGI-2/data \
    --output-dir data/arc-2-aug-1000

# Sudoku-Extreme (full version)
python scripts/build_sudoku_dataset.py \
    --output-dir data/sudoku-extreme-full

# Sudoku-Extreme (1000 examples)
python scripts/build_sudoku_dataset.py \
    --output-dir data/sudoku-extreme-1k-aug-1000 \
    --subsample-size 1000 \
    --num-aug 1000

# Maze (1000 examples)
python scripts/build_maze_dataset.py \
    --output-dir data/maze-30x30-hard-1k
```

### Dataset Visualization

Explore the puzzles visually:

* Open `puzzle_visualizer.html` in your browser.
* Upload the generated dataset folder located in `data/...`.

## Launch experiments

### Small-sample (1K)

ARC-1:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 scripts/train.py
```

*Runtime:* ~24 hours

ARC-2:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 scripts/train.py \
    data_path=data/arc-2-aug-1000
```

*Runtime:* ~24 hours (checkpoint after 8 hours is often sufficient)

Sudoku Extreme (1k):

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 scripts/train.py \
    data_path=data/sudoku-extreme-1k-aug-1000 \
    epochs=20000 \
    eval_interval=2000 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0
```

*Runtime:* ~10 minutes

Maze 30x30 Hard (1k):

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 scripts/train.py \
    data_path=data/maze-30x30-hard-1k \
    epochs=20000 \
    eval_interval=2000 \
    lr=1e-4 \
    puzzle_emb_lr=1e-4 \
    weight_decay=1.0 \
    puzzle_emb_weight_decay=1.0
```

*Runtime:* ~1 hour

### Full Sudoku-Hard

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 scripts/train.py \
    data_path=data/sudoku-hard-full \
    epochs=100 \
    eval_interval=10 \
    lr_min_ratio=0.1 \
    global_batch_size=2304 \
    lr=3e-4 \
    puzzle_emb_lr=3e-4 \
    weight_decay=0.1 \
    puzzle_emb_weight_decay=0.1 \
    arch.loss.loss_type=softmax_cross_entropy \
    arch.L_cycles=8 \
    arch.halt_max_steps=8 \
    arch.pos_encodings=learned
```

*Runtime:* ~2 hours

## Evaluation

Evaluate your trained models:

* Check `eval/exact_accuracy` in W&B.
* For ARC-AGI, follow these additional steps:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 scripts/evaluate.py \
    checkpoint=<CHECKPOINT_PATH>
```

* Then use the provided `arc_eval.ipynb` notebook to finalize and inspect your results.

## Notes

 - Small-sample learning typically exhibits accuracy variance of around ±2 points.
 - For Sudoku-Extreme (1,000-example dataset), late-stage overfitting may cause numerical instability during training and Q-learning. It is advisable to use early stopping once the training accuracy approaches 100%.

## Citation 📜

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model},
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734},
}
```
