"""Train HRM on Sudoku-Extreme dataset according to paper specifications.

This example demonstrates how to train a Hierarchical Reasoning Model (HRM)
on the Sudoku-Extreme-1k dataset from HuggingFace, following the exact
specifications from the paper (arXiv:2506.21734v3).

Key features from the paper:
- Two-level hierarchical architecture (H and L modules)
- Adaptive Computation Time (ACT) with Q-learning
- Deep supervision for stable training
- Sparse puzzle embeddings with SignSGD optimizer
- One-step gradient approximation (no BPTT)

Dataset: sapientinc/sudoku-extreme-1k
Expected performance: Near-perfect accuracy (~99%+) after ~20k epochs
Training time: ~10 minutes on 8 GPUs (or ~1 hour on single GPU)

Requirements:
    - CUDA-capable GPU with FlashAttention installed
    - datasets library: pip install datasets
    - wandb (optional): for experiment tracking
"""

import os
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from dotenv import load_dotenv
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hierarchical_reasoning_model import (
    ACTLossHead,
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config,
)
from hierarchical_reasoning_model.core.embeddings import (
    CastedSparseEmbeddingSignSGD_Distributed,
)

# Optional wandb integration for experiment tracking
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SudokuDataset(Dataset):
    """Sudoku dataset wrapper for HuggingFace dataset.

    Args:
        split: Dataset split ('train' or 'test')
        cache_dir: Directory to cache the downloaded dataset
    """

    def __init__(self, split: str = "train", cache_dir: str | None = None):
        """Initialize Sudoku dataset from HuggingFace."""
        print(f"Loading sudoku-extreme-1k dataset (split={split})...")
        self.dataset = load_dataset(
            "sapientinc/sudoku-extreme-1k",
            split=split,
            cache_dir=cache_dir,
        )
        print(f"Loaded {len(self.dataset)} Sudoku puzzles")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single Sudoku puzzle.

        Args:
            idx: Index of the puzzle to retrieve

        Returns:
            Dictionary with keys:
                - inputs: Puzzle with blanks (81 values, 0-9)
                - labels: Complete solution (81 values, 1-9)
                - puzzle_identifiers: Unique puzzle ID
        """
        item = self.dataset[idx]

        # Convert to tensors
        # Puzzles are stored as strings of 81 characters
        # In the dataset, '.' represents blanks, which we convert to 0
        question = item["question"].replace(".", "0")
        inputs = torch.tensor([int(c) for c in question], dtype=torch.int32)
        labels = torch.tensor([int(c) for c in item["answer"]], dtype=torch.int32)
        puzzle_id = torch.tensor(idx, dtype=torch.int32)

        return {
            "inputs": inputs,
            "labels": labels,
            "puzzle_identifiers": puzzle_id,
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function for DataLoader.

    Args:
        batch: List of dictionaries from __getitem__

    Returns:
        Batched dictionary with stacked tensors
    """
    return {
        "inputs": torch.stack([item["inputs"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "puzzle_identifiers": torch.stack(
            [item["puzzle_identifiers"] for item in batch]
        ),
    }


def create_model_config(
    batch_size: int = 384,
    num_puzzles: int = 1000,
) -> HierarchicalReasoningModel_ACTV1Config:
    """Create HRM model configuration following paper specifications.

    Paper specifications for Sudoku-Extreme (27M parameters):
    - H_cycles=2, L_cycles=2 (hierarchical convergence)
    - H_layers=4, L_layers=4 (Transformer blocks each)
    - hidden_size=512, num_heads=8
    - RoPE positional encodings
    - halt_max_steps=16 for ACT
    - Stablemax loss for numerical stability

    Args:
        batch_size: Training batch size
        num_puzzles: Number of unique puzzles in dataset

    Returns:
        Model configuration object
    """
    return HierarchicalReasoningModel_ACTV1Config(
        # Batch and sequence configuration
        batch_size=batch_size,
        seq_len=81,  # 9x9 Sudoku grid
        vocab_size=11,  # 0-9 digits + special tokens
        num_puzzle_identifiers=num_puzzles,
        # Hierarchical processing (paper spec)
        H_cycles=2,  # High-level reasoning cycles
        L_cycles=2,  # Low-level computation cycles per H-cycle
        H_layers=4,  # High-level Transformer layers
        L_layers=4,  # Low-level Transformer layers
        # Model architecture (paper spec)
        hidden_size=512,
        num_heads=8,
        expansion=4.0,  # MLP expansion ratio
        # Positional encoding
        pos_encodings="rope",  # Rotary Position Embeddings
        # Adaptive Computation Time (ACT) with Q-learning
        halt_max_steps=16,
        halt_exploration_prob=0.1,  # ε-greedy exploration
        # Sparse puzzle embeddings
        puzzle_emb_ndim=512,
        use_sparse_puzzle_emb=True,
        # Use float32 for better stability and compatibility
        forward_dtype="float32",
    )


def setup_distributed():
    """Setup distributed training if available.

    Returns:
        Tuple of (rank, world_size, device)
    """
    if "RANK" in os.environ:
        # Multi-GPU distributed training
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        # Single GPU training
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return rank, world_size, device


def create_optimizers(
    model: torch.nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1.0,
    puzzle_emb_lr: float = 1e-4,
    puzzle_emb_weight_decay: float = 1.0,
    world_size: int = 1,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer | None]:
    """Create optimizers following paper specifications.

    Paper uses:
    - Adam-atan2 optimizer for main parameters
    - SignSGD for sparse puzzle embeddings
    - High weight decay (1.0) for regularization

    Args:
        model: The HRM model
        lr: Learning rate for main parameters
        weight_decay: Weight decay for main parameters
        puzzle_emb_lr: Learning rate for puzzle embeddings
        puzzle_emb_weight_decay: Weight decay for puzzle embeddings
        world_size: Number of distributed workers

    Returns:
        Tuple of (main_optimizer, puzzle_embedding_optimizer)
    """
    # Separate parameters for main model and puzzle embeddings
    main_params = []
    puzzle_emb_params = []

    for name, param in model.named_parameters():
        if "puzzle_emb" in name:
            puzzle_emb_params.append(param)
        else:
            main_params.append(param)

    # Main optimizer (Adam for most parameters)
    main_optimizer = Adam(
        main_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # Puzzle embedding optimizer (SignSGD for sparse updates)
    puzzle_emb_optimizer = None
    if puzzle_emb_params:
        # Get the sparse embedding module
        sparse_emb_module = None
        for module in model.modules():
            if hasattr(module, "local_weights"):
                sparse_emb_module = module
                break

        if sparse_emb_module is not None:
            puzzle_emb_optimizer = CastedSparseEmbeddingSignSGD_Distributed(
                [
                    sparse_emb_module.weights,
                    sparse_emb_module.local_weights,
                    sparse_emb_module.local_ids,
                ],
                world_size=world_size,
                lr=puzzle_emb_lr,
                weight_decay=puzzle_emb_weight_decay,
            )

    return main_optimizer, puzzle_emb_optimizer


def train_epoch(
    model: ACTLossHead,
    dataloader: DataLoader,
    main_optimizer: torch.optim.Optimizer,
    puzzle_emb_optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    epoch: int,
    rank: int = 0,
    wandb_run=None,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: HRM model with ACT loss head
        dataloader: Training data loader
        main_optimizer: Optimizer for main parameters
        puzzle_emb_optimizer: Optimizer for puzzle embeddings
        device: Device to train on
        epoch: Current epoch number
        rank: Distributed rank (0 for main process)
        wandb_run: Weights & Biases run object for logging (optional)

    Returns:
        Dictionary of training metrics
    """
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    total_exact_accuracy = 0.0
    num_batches = 0

    # Progress bar (only on rank 0)
    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}",
        disable=(rank != 0),
    )

    for batch in pbar:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Initialize carry state
        carry = model.initial_carry(batch)

        # Forward pass with deep supervision
        # The model uses one-step gradient approximation
        carry, loss, metrics, predictions, all_halted = model.forward(
            return_keys=["logits"],
            carry=carry,
            batch=batch,
        )

        # Backward pass
        main_optimizer.zero_grad()
        if puzzle_emb_optimizer is not None:
            puzzle_emb_optimizer.zero_grad()

        loss.backward()

        # Gradient clipping (optional, helps with stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        main_optimizer.step()
        if puzzle_emb_optimizer is not None:
            puzzle_emb_optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_accuracy += metrics["accuracy"].item()
        total_exact_accuracy += metrics["exact_accuracy"].item()
        num_batches += 1

        # Update progress bar
        if rank == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{metrics['accuracy'].item():.4f}",
                    "exact_acc": f"{metrics['exact_accuracy'].item():.4f}",
                }
            )

    metrics_dict = {
        "train/loss": total_loss / num_batches,
        "train/accuracy": total_accuracy / num_batches,
        "train/exact_accuracy": total_exact_accuracy / num_batches,
        "epoch": epoch,
    }

    # Log to wandb if available
    if wandb_run is not None and rank == 0:
        wandb_run.log(metrics_dict)

    return metrics_dict


@torch.no_grad()
def evaluate(
    model: ACTLossHead,
    dataloader: DataLoader,
    device: torch.device,
    rank: int = 0,
    wandb_run=None,
    epoch: int | None = None,
) -> dict[str, float]:
    """Evaluate model on validation/test set.

    Args:
        model: HRM model with ACT loss head
        dataloader: Evaluation data loader
        device: Device to evaluate on
        rank: Distributed rank (0 for main process)
        wandb_run: Weights & Biases run object for logging (optional)
        epoch: Current epoch number for logging (optional)

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_exact_accuracy = 0.0
    num_batches = 0

    pbar = tqdm(
        dataloader,
        desc="Evaluating",
        disable=(rank != 0),
    )

    for batch in pbar:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Initialize carry state
        carry = model.initial_carry(batch)

        # Forward pass (no gradients)
        carry, loss, metrics, predictions, all_halted = model.forward(
            return_keys=["logits"],
            carry=carry,
            batch=batch,
        )

        # Accumulate metrics
        total_loss += loss.item()
        total_accuracy += metrics["accuracy"].item()
        total_exact_accuracy += metrics["exact_accuracy"].item()
        num_batches += 1

    metrics_dict = {
        "eval/loss": total_loss / num_batches,
        "eval/accuracy": total_accuracy / num_batches,
        "eval/exact_accuracy": total_exact_accuracy / num_batches,
    }

    # Add epoch to metrics if provided
    if epoch is not None:
        metrics_dict["epoch"] = epoch

    # Log to wandb if available
    if wandb_run is not None and rank == 0:
        wandb_run.log(metrics_dict)

    return metrics_dict


def main():  # noqa: N806
    """Main training function."""
    # Load environment variables (for WANDB_API_KEY)
    load_dotenv()

    # Paper specifications for Sudoku-Extreme-1k
    # Note: Using uppercase for clarity as these are fixed paper hyperparameters
    BATCH_SIZE = 64  # Global batch size (reduced for GPU memory)  # noqa: N806
    EPOCHS = 20000  # Paper uses 20k epochs for 1k dataset  # noqa: N806
    EVAL_INTERVAL = 2000  # Evaluate every 2k epochs  # noqa: N806
    LR = 1e-4  # Learning rate (paper spec)  # noqa: N806
    WEIGHT_DECAY = 1.0  # High weight decay for regularization  # noqa: N806
    PUZZLE_EMB_LR = 1e-4  # Same as main LR  # noqa: N806
    PUZZLE_EMB_WD = 1.0  # Same weight decay  # noqa: N806

    print("=" * 70)
    print("HRM Training on Sudoku-Extreme-1k (Paper Specifications)")
    print("=" * 70)
    print("\nPaper: arXiv:2506.21734v3 - Hierarchical Reasoning Model")
    print("Dataset: sapientinc/sudoku-extreme-1k")
    print("Expected: ~99%+ exact accuracy after 20k epochs")
    print("=" * 70)

    # Setup distributed training
    rank, world_size, device = setup_distributed()
    if rank == 0:
        print(f"\n✓ Device: {device}")
        print(f"✓ World size: {world_size}")

    # Adjust batch size for distributed training
    local_batch_size = BATCH_SIZE // world_size
    if rank == 0:
        print(f"✓ Global batch size: {BATCH_SIZE}")
        print(f"✓ Local batch size: {local_batch_size}")

    # Load datasets
    if rank == 0:
        print("\n[1] Loading datasets...")

    train_dataset = SudokuDataset(split="train")
    test_dataset = SudokuDataset(split="test_hard")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=local_batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    if rank == 0:
        print(f"✓ Train samples: {len(train_dataset)}")
        print(f"✓ Test samples: {len(test_dataset)}")

    # Create model following paper specifications
    if rank == 0:
        print("\n[2] Creating HRM model (paper specifications)...")

    config = create_model_config(
        batch_size=local_batch_size,
        num_puzzles=len(train_dataset) + len(test_dataset),  # Total unique puzzles
    )

    # Initialize base model
    base_model = HierarchicalReasoningModel_ACTV1(config.model_dump())

    # Move base model to device first
    base_model = base_model.to(device)

    # Wrap with ACT loss head (stablemax for stability)
    model = ACTLossHead(base_model, loss_type="stablemax_cross_entropy")

    # Ensure entire model is on correct device
    model = model.to(device)

    # Wrap with DDP if multi-GPU
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
        )

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created: {num_params:,} parameters (~27M)")
        print(f"  - H_cycles: {config.H_cycles}, L_cycles: {config.L_cycles}")
        print(f"  - H_layers: {config.H_layers}, L_layers: {config.L_layers}")
        print(f"  - hidden_size: {config.hidden_size}")
        print(f"  - ACT max steps: {config.halt_max_steps}")

    # Create optimizers
    if rank == 0:
        print("\n[3] Creating optimizers...")

    main_optimizer, puzzle_emb_optimizer = create_optimizers(
        model,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        puzzle_emb_lr=PUZZLE_EMB_LR,
        puzzle_emb_weight_decay=PUZZLE_EMB_WD,
        world_size=world_size,
    )

    if rank == 0:
        print(f"✓ Main optimizer: Adam (lr={LR}, wd={WEIGHT_DECAY})")
        if puzzle_emb_optimizer:
            print(
                f"✓ Puzzle embedding optimizer: SignSGD "
                f"(lr={PUZZLE_EMB_LR}, wd={PUZZLE_EMB_WD})"
            )

    # Initialize Weights & Biases (only on rank 0)
    wandb_run = None
    if WANDB_AVAILABLE and rank == 0:
        wandb_run = wandb.init(
            entity="zbloss",
            project="hierarchical_reasoning_model",
            config={
                # Hyperparameters
                "batch_size": BATCH_SIZE,
                "local_batch_size": local_batch_size,
                "epochs": EPOCHS,
                "learning_rate": LR,
                "weight_decay": WEIGHT_DECAY,
                "puzzle_emb_lr": PUZZLE_EMB_LR,
                "puzzle_emb_wd": PUZZLE_EMB_WD,
                # Model architecture
                "H_cycles": config.H_cycles,
                "L_cycles": config.L_cycles,
                "H_layers": config.H_layers,
                "L_layers": config.L_layers,
                "hidden_size": config.hidden_size,
                "num_heads": config.num_heads,
                "expansion": config.expansion,
                "halt_max_steps": config.halt_max_steps,
                "halt_exploration_prob": config.halt_exploration_prob,
                "pos_encodings": config.pos_encodings,
                "vocab_size": config.vocab_size,
                "seq_len": config.seq_len,
                "num_puzzle_identifiers": config.num_puzzle_identifiers,
                "forward_dtype": config.forward_dtype,
                # Dataset info
                "dataset": "sapientinc/sudoku-extreme-1k",
                "train_samples": len(train_dataset),
                "test_samples": len(test_dataset),
                # Distributed training
                "world_size": world_size,
                # Model size
                "num_parameters": sum(p.numel() for p in model.parameters()),
            },
            name=f"hrm_sudoku_extreme_bs{BATCH_SIZE}_lr{LR}",
        )
        print("✓ Weights & Biases initialized")
    elif rank == 0 and not WANDB_AVAILABLE:
        print("⚠ Weights & Biases not available (install with: pip install wandb)")

    # Training loop
    if rank == 0:
        print(f"\n[4] Training for {EPOCHS} epochs...")
        print("=" * 70)

    best_exact_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Train one epoch
        train_metrics = train_epoch(
            model,
            train_loader,
            main_optimizer,
            puzzle_emb_optimizer,
            device,
            epoch,
            rank,
            wandb_run,
        )

        # Evaluate periodically
        if epoch % EVAL_INTERVAL == 0 or epoch == 1:
            eval_metrics = evaluate(model, test_loader, device, rank, wandb_run, epoch)

            if rank == 0:
                print(f"\nEpoch {epoch}/{EPOCHS}:")
                print(f"  Train Loss: {train_metrics['train/loss']:.4f}")
                print(f"  Train Accuracy: {train_metrics['train/accuracy']:.4f}")
                print(
                    f"  Train Exact Accuracy: "
                    f"{train_metrics['train/exact_accuracy']:.4f}"
                )
                print(f"  Eval Loss: {eval_metrics['eval/loss']:.4f}")
                print(f"  Eval Accuracy: {eval_metrics['eval/accuracy']:.4f}")
                print(
                    f"  Eval Exact Accuracy: {eval_metrics['eval/exact_accuracy']:.4f}"
                )

                # Save best model
                if eval_metrics["eval/exact_accuracy"] > best_exact_accuracy:
                    best_exact_accuracy = eval_metrics["eval/exact_accuracy"]
                    print(f"  ✓ New best exact accuracy: {best_exact_accuracy:.4f}")

                    # Log best accuracy to wandb
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "best_eval/exact_accuracy": best_exact_accuracy,
                                "best_eval/epoch": epoch,
                            }
                        )

                    # Save checkpoint
                    checkpoint_dir = Path("checkpoints")
                    checkpoint_dir.mkdir(exist_ok=True)
                    checkpoint_path = checkpoint_dir / "best_model.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": main_optimizer.state_dict(),
                            "metrics": eval_metrics,
                        },
                        checkpoint_path,
                    )

                    # Log checkpoint to wandb
                    if wandb_run is not None:
                        wandb_run.save(str(checkpoint_path))

    if rank == 0:
        print("\n" + "=" * 70)
        print("Training completed!")
        print(f"Best exact accuracy: {best_exact_accuracy:.4f}")
        print("=" * 70)

    # Finish wandb run
    if wandb_run is not None and rank == 0:
        wandb_run.finish()
        print("✓ Weights & Biases run finished")

    # Cleanup distributed training
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
