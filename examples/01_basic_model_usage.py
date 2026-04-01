"""Basic usage example for Hierarchical Reasoning Model.

This example demonstrates how to:
1. Create an HRM model configuration
2. Initialize the model
3. Prepare input data
4. Run forward pass
5. Compute losses

Note: This example requires a GPU with CUDA and FlashAttention installed.
"""

import torch

from hierarchical_reasoning_model import (
    ACTLossHead,
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config,
)


def create_model():
    """Create a small HRM model for demonstration."""
    config = HierarchicalReasoningModel_ACTV1Config(
        # Batch and sequence configuration
        batch_size=4,
        seq_len=81,  # Sudoku grid size (9x9)
        vocab_size=11,  # 0-9 digits + padding
        num_puzzle_identifiers=1,
        # Hierarchical processing configuration
        H_cycles=2,  # High-level reasoning cycles
        L_cycles=2,  # Low-level computation cycles
        H_layers=4,  # High-level transformer layers
        L_layers=4,  # Low-level transformer layers
        # Model architecture
        hidden_size=512,
        num_heads=8,
        expansion=4.0,
        # Positional encoding
        pos_encodings="rope",  # RoPE positional embeddings
        # Adaptive Computation Time (ACT) configuration
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        # Puzzle embeddings
        puzzle_emb_ndim=512,
    )

    # Create base model
    model = HierarchicalReasoningModel_ACTV1(config.model_dump())

    # Wrap with ACT loss head
    model_with_loss = ACTLossHead(model, loss_type="stablemax_cross_entropy")

    return model_with_loss


def create_sample_batch():
    """Create a sample batch of Sudoku puzzles.

    Returns:
        Dictionary with keys:
            - inputs: Tensor of shape (batch, seq_len) with puzzle inputs
            - labels: Tensor of shape (batch, seq_len) with puzzle solutions
            - puzzle_identifiers: Tensor of shape (batch,) with puzzle IDs
    """
    batch_size = 4
    seq_len = 81

    # Create random Sudoku-like data (values 0-9, with 0 as blank)
    # In practice, these would be real Sudoku puzzles
    inputs = torch.randint(0, 10, (batch_size, seq_len), dtype=torch.int32)
    labels = torch.randint(1, 10, (batch_size, seq_len), dtype=torch.int32)
    puzzle_identifiers = torch.zeros(batch_size, dtype=torch.int32)

    return {
        "inputs": inputs,
        "labels": labels,
        "puzzle_identifiers": puzzle_identifiers,
    }


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("Hierarchical Reasoning Model - Basic Usage Example")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA is not available. This example requires a GPU.")
        print("The model is designed to run on CUDA devices.")
        return

    print(f"\n✓ CUDA is available: {torch.cuda.get_device_name(0)}")

    # Create model
    print("\n[1] Creating HRM model...")
    try:
        model = create_model()
        model = model.cuda()
        num_params = sum(p.numel() for p in model.parameters())
        print("✓ Model created successfully")
        print(f"  - Total parameters: {num_params:,}")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        print("\nNote: This example requires FlashAttention to be installed.")
        print("For Ampere GPUs: pip install flash-attn")
        print("For Hopper GPUs: install FlashAttention 3 from source")
        return

    # Create sample data
    print("\n[2] Creating sample batch...")
    batch = create_sample_batch()
    batch = {k: v.cuda() for k, v in batch.items()}
    print("✓ Batch created")
    print(f"  - Batch size: {batch['inputs'].shape[0]}")
    print(f"  - Sequence length: {batch['inputs'].shape[1]}")

    # Initialize carry state
    print("\n[3] Initializing model carry state...")
    carry = model.initial_carry(batch)
    print("✓ Carry state initialized")

    # Forward pass
    print("\n[4] Running forward pass...")
    try:
        carry, loss, metrics, predictions, all_halted = model.forward(
            return_keys=["logits"],
            carry=carry,
            batch=batch,
        )
        print("✓ Forward pass completed")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - All sequences halted: {all_halted.item()}")
        print("\nMetrics:")
        for key, value in metrics.items():
            if key != "count":
                print(f"  - {key}: {value.item():.4f}")

    except Exception as e:
        print(f"✗ Error during forward pass: {e}")
        return

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
