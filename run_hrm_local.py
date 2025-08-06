"""
run_hrm_local.py

This script runs inference using the HRM model on a local, CPU-only machine.

Prerequisites:
1. Python 3.8+
2. The HRM repository cloned locally.
3. The required Python packages installed. You can do this from the HRM repository root:
   pip install -r requirements.txt
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Usage:
python run_hrm_local.py --repo-dir /path/to/hrm --checkpoint-dir /path/to/HRM-checkpoint-ARC-2 --data-dir /path/to/hrm/data/arc-2-aug-1000

NOTE: This script is for CPU-only inference. Performance will be significantly
slower than on a GPU. It assumes the model architecture does not contain custom
CUDA kernels that would prevent it from running on a CPU.
"""

import os
import sys
import yaml
import torch
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run HRM inference on a local CPU.")
    parser.add_argument("--repo-dir", required=True, help="Path to the cloned HRM repository.")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to the downloaded HRM checkpoint directory.")
    parser.add_argument("--data-dir", required=True, help="Path to the generated ARC dataset directory.")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of the sample to test from the dataset.")
    args = parser.parse_args()

    # --- Add repo to Python path ---
    if not os.path.isdir(args.repo_dir):
        print(f"Error: Repository directory not found at {args.repo_dir}")
        sys.exit(1)
    sys.path.append(os.path.abspath(args.repo_dir))

    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetMetadata

    # --- Configuration ---
    CONFIG_PATH = os.path.join(args.checkpoint_dir, "all_config.yaml")
    CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, "checkpoint")

    if not os.path.exists(CONFIG_PATH) or not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint or config not found in {args.checkpoint_dir}")
        sys.exit(1)

    # --- Load Model Configuration ---
    print("Loading model configuration...")
    with open(CONFIG_PATH, "r") as f:
        config_yaml = yaml.safe_load(f)

    model_config = config_yaml['arch']
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # The PuzzleDataset provides this metadata, which we need for model instantiation
    try:
        train_metadata = PuzzleDatasetMetadata.from_data_path(os.path.join(args.data_dir, "train"))
    except FileNotFoundError:
        print(f"Error: Could not find training metadata in {args.data_dir}/train. Is the data directory correct?")
        sys.exit(1)

    model_config['batch_size'] = 1
    model_config['seq_len'] = train_metadata.seq_len
    model_config['num_puzzle_identifiers'] = train_metadata.num_puzzle_identifiers
    model_config['vocab_size'] = train_metadata.vocab_size
    model_config['forward_dtype'] = "float32" # Force float32 for CPU

    # --- Instantiate and Load Model ---
    print("Instantiating the model...")
    model = HierarchicalReasoningModel_ACTV1(model_config)
    model.to(device)
    model.eval()

    print("Loading checkpoint...")
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    if list(state_dict.keys())[0].startswith('_orig_mod.'):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, assign=True)
    print("Model loaded successfully.")

    # --- Load Dataset and Sample ---
    print(f"Loading test data from {args.data_dir}/test")
    test_dataset_path = os.path.join(args.data_dir, "test")
    if not os.path.exists(test_dataset_path):
        print(f"Error: Test dataset not found at {test_dataset_path}")
        sys.exit(1)

    test_dataset = PuzzleDataset(test_dataset_path, test_set_mode=True)
    if args.sample_idx >= len(test_dataset):
        print(f"Error: Sample index {args.sample_idx} is out of bounds for dataset of size {len(test_dataset)}")
        sys.exit(1)

    sample = test_dataset[args.sample_idx]
    print(f"Selected sample index: {args.sample_idx}")

    # --- Prepare Input for Inference ---
    puzz_id = torch.tensor([sample['puzzle_identifier']], device=device)
    inputs = torch.tensor(sample['test_inputs'][0], device=device).unsqueeze(0)
    label_grid = sample['test_labels'][0]

    num_steps = model.config.halt_max_steps
    print(f"Running inference for {num_steps} steps...")

    # --- Run Inference ---
    with torch.no_grad():
        batch = {"inputs": inputs, "puzzle_identifiers": puzz_id}
        carry = model.initial_carry(batch)
        for step in range(num_steps):
            carry, outputs = model(carry, batch)
            if step == num_steps - 1:
                final_logits = outputs['logits']

    # --- Process and Display Output ---
    predicted_sequence = torch.argmax(final_logits, dim=-1).squeeze(0)
    grid_size = int(np.sqrt(inputs.shape[1]))
    predicted_grid = predicted_sequence.reshape(grid_size, grid_size).cpu().numpy()
    input_grid = inputs.squeeze(0).reshape(grid_size, grid_size).cpu().numpy()

    print("\n--- Inference Results ---")
    print("\nTest Input Grid:\n", input_grid)
    print("\nModel Prediction Grid:\n", predicted_grid)
    print("\nGround Truth Grid:\n", label_grid)

    is_correct = np.array_equal(predicted_grid, label_grid)
    print(f"\nPrediction Correct: {is_correct}\n")


if __name__ == "__main__":
    main()
