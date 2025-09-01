#!/usr/bin/env python3
"""
Script to train HRM on CIFAR dataset.

Usage:
    python train_cifar.py

This script will:
1. Build the CIFAR dataset in HRM format
2. Train the vision HRM model
3. Evaluate on test set
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error in {description}:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    else:
        print(f"Success: {description}")
        if result.stdout:
            print(f"Output: {result.stdout}")


def main():
    parser = argparse.ArgumentParser(description="Train HRM on CIFAR dataset")
    parser.add_argument("--dataset", choices=["CIFAR10", "CIFAR100"], default="CIFAR10",
                       help="CIFAR dataset to use")
    parser.add_argument("--skip-dataset-build", action="store_true",
                       help="Skip dataset building (use existing dataset)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Global batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    # Change to HRM directory
    hrm_dir = Path(__file__).parent
    os.chdir(hrm_dir)
    
    print(f"Working directory: {os.getcwd()}")
    print(f"Training HRM on {args.dataset}")
    
    # Step 1: Build CIFAR dataset
    if not args.skip_dataset_build:
        dataset_cmd = f"python dataset/build_cifar_dataset.py --dataset_name {args.dataset}"
        run_command(dataset_cmd, f"Building {args.dataset} dataset")
    else:
        print("Skipping dataset building...")
    
    # Step 2: Update config for the selected dataset
    config_file = "config/cfg_vision_pretrain.yaml"
    arch_config_file = "config/arch/hrm_vision_v1.yaml"
    
    # Update number of classes in arch config
    num_classes = 10 if args.dataset == "CIFAR10" else 100
    with open(arch_config_file, 'r') as f:
        arch_content = f.read()
    
    # Replace num_classes
    arch_content = arch_content.replace("num_classes: 10", f"num_classes: {num_classes}")
    
    with open(arch_config_file, 'w') as f:
        f.write(arch_content)
    
    print(f"Updated {arch_config_file} with num_classes: {num_classes}")
    
    # Step 3: Train the model
    train_cmd = f"python pretrain_vision.py --config-name cfg_vision_pretrain epochs={args.epochs} global_batch_size={args.batch_size} lr={args.lr}"
    run_command(train_cmd, f"Training HRM on {args.dataset}")
    
    print(f"\n{'='*60}")
    print("Training completed successfully!")
    print(f"Check the checkpoints directory for saved models.")
    print(f"Check wandb for training logs and metrics.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
