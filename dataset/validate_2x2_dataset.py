"""
Validates 2x2x2 cube dataset: checks data format, solution correctness,
optimality, vocabulary ranges, and statistics.

Usage: python validate_2x2_dataset.py --dataset_dir data/cube-2-by-2-solution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
import py222


cli = ArgParser()


class ValidationConfig(BaseModel):
    dataset_dir: str = "data/cube-2-by-2-solution"
    num_samples_to_check: int = None  # None = check all samples
    check_optimality: bool = True  # Verify solutions are optimal
    verbose: bool = False


MOVE_NAMES = ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2"]


def load_split(dataset_dir: Path, split_name: str):
    split_dir = dataset_dir / split_name
    if not split_dir.exists():
        return None
    
    with open(split_dir / "dataset.json", "r") as f:
        metadata = json.load(f)
    
    data = {}
    for key in ["inputs", "labels", "group_indices", "puzzle_indices", "puzzle_identifiers"]:
        filepath = split_dir / f"all__{key}.npy"
        if filepath.exists():
            data[key] = np.load(filepath)
    
    return metadata, data


def format_solution(moves):
    """Format a solution sequence for display."""
    if len(moves) == 0:
        return "SOLVED"
    return " ".join([MOVE_NAMES[m] for m in moves])


def validate_split(dataset_dir: Path, split_name: str, config: ValidationConfig):
    print(f"\n{'='*50}")
    print(f"Validating {split_name} split")
    print(f"{'='*50}")
    
    result = load_split(dataset_dir, split_name)
    if result is None:
        print(f"  Error: Split not found: {split_name}")
        return False
    
    metadata, data = result
    
    print("\n[1] Checking data format...")
    inputs = data["inputs"]
    labels = data["labels"]
    num_samples = len(inputs)
    print(f"  Samples: {num_samples}")
    
    expected_input_shape = (num_samples, 24)
    if inputs.shape != expected_input_shape:
        print(f"  Error: Input shape {inputs.shape} != {expected_input_shape}")
        return False
    print(f"  Input shape: {inputs.shape}")
    
    seq_len = metadata["seq_len"]
    expected_label_shape = (num_samples, seq_len)
    if labels.shape != expected_label_shape:
        print(f"  Error: Label shape {labels.shape} != {expected_label_shape}")
        return False
    print(f"  Label shape: {labels.shape}")
    
    vocab_size = metadata["vocab_size"]
    
    # Inputs: colors 0-5 shifted by +1, so should be in [1, 6]
    input_min, input_max = inputs.min(), inputs.max()
    if input_min < 1 or input_max > 6:
        print(f"  Error: Input range [{input_min}, {input_max}], expected [1, 6]")
        return False
    print(f"  Input range: [{input_min}, {input_max}]")
    
    # Labels: 0=PAD, 1-9=moves
    label_min, label_max = labels.min(), labels.max()
    if label_min < 0 or label_max >= vocab_size:
        print(f"  Error: Label range [{label_min}, {label_max}], expected [0, {vocab_size-1}]")
        return False
    print(f"  Label range: [{label_min}, {label_max}]")
    
    print("\n[2] Checking solution validity...")
    num_to_check = config.num_samples_to_check if config.num_samples_to_check else num_samples
    num_to_check = min(num_to_check, num_samples)
    indices_to_check = np.random.choice(num_samples, size=num_to_check, replace=False)
    
    invalid_solutions = []
    solution_lengths = []
    
    for idx in tqdm(indices_to_check, desc="  Verifying solutions"):
        state = inputs[idx] - 1  # Convert to 0-5 colors
        solution_encoded = labels[idx]
        solution = [m - 1 for m in solution_encoded if m > 0]  # Decode moves
        solution_lengths.append(len(solution))
        
        test_state = state.copy()
        for move in solution:
            if move < 0 or move > 8:
                print(f"  Error: Sample {idx} has invalid move {move}")
                invalid_solutions.append(idx)
                break
            test_state = py222.doMove(test_state, move)
        else:
            if not py222.isSolved(test_state):
                if config.verbose:
                    print(f"  Error: Sample {idx} solution doesn't solve cube")
                    print(f"    State: {state}")
                    print(f"    Solution: {format_solution(solution)}")
                invalid_solutions.append(idx)
    
    if invalid_solutions:
        print(f"  Error: {len(invalid_solutions)} invalid solutions found")
        if not config.verbose:
            print(f"    Indices: {invalid_solutions[:10]}{'...' if len(invalid_solutions) > 10 else ''}")
        return False
    else:
        print(f"  All {num_to_check} solutions valid")
    
    if config.check_optimality:
        print("\n[3] Checking optimality (may be slow)...")
        num_optimality_checks = min(100, num_to_check)
        optimality_indices = np.random.choice(indices_to_check, size=num_optimality_checks, replace=False)
        
        non_optimal = []
        
        for idx in tqdm(optimality_indices, desc="  Checking optimality"):
            state = inputs[idx] - 1
            solution_encoded = labels[idx]
            solution = [m - 1 for m in solution_encoded if m > 0]
            optimal_solution = py222.solve(state)
            
            if len(solution) != len(optimal_solution):
                if config.verbose:
                    print(f"  Sample {idx}: length {len(solution)} != optimal {len(optimal_solution)}")
                    print(f"    Dataset: {format_solution(solution)}")
                    print(f"    Optimal: {format_solution(optimal_solution)}")
                non_optimal.append(idx)
        
        if non_optimal:
            print(f"  Note: {len(non_optimal)} non-optimal (may have multiple optimal paths)")
        else:
            print(f"  All {num_optimality_checks} solutions optimal")
    
    print("\n[4] Statistics...")
    if not solution_lengths:
        solution_lengths = [np.count_nonzero(label) for label in labels]
    
    print(f"  Solution lengths:")
    print(f"    Min: {min(solution_lengths)}")
    print(f"    Max: {max(solution_lengths)}")
    print(f"    Mean: {np.mean(solution_lengths):.2f}")
    print(f"    Median: {np.median(solution_lengths):.1f}")
    print(f"    Std: {np.std(solution_lengths):.2f}")
    
    if max(solution_lengths) > seq_len:
        print(f"  Error: Max solution {max(solution_lengths)} exceeds seq_len {seq_len}")
        return False
    
    print("\n  Length distribution:")
    for length in range(min(solution_lengths), max(solution_lengths) + 1):
        count = sum(1 for l in solution_lengths if l == length)
        if count > 0:
            pct = 100 * count / len(solution_lengths)
            bar = "#" * int(pct / 2)
            print(f"    {length:2d} moves: {count:5d} ({pct:5.2f}%) {bar}")
    
    print(f"\n[5] Metadata...")
    if metadata["total_groups"] != num_samples:
        print(f"  Warning: total_groups {metadata['total_groups']} != samples {num_samples}")
    else:
        print(f"  total_groups matches sample count")
    
    if metadata["vocab_size"] != 10:
        print(f"  Warning: vocab_size {metadata['vocab_size']}, expected 10")
    else:
        print(f"  vocab_size correct")
    
    print(f"\n{'='*50}")
    print(f"{split_name} validation PASSED")
    print(f"{'='*50}")
    
    return True


@cli.command(singleton=True)
def validate(config: ValidationConfig):
    dataset_dir = Path(config.dataset_dir)
    if not dataset_dir.exists():
        print(f"Error: Dataset not found: {dataset_dir}")
        return
    
    print(f"Validating: {dataset_dir}")
    print(f"  Samples to check: {config.num_samples_to_check or 'ALL'}")
    print(f"  Optimality check: {config.check_optimality}")
    
    splits = ["train", "test", "val"]
    results = {}
    for split in splits:
        results[split] = validate_split(dataset_dir, split, config)
    
    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    for split, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {split:10s}: {status}")
    
    if all(results.values()):
        print("\nAll validations passed. Dataset ready.")
    else:
        print("\nSome validations failed. Check errors above.")


if __name__ == "__main__":
    cli()
