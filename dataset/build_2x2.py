import os
import sys
import json
import numpy as np
from typing import Optional
from multiprocessing import Pool, cpu_count

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

# Add parent directory so we can import py222
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import py222
from py222.solver import solveCube

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str = "data/cube-2-by-2-all-solutions"
    num_scrambles: int = 50000
    scramble_length: int = 20
    seed: int = 42
    
    # Split ratios for train/test/val
    train_split: float = 0.8
    test_split: float = 0.1
    val_split: float = 0.1
    num_workers: int = 0


def build_augmentation_tables():
    base_rotations = [
        [],       #           (U up)
        [18],     # x         (F up)
        [20],     # x2        (D up)
        [19],     # x'        (B up)
        [24],     # z         (L up)
        [25],     # z'        (R up)
    ]
    y_rotations = [
        [],
        [21],     # y
        [23],     # y2
        [22],     # y'
    ]

    inv_map = {18: 19, 19: 18, 20: 20,
               21: 22, 22: 21, 23: 23,
               24: 25, 25: 24}

    # Precompute state remappings for each orientaion
    test_state = np.arange(24)
    face_move_fp = {}
    for m in range(18):
        face_move_fp[tuple(py222.doMove(test_state, m).tolist())] = m

    ori_perms = np.zeros((24, 24), dtype=np.int32)
    remap_arrays = np.zeros((24, 9), dtype=np.int32)

    idx = 0
    for base in base_rotations:
        for y_rot in y_rotations:
            rot = base + y_rot
            inv_rot = [inv_map[m] for m in reversed(rot)]

            # Precompute combined sticker permutation for this orientation
            perm = np.arange(24)
            for m in rot:
                perm = perm[py222.moveDefs[m]]
            ori_perms[idx] = perm

            # Precompute move remppings
            for solver_move in range(9):
                s = test_state.copy()
                for rm in inv_rot:
                    s = py222.doMove(s, rm)
                s = py222.doMove(s, solver_move)
                for rm in rot:
                    s = py222.doMove(s, rm)
                remap_arrays[idx, solver_move] = face_move_fp[tuple(s.tolist())]

            idx += 1

    return ori_perms, remap_arrays


def generate_dataset(config: DataProcessConfig):
    np.random.seed(config.seed)

    # precompute tables for augmentation
    ori_perms, remap_arrays = build_augmentation_tables()

    # generate unique scrambled states
    scrambles = []
    seen_base = set() # only unique states
    pbar = tqdm(total=config.num_scrambles, desc="Generating scrambles")
    while len(scrambles) < config.num_scrambles:
        s = py222.scramble(config.scramble_length)
        s_norm = py222.normFC(s)
        key = tuple(s_norm.tolist())
        if key not in seen_base:
            seen_base.add(key)
            scrambles.append(s)
            pbar.update(1)
    pbar.close()

    # Solve all base states in parallel
    num_workers = config.num_workers if config.num_workers > 0 else max(1, cpu_count() - 1)
    chunk = max(1, len(scrambles) // (num_workers * 4))
    with Pool(num_workers) as pool:
        all_base_solutions = list(tqdm(
            pool.imap(solveCube, scrambles, chunksize=chunk),
            total=len(scrambles), desc="Solving"))

    #Build orientations using remapping table
    inputs = []
    labels = []
    puzzle_indices = [0]
    group_indices = [0]
    puzzle_identifiers = []
    seen_states = set(seen_base)

    puzzle_count = 0
    group_id = 0

    for i in tqdm(range(len(scrambles)), desc="Orientations"):
        s = scrambles[i]
        base_solutions = all_base_solutions[i]
        if not base_solutions:
            continue

        group_puzzle_count = 0
        for ori_idx in range(24):
            # Orientation with precomputed tables
            oriented_s = s[ori_perms[ori_idx]]
            s_norm = py222.normFC(oriented_s)
            state_key = tuple(s_norm.tolist())

            if state_key in seen_states:
                continue
            seen_states.add(state_key)

            # Transform base solutions
            moves = remap_arrays[ori_idx]
            transformed_solutions = [
                [int(moves[m]) for m in sol] for sol in base_solutions
            ]

            state_encoded = s_norm + 1  # values 1-6
            inputs.append(state_encoded)
            labels.append(transformed_solutions)
            puzzle_identifiers.append(0)

            puzzle_count += 1
            puzzle_indices.append(puzzle_count)
            group_puzzle_count += 1

        if group_puzzle_count > 0:
            group_indices.append(puzzle_count)
            group_id += 1

    total_solutions = sum(len(sols) for sols in labels)
    print(f"Generated {group_id} cubes augmented to {puzzle_count} cubes")
    print(f"Total solutions: {total_solutions}")
    print(f"Average solutions per state: {total_solutions / puzzle_count:.2f}")

    # Pad solutions
    seq_len = 11
    max_solutions = max(len(sols) for sols in labels)

    labels_padded = np.zeros((len(labels), max_solutions, seq_len), dtype=np.int32)
    for i, solutions in enumerate(labels):
        for s_idx, sol in enumerate(solutions):
            sol_len = len(sol)
            if sol_len > 0:
                labels_padded[i, s_idx, :sol_len] = np.array(sol, dtype=np.int32) + 1

    # Split by groups so all orientations of the same cube stay together
    total_groups = group_id
    train_group_count = int(total_groups * config.train_split)
    test_group_count = int(total_groups * config.test_split)

    group_perm = np.random.permutation(total_groups)
    train_groups = sorted(group_perm[:train_group_count])
    test_groups = sorted(group_perm[train_group_count:train_group_count + test_group_count])
    val_groups = sorted(group_perm[train_group_count + test_group_count:])

    splits = {
        "train": train_groups,
        "test": test_groups,
        "val": val_groups,
    }

    # Save identifiers mapping
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    group_indices_arr = np.array(group_indices, dtype=np.int32)
    inputs_arr = np.array(inputs, dtype=np.int32)
    identifiers_arr = np.array(puzzle_identifiers, dtype=np.int32)

    # Save each split
    for split_name, split_groups in splits.items():
        # Collect all puzzle (row) indices belonging to these groups
        split_row_indices = []
        for gid in split_groups:
            start = group_indices_arr[gid]
            end = group_indices_arr[gid + 1]
            split_row_indices.extend(range(start, end))

        split_row_indices = np.array(split_row_indices, dtype=np.int32)

        # Extract data for this split
        inputs_split = inputs_arr[split_row_indices]
        labels_split = labels_padded[split_row_indices]
        identifiers_split = identifiers_arr[split_row_indices]

        # regroup for test and val set
        new_puzzle_indices = [0]
        new_group_indices = [0]
        puzzle_counter = 0

        for gid in split_groups:
            group_size = int(group_indices_arr[gid + 1] - group_indices_arr[gid])
            for _ in range(group_size):
                puzzle_counter += 1
                new_puzzle_indices.append(puzzle_counter)
            new_group_indices.append(puzzle_counter)

        num_groups = len(split_groups)
        mean_examples = len(split_row_indices) / max(num_groups, 1)

        results_split = {
            "inputs": inputs_split,
            "labels": labels_split,
            "group_indices": np.array(new_group_indices, dtype=np.int32),
            "puzzle_indices": np.array(new_puzzle_indices, dtype=np.int32),
            "puzzle_identifiers": identifiers_split,
        }

        # Metadata for this split
        metadata = PuzzleDatasetMetadata(
            seq_len=seq_len,
            vocab_size=19,  # PAD(0) + 18 moves (because of the augmentation)

            pad_id=0,
            ignore_label_id=0,

            blank_identifier_id=0,
            num_puzzle_identifiers=1,

            total_groups=num_groups,
            mean_puzzle_examples=mean_examples,
            sets=[split_name],
        )

        # Save split
        save_dir = os.path.join(config.output_dir, split_name)
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)

        for k, v in results_split.items():
            np.save(os.path.join(save_dir, f"{split_name}__{k}.npy"), v)


@cli.command(singleton=True)
def build(config: DataProcessConfig):
    generate_dataset(config)


if __name__ == "__main__":
    cli()
