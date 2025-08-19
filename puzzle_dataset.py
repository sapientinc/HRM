import os
import json

import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from models.losses import IGNORE_LABEL_ID
from dataset.common import PuzzleDatasetMetadata


def _sample_batch(rng: np.random.Generator, group_order: np.ndarray, puzzle_indices: np.ndarray, group_indices: np.ndarray, start_index: int, global_batch_size: int):
    # Pack examples into a full batch
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # Get range of the puzzle
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, global_batch_size - current_size)

        # Put into batch
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(puzzle_start + np.random.choice(puzzle_size, append_size, replace=False))

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_path: str
    global_batch_size: int
    test_set_mode: bool

    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.

    rank: int
    num_replicas: int


class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split
        self.metadata = self._load_metadata()
        
        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0

    def _load_metadata(self) -> PuzzleDatasetMetadata:
        with open(os.path.join(self.config.dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",
            "labels": "r",

            # Keep indices in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None
        }

        # Load data
        self._data = {}
        for set_name in self.metadata.sets:
            # Load subset
            self._data[set_name] = {
                field_name: np.load(os.path.join(self.config.dataset_path, self.split, f"{set_name}__{field_name}.npy"), mmap_mode=mmap_mode)
                for field_name, mmap_mode in field_mmap_modes.items()
            }

    def _collate_batch(self, batch):
        # Convert dtype
        batch = {k: v.astype(np.int32) for k, v in batch.items()}

        # Convert ignore label IDs
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID

        # Pad
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size

            pad_values = {
                "inputs": self.metadata.pad_id,
                "labels": IGNORE_LABEL_ID,

                "puzzle_identifiers": self.metadata.blank_identifier_id
            }
            batch = {k: np.pad(v, ((0, pad_size), ) + ((0, 0), ) * (v.ndim - 1), constant_values=pad_values[k]) for k, v in batch.items()}

        # To tensor
        return {k: torch.from_numpy(v) for k, v in batch.items()}
    
    def _iter_test(self):
        for set_name, dataset in self._data.items():  # type: ignore
            total_examples = len(dataset["inputs"])

            # Load examples one by one
            start_index = 0
            while start_index < total_examples:
                # Compute indices
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end   = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
                
                # Get batch of examples, and also puzzle IDs
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)
                
                batch = self._collate_batch({
                    "inputs": dataset["inputs"][local_start: local_end],
                    "labels": dataset["labels"][local_start: local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
                })

                yield set_name, batch, end_index - start_index
                
                # Advance to next batch
                start_index += self.config.global_batch_size

    def _iter_train(self):
        # This method will now be called by the new __iter__ method.
        # Its logic is being moved into __iter__ to handle workers correctly.
        # For clarity, we will replace its contents inside __iter__.
        pass

    def __iter__(self):
        self._lazy_load_dataset()
        worker_info = get_worker_info()

        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        
        # --- TEST MODE LOGIC (NOW PARALLEL-SAFE) ---
        if self.config.test_set_mode:
            for set_name, dataset in self._data.items():
                total_examples = len(dataset["inputs"])
                
                # Assign a unique, non-overlapping slice of the test set to each worker
                per_worker = int(np.ceil(total_examples / float(num_workers)))
                start_index = worker_id * per_worker
                end_index = min(start_index + per_worker, total_examples)

                # Each worker now iterates only over its assigned slice
                current_pos = start_index
                while current_pos < end_index:
                    batch_end = min(current_pos + self.local_batch_size, end_index)
                    
                    # The logic from the original _iter_test to find puzzle identifiers
                    puzzle_indices = []
                    puzzle_index = np.searchsorted(dataset["puzzle_indices"], current_pos, side="right") - 1
                    for i in range(current_pos, batch_end):
                        while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                            puzzle_index += 1
                        puzzle_indices.append(puzzle_index)
                    
                    batch = self._collate_batch({
                        "inputs": dataset["inputs"][current_pos:batch_end],
                        "labels": dataset["labels"][current_pos:batch_end],
                        "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices]
                    })

                    yield set_name, batch, self.config.global_batch_size
                    current_pos = batch_end
            return # End iteration after test mode is complete

        # --- TRAINING MODE LOGIC (FROM PREVIOUS FIX) ---
        for set_name, dataset in self._data.items():
            self._iters += 1
            seed = self.config.seed + self._iters
            rng = np.random.Generator(np.random.Philox(seed=seed))

            if (dataset["group_indices"].size - 1) <= 0: continue

            group_order = np.concatenate([rng.permutation(dataset["group_indices"].size - 1) for _i in range(self.config.epochs_per_iter)])
            
            per_worker = int(np.ceil(len(group_order) / float(num_workers)))
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(group_order))
            worker_group_order = group_order[start_idx:end_idx]

            if len(worker_group_order) == 0: continue

            start_index = 0
            while start_index < len(worker_group_order):
                batch_size_for_worker = self.local_batch_size
                
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=worker_group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=batch_size_for_worker,
                )

                if batch_puzzle_indices.size == 0: break # Break if no more batches can be formed

                batch = self._collate_batch({
                    "inputs": dataset["inputs"][batch_indices],
                    "labels": dataset["labels"][batch_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices]
                })

                yield set_name, batch, self.config.global_batch_size
