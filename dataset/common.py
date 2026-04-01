from typing import List, Optional

import pydantic
import numpy as np


# Global list mapping each dihedral transform id to its inverse.
# Index corresponds to the original tid, and the value is its inverse.
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


class PuzzleDatasetMetadata(pydantic.BaseModel):
    pad_id: int
    ignore_label_id: Optional[int]
    blank_identifier_id: int
    
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    
    total_groups: int
    mean_puzzle_examples: float

    sets: List[str]


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries by rotate, flip and mirror"""
    
    if tid == 0:
        return arr  # identity
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)       # horizontal flip
    elif tid == 5:
        return np.flipud(arr)       # vertical flip
    elif tid == 6:
        return arr.T                # transpose (reflection along main diagonal)
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-diagonal reflection
    else:
        return arr
    
    
def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])


def split_puzzles_by_id(puzzle_ids: List[str], test_fraction: float = 0.2, seed: int = 42) -> tuple[set[str], set[str]]:
    """Split puzzle IDs into train and test sets to avoid data leakage.
    
    Args:
        puzzle_ids: List of puzzle identifiers
        test_fraction: Fraction of puzzles to reserve for testing
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_puzzle_ids, test_puzzle_ids)
    """
    import random
    random.seed(seed)
    
    shuffled_ids = puzzle_ids.copy()
    random.shuffle(shuffled_ids)
    
    num_test = int(len(shuffled_ids) * test_fraction)
    test_ids = set(shuffled_ids[:num_test])
    train_ids = set(shuffled_ids[num_test:])
    
    return train_ids, test_ids
