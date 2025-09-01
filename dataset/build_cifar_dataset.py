from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
import os
import json
import hashlib
import numpy as np
from glob import glob

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from argdantic import ArgParser
from pydantic import BaseModel

from common import PuzzleDatasetMetadata, dihedral_transform


cli = ArgParser()


class DataProcessConfig(BaseModel):
    # CIFAR dataset configuration
    dataset_name: str = "CIFAR10"  # or "CIFAR100"
    output_dir: str = "data/cifar-aug-1000"
    
    # Data augmentation
    seed: int = 42
    num_aug: int = 1000
    
    # Image processing
    image_size: int = 32  # CIFAR images are 32x32
    patch_size: int = 4   # Divide image into 4x4 patches for sequence modeling
    num_channels: int = 3  # RGB channels


CIFARMaxGridSize = 32  # 32x32 images
CIFARAugmentRetriesFactor = 5


@dataclass
class CIFARPuzzle:
    id: str
    examples: List[Tuple[np.ndarray, np.ndarray]]  # (image_patches, class_label)


def image_to_patches(image: np.ndarray, patch_size: int = 4) -> np.ndarray:
    """Convert image to sequence of patches for HRM processing."""
    # image shape: (H, W, C) or (C, H, W)
    if image.shape[0] == 3:  # (C, H, W)
        image = np.transpose(image, (1, 2, 0))  # Convert to (H, W, C)
    
    H, W, C = image.shape
    assert H % patch_size == 0 and W % patch_size == 0, f"Image size {H}x{W} must be divisible by patch size {patch_size}"
    
    # Reshape to patches
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    patches = image.reshape(num_patches_h, patch_size, num_patches_w, patch_size, C)
    patches = patches.transpose(0, 2, 1, 3, 4)  # (num_patches_h, num_patches_w, patch_size, patch_size, C)
    patches = patches.reshape(-1, patch_size, patch_size, C)  # (num_patches, patch_size, patch_size, C)
    
    # Flatten each patch and normalize to [0, 1] range
    patches_flat = patches.reshape(patches.shape[0], -1)  # (num_patches, patch_size*patch_size*C)
    patches_flat = patches_flat / 255.0  # Normalize to [0, 1]
    
    # Convert to integer tokens (0-255 range, but we'll use a smaller vocab)
    # We'll quantize the values to reduce vocabulary size
    vocab_size = 256  # 0-255 for pixel values
    patches_quantized = (patches_flat * (vocab_size - 1)).astype(np.uint8)
    
    return patches_quantized


def patches_to_sequence(patches: np.ndarray, max_seq_len: int) -> np.ndarray:
    """Convert patches to sequence format expected by HRM."""
    # Flatten patches to sequence
    seq = patches.flatten()
    
    # Pad or truncate to max_seq_len
    if len(seq) > max_seq_len:
        seq = seq[:max_seq_len]
    else:
        # Pad with zeros
        pad_len = max_seq_len - len(seq)
        seq = np.concatenate([seq, np.zeros(pad_len, dtype=np.uint8)])
    
    return seq


def puzzle_hash(puzzle: dict):
    """Hash the puzzle for checking equivalence."""
    def _image_hash(image: np.ndarray):
        buffer = [x.to_bytes(1) for x in image.shape]
        buffer.append(image.tobytes())
        return hashlib.sha256(b"".join(buffer)).hexdigest()
    
    hashes = []
    for example_type, example in puzzle.items():
        for image, label in example.examples:
            hashes.append(f"{_image_hash(image)}|{label}")
            
    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def convert_single_cifar_puzzle(results: dict, default_name: str, puzzle: dict, aug_count: int, dest_mapping: Dict[str, Tuple[str, str]]):
    """Convert a single CIFAR puzzle to HRM format."""
    # Remove "name"
    name = puzzle.pop("name", default_name)
    
    # Convert
    dests = set(dest_mapping.values())
    converted = {dest: CIFARPuzzle(name, []) for dest in dests}
    for example_type, examples in puzzle.items():
        dest = dest_mapping[example_type]
        converted[dest].examples.extend([(image_to_patches(example["image"]), example["label"]) for example in examples])

    group = [converted]
    
    # Augment
    if aug_count > 0:
        hashes = {puzzle_hash(converted)}

        for _trial in range(CIFARAugmentRetriesFactor * aug_count):
            # Augment plan - use dihedral transforms for images
            trans_id = np.random.randint(0, 8)
            
            # Color augmentation - randomly adjust brightness/contrast
            brightness_factor = np.random.uniform(0.8, 1.2)
            contrast_factor = np.random.uniform(0.8, 1.2)
            
            aug_repr = f"t{trans_id}_b{brightness_factor:.2f}_c{contrast_factor:.2f}"

            def _augment_image(image_patches: np.ndarray):
                # Convert back to image format for augmentation
                patch_size = 4
                num_patches = image_patches.shape[0]
                patches_per_side = int(np.sqrt(num_patches))
                
                # Reshape to image
                image = image_patches.reshape(patches_per_side, patches_per_side, patch_size, patch_size, 3)
                image = image.transpose(0, 2, 1, 3, 4).reshape(32, 32, 3)
                
                # Apply dihedral transform
                image = dihedral_transform(image, trans_id)
                
                # Apply color augmentation
                image = image * brightness_factor
                image = np.clip(image, 0, 255)
                
                # Apply contrast
                mean = np.mean(image)
                image = (image - mean) * contrast_factor + mean
                image = np.clip(image, 0, 255)
                
                # Convert back to patches
                return image_to_patches(image.astype(np.uint8))
            
            # Check duplicate
            augmented = {dest: CIFARPuzzle(f"{name}_{aug_repr}", [(_augment_image(image), label) for (image, label) in puzzle.examples]) for dest, puzzle in converted.items()}
            h = puzzle_hash(augmented)
            if h not in hashes:
                hashes.add(h)
                group.append(augmented)
                
            if len(group) >= aug_count + 1:
                break
            
        if len(group) < aug_count + 1:
            print(f"[Puzzle {name}] augmentation not full, only {len(group)}")

    # Append
    for dest in dests:
        # Convert the examples
        dest_split, dest_set = dest

        results.setdefault(dest_split, {})
        results[dest_split].setdefault(dest_set, [])
        results[dest_split][dest_set].append([converted[dest] for converted in group])


def load_cifar_dataset(config: DataProcessConfig):
    """Load CIFAR dataset and convert to HRM format."""
    # Load CIFAR dataset
    if config.dataset_name == "CIFAR10":
        dataset_class = torchvision.datasets.CIFAR10
        num_classes = 10
    elif config.dataset_name == "CIFAR100":
        dataset_class = torchvision.datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset_name}")
    
    # Load train and test sets
    train_dataset = dataset_class(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = dataset_class(root='./data', train=False, download=True, transform=transforms.ToTensor())
    
    # Convert to our format
    train_examples_dest = ("train", "all")
    test_examples_dest = ("test", "all")
    
    # Process training data
    train_puzzles = []
    for idx, (image, label) in enumerate(train_dataset):
        puzzle = {
            "train": {
                "examples": [{"image": image.numpy().transpose(1, 2, 0) * 255, "label": label}]
            }
        }
        train_puzzles.append((f"train_{idx}", puzzle))
    
    # Process test data
    test_puzzles = []
    for idx, (image, label) in enumerate(test_dataset):
        puzzle = {
            "test": {
                "examples": [{"image": image.numpy().transpose(1, 2, 0) * 255, "label": label}]
            }
        }
        test_puzzles.append((f"test_{idx}", puzzle))
    
    return train_puzzles, test_puzzles, num_classes


def convert_dataset(config: DataProcessConfig):
    """Convert CIFAR dataset to HRM format."""
    np.random.seed(config.seed)
    
    # Load CIFAR data
    train_puzzles, test_puzzles, num_classes = load_cifar_dataset(config)
    
    # Combine all puzzles
    all_puzzles = train_puzzles + test_puzzles
    
    # Map global puzzle identifiers
    num_identifiers = 1  # 0 is blank
    identifier_map = {}
    for puzzle_id, _ in all_puzzles:
        if puzzle_id not in identifier_map:
            identifier_map[puzzle_id] = num_identifiers
            num_identifiers += 1

    print(f"Total puzzle IDs (including <blank>): {num_identifiers}")
    print(f"Number of classes: {num_classes}")

    # Process puzzles
    results = {}
    
    # Process training puzzles
    for default_name, puzzle in train_puzzles:
        convert_single_cifar_puzzle(results, default_name, puzzle, config.num_aug, 
                                  {"train": ("train", "all"), "test": ("train", "all")})
    
    # Process test puzzles
    for default_name, puzzle in test_puzzles:
        convert_single_cifar_puzzle(results, default_name, puzzle, 0,  # No augmentation for test
                                  {"train": ("test", "all"), "test": ("test", "all")})

    # Calculate sequence length
    # Each image is 32x32, divided into 4x4 patches = 8x8 = 64 patches
    # Each patch has 4*4*3 = 48 values, so total sequence length = 64 * 48 = 3072
    patch_size = config.patch_size
    image_size = config.image_size
    patches_per_side = image_size // patch_size
    values_per_patch = patch_size * patch_size * config.num_channels
    seq_len = patches_per_side * patches_per_side * values_per_patch
    
    # Save
    for split_name, split in results.items():
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)
        
        # Statistics
        total_examples = 0
        total_puzzles = 0
        total_groups = 0
        
        for subset_name, subset in split.items():
            # Construct subset
            results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
            results["puzzle_indices"].append(0)
            results["group_indices"].append(0)
            
            example_id = 0
            puzzle_id = 0
            
            for group in subset:
                for puzzle in group:
                    # Push puzzle
                    for _idx_ex, (image_patches, label) in enumerate(puzzle.examples):
                        # Convert patches to sequence
                        input_seq = patches_to_sequence(image_patches, seq_len)
                        
                        # Create label sequence (repeat class label for the entire sequence)
                        label_seq = np.full(seq_len, label, dtype=np.uint8)
                        
                        results["inputs"].append(input_seq)
                        results["labels"].append(label_seq)
                        example_id += 1
                        
                        total_examples += 1

                    results["puzzle_indices"].append(example_id)
                    results["puzzle_identifiers"].append(identifier_map[puzzle.id])
                    
                    puzzle_id += 1
                    total_puzzles += 1
                    
                # Push group
                results["group_indices"].append(puzzle_id)
                total_groups += 1
            
            for k, v in results.items():
                if k in {"inputs", "labels"}:
                    v = np.stack(v, 0)
                else:
                    v = np.array(v, dtype=np.int32)
                
                np.save(os.path.join(config.output_dir, split_name, f"{subset_name}__{k}.npy"), v)
        
        # Metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=seq_len,
            vocab_size=256,  # 0-255 for pixel values
            
            pad_id=0,
            ignore_label_id=0,
            
            blank_identifier_id=0,
            num_puzzle_identifiers=num_identifiers,
            
            total_groups=total_groups,
            mean_puzzle_examples=total_examples / total_puzzles,
            sets=list(split.keys())
        )

        # Save metadata as JSON.
        with open(os.path.join(config.output_dir, split_name, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)
            
    # Save IDs mapping
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        ids_mapping = {v: k for k, v in identifier_map.items()}
        json.dump([ids_mapping.get(i, "<blank>") for i in range(num_identifiers)], f)


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    convert_dataset(config)


if __name__ == "__main__":
    cli()
