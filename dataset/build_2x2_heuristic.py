import json
from argdantic import ArgParser
import numpy as np
from pathlib import Path
import json
import numpy as np
import magiccube
import os
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata

class DataProcessConfig(BaseModel):
    source_repo: str = "data/cube-2-by-2"
    output_dir: str = "data/cube-2-by-2-heuristic"

    train_size: int = 1000000
    test_size: int = 1000

def load_dataset_split(data_dir: Path, split: str):
    #Load a dataset split (train/test/val)
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split {split} not found in {data_dir}")

    # Load metadata
    with open(split_dir / "dataset.json", "r") as f:
        metadata = json.load(f)

    # Load arrays
    inputs = np.load(split_dir / "all__inputs.npy")
    labels = np.load(split_dir / "all__labels.npy")
    group_indices = np.load(split_dir / "all__group_indices.npy")
    puzzle_indices = np.load(split_dir / "all__puzzle_indices.npy")
    puzzle_identifiers = np.load(split_dir / "all__puzzle_identifiers.npy")

    return {
        'metadata': metadata,
        'inputs': inputs,
        'labels': labels,
        'group_indices': group_indices,
        'puzzle_indices': puzzle_indices,
        'puzzle_identifiers': puzzle_identifiers,
    }


def py222_to_magiccube(state):
    color_map = {0: 'W', 1: 'R', 2: 'G', 3: 'Y', 4: 'O', 5: 'B'}
    
    # Reorder stickers from px222 to magiccube notation
    magic_order = (
        list(range(0, 4)) +
        list(range(16, 20)) +
        list(range(8, 12)) +
        list(range(4, 8)) +
        list(range(20, 24)) +
        list(range(12, 16))
    )
    return ''.join([color_map[state[i]] for i in magic_order])


def process(data, size):
    # Generate cubes
    results = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [0],
        "group_indices": [0],
    }

    example_id = 0
    count = 0
    
    for j in tqdm(range(len(data["inputs"]))):
        if count >= size:
            break

        inputs = data["inputs"][j]
        labels = data["labels"][j]
        moves = [int(m) - 1 for m in labels if m != 0]
        if len(moves) < 8:
            continue
    
        count += 1

        py222_state = inputs - 1
        state = py222_to_magiccube(py222_state)
        cube = magiccube.Cube(2, state)
        move_names = {0: "U", 1: "U'", 2: "U2", 3: "R", 4: "R'", 5: "R2", 6: "F", 7: "F'", 8: "F2"}
        solution = [move_names[m] for m in moves]
        #colors = {1: 'Y', 2: 'R', 3: 'G', 4: 'O', 5: 'B', 6: 'W'}
        colors = {'Y': 0, 'R': 1, 'G': 2, 'O': 3, 'B': 4, 'W': 5}

        for i, move in enumerate(solution):
            distance_to_solved = len(solution) - i
            current_state_str = str(cube).replace(" ", "").replace("\n", "")
            current_state = [colors[c] for c in current_state_str]
            
            results['inputs'].append(current_state)
            results['labels'].append(distance_to_solved)
            results['puzzle_identifiers'].append(0)
            
            example_id += 1
            results['puzzle_indices'].append(example_id)
            
            cube.rotate(move)
        
        # Also add the solved state with distance 0
        solved_state_str = str(cube).replace(" ", "").replace("\n", "")
        solved_state = [colors[c] for c in solved_state_str]
        results['inputs'].append(solved_state)
        results['labels'].append(0)
        results['puzzle_identifiers'].append(0)
        
        example_id += 1
        results['puzzle_indices'].append(example_id)
        
        results['group_indices'].append(example_id)
    
    return results
    

def create_dataset(set_name, config: DataProcessConfig):
    data = load_dataset_split(Path(config.source_repo), set_name)


    if set_name == "train":
        size = config.train_size
    else:
        size = config.test_size
    
    results = process(data, size)

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=24,        # 4 * 6
        vocab_size=6,      # 6 colors
        
        pad_id=0,              # Not used
        ignore_label_id=None,  # No ignored labels
        
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        sets=["all"]
    )

    # Save metadata as JSON.
    save_dir = Path(config.output_dir).resolve() / set_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(str(save_dir / "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
        
    # Labels need to be (N, 1) to match dataloader
    results["labels"] = np.array(results["labels"]).reshape(-1, 1)
    results["inputs"] = np.array(results["inputs"])
    results["puzzle_identifiers"] = np.array(results["puzzle_identifiers"])
    results["puzzle_indices"] = np.array(results["puzzle_indices"])
    results["group_indices"] = np.array(results["group_indices"])
    
    for k, v in results.items():
        np.save(str(save_dir / f"all__{k}.npy"), v)
        
    # Save IDs mapping (for visualization only)
    identifiers_path = Path(config.output_dir).resolve() / "identifiers.json"
    identifiers_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(identifiers_path), "w") as f:
        json.dump(["<blank>"], f)


cli = ArgParser()

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    create_dataset("train", config)
    create_dataset("test", config)
    create_dataset("val", config)


if __name__ == "__main__":
    cli()
