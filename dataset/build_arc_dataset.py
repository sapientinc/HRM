from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from functools import cached_property, partial
import itertools
from pathlib import Path
import random
from typing import Annotated, Final, Literal , get_args

import hashlib
import json
import numpy as np
from numpy.typing import NDArray

import tqdm

from pydantic import BaseModel, BeforeValidator, Field, ConfigDict, TypeAdapter, ValidationError

from common import PuzzleDatasetMetadata, dihedral_transform


ARC_MAX_GRID_SIZE: Final[int] = 30
ARC_MAX_COLOR_VALUE: Final[int] = 9
BLANK_PUZZLE_ID: Final[str] = "<blank>"
BLACK_COLOR: Final[int] = 0
N_PADDING_TOKENS: Final[int] = 1
N_EOS_TOKENS: Final[int] = 1
DIHEDRAL_SYMMETRIES: Final[int] = 8

PAD_TOKEN: Final[int] = 0
END_OF_SEQUENCE_TOKEN: Final[int] = 1
COLOR_OFFSET_TOKEN: Final[int] = 2

# TODO: I've removed the "set" logic as it was unused
SET_NAME: Final[str] = "all"


class ArcIOKey(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


RawArcSplit = Annotated[
    Literal["training", "evaluation"],
    BeforeValidator(lambda x: "evaluation" if x == "evaluation" else "training"), # ConceptARC data assumed to be training
]
RawArcSplitAdapter: TypeAdapter[RawArcSplit] = TypeAdapter(RawArcSplit)

ProcessedArcSplit = Literal["train", "test"]
ARCExampleType = Literal["train", "test"]
RawPuzzle = dict[ARCExampleType, list[dict[ArcIOKey, list[list[int]]]]]

# For clarity
GridArray = NDArray[np.uint8]
FlatArray = NDArray[np.uint8]



class ARCDatasetBuildConfig(BaseModel):
    seed: int = Field(default=42)
    num_aug: int = Field(default=1000, ge=0)
    augment_retries_factor: int = Field(default=5)

    raw_dataset_dirs: list[str | Path] = Field(default=["dataset/raw-data/ARC-AGI/data", "dataset/raw-data/ConceptARC/corpus"])

    processed_dataset_dir: str = Field(default="data/arc-aug-1000")
    identifiers_filename: str = Field(default="identifiers")
    metadata_filename: str = Field(default="dataset")


class PuzzleExample(BaseModel):
    example_type: ARCExampleType
    input_grid: GridArray
    output_grid: GridArray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def pad(self) -> "PuzzleExample":
        return self.model_copy(update={
            "input_grid": _pad_grid(self.input_grid),
            "output_grid": _pad_grid(self.output_grid),
        })


def _grid_to_bytes(grid: GridArray) -> bytes:
    return (
        grid.shape[0].to_bytes(1, "little")
        + grid.shape[1].to_bytes(1, "little")
        + grid.tobytes()
    )


class ARCPuzzle(BaseModel):
    id: str
    split: RawArcSplit
    examples: list[PuzzleExample]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def hash(self) -> str:
        example_buffers = sorted(
            _grid_to_bytes(example.input_grid) + _grid_to_bytes(example.output_grid)
            for example in self.examples
        )

        hasher = hashlib.blake2b(digest_size=8)
        for buf in example_buffers:
            hasher.update(buf)

        return hasher.hexdigest()

    @classmethod
    def from_raw_file(cls, filepath: Path, arc_split: RawArcSplit, puzzle_id: str) -> "ARCPuzzle":
        with open(filepath) as f:
            data = json.load(f)
        
        valid_example_types = get_args(ARCExampleType)
        raw_puzzle_data = {
            key: data[key] for key in valid_example_types if key in data
        }
        return cls._from_raw(raw_puzzle_data, arc_split, puzzle_id)

    @classmethod
    def _from_raw(cls, raw_puzzle: RawPuzzle, split: RawArcSplit, puzzle_id: str) -> "ARCPuzzle":
        return cls(
            id=puzzle_id,
            split=split,
            examples=[
                PuzzleExample(
                    example_type=example_type,
                    input_grid=_parse_raw_grid(example[ArcIOKey.INPUT]),
                    output_grid=_parse_raw_grid(example[ArcIOKey.OUTPUT]),
                )
                for example_type, examples in raw_puzzle.items()
                for example in examples
            ],
        )


class DatasetSplit(BaseModel):
    flattened_inputs: list[FlatArray] = Field(default_factory=list)
    flattened_labels: list[FlatArray] = Field(default_factory=list)
    puzzle_identifiers: list[int] = Field(default_factory=list)
    puzzle_indices: list[int] = Field(default_factory=lambda: [0])
    group_indices: list[int] = Field(default_factory=lambda: [0])

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _save_dataset_split(dataset_split: DatasetSplit, split_dir: Path) -> None:
    data = {
        "inputs": np.stack(dataset_split.flattened_inputs, 0),
        "labels": np.stack(dataset_split.flattened_labels, 0),
        "puzzle_identifiers": np.array(dataset_split.puzzle_identifiers, dtype=np.int32),
        "puzzle_indices": np.array(dataset_split.puzzle_indices, dtype=np.int32),
        "group_indices": np.array(dataset_split.group_indices, dtype=np.int32),
    }

    for name, array in data.items():
        np.save(split_dir / f"{SET_NAME}__{name}.npy", array)


def _parse_raw_grid(raw_grid: list[list[int]]) -> GridArray:
    arr = np.array(raw_grid, dtype=np.uint8)

    if arr.ndim != 2:
        raise ValueError(f"Grid must be 2D, got {arr.ndim}D")
    if arr.shape[0] > ARC_MAX_GRID_SIZE or arr.shape[1] > ARC_MAX_GRID_SIZE:
        raise ValueError(f"Grid size {arr.shape} exceeds maximum {ARC_MAX_GRID_SIZE}")
    if not np.all((arr >= BLACK_COLOR) & (arr <= ARC_MAX_COLOR_VALUE)):
        raise ValueError(f"Grid values must be in range [0, {ARC_MAX_COLOR_VALUE}]")

    return arr


def _pad_grid(grid: GridArray, row_padding: int = 0, col_padding: int = 0) -> GridArray:
    n_rows, n_cols = grid.shape
    
    padded = np.full((ARC_MAX_GRID_SIZE, ARC_MAX_GRID_SIZE), PAD_TOKEN, dtype=np.uint8)
    padded[row_padding:row_padding + n_rows, col_padding:col_padding + n_cols] = grid + COLOR_OFFSET_TOKEN

    eos_row, eos_col = row_padding + n_rows, col_padding + n_cols
    if eos_row < ARC_MAX_GRID_SIZE:
        padded[eos_row, col_padding:eos_col] = END_OF_SEQUENCE_TOKEN
    if eos_col < ARC_MAX_GRID_SIZE:
        padded[row_padding:eos_row, eos_col] = END_OF_SEQUENCE_TOKEN
    
    return padded


def _apply_translational_augment(example: PuzzleExample) -> PuzzleExample:
    max_rows = max(example.input_grid.shape[0], example.output_grid.shape[0])
    max_cols = max(example.input_grid.shape[1], example.output_grid.shape[1])
    row_padding = np.random.randint(0, ARC_MAX_GRID_SIZE - max_rows + 1)
    col_padding = np.random.randint(0, ARC_MAX_GRID_SIZE - max_cols + 1)

    return PuzzleExample(
        example_type=example.example_type,
        input_grid=_pad_grid(example.input_grid, row_padding, col_padding),
        output_grid=_pad_grid(example.output_grid, row_padding, col_padding),
    )


def _apply_grid_transform(grid: GridArray, trans_id: int, color_map: np.ndarray) -> GridArray:
    transformed_grid = dihedral_transform(grid, trans_id)
    return np.take(color_map, transformed_grid)


def _apply_dihedral_transform_augment(puzzle: ARCPuzzle) -> ARCPuzzle:
    trans_id = np.random.randint(0, DIHEDRAL_SYMMETRIES)
    color_map = np.concatenate(
        [
            np.arange(BLACK_COLOR, BLACK_COLOR + 1, dtype=np.uint8),
            np.random.permutation(np.arange(BLACK_COLOR + 1, ARC_MAX_COLOR_VALUE + 1, dtype=np.uint8)),
        ]
    )
    aug_repr = f"t{trans_id}_{hashlib.blake2b(color_map.tobytes(), digest_size=8).hexdigest()}"

    augmented_examples = [
        PuzzleExample(
            example_type=example.example_type,
            input_grid=_apply_grid_transform(example.input_grid, trans_id, color_map),
            output_grid=_apply_grid_transform(example.output_grid, trans_id, color_map),
        )
        for example in puzzle.examples
    ]

    return ARCPuzzle(id=f"{puzzle.id}_{aug_repr}", split=puzzle.split, examples=augmented_examples)


def _generate_puzzle_augmentations(puzzle: ARCPuzzle, aug_count: int, augment_retries_factor: int) -> list[ARCPuzzle]:
    group = [puzzle]

    seen_puzzles = {puzzle.hash}
    for _ in range(augment_retries_factor * aug_count):
        if (augmented := _apply_dihedral_transform_augment(puzzle)).hash not in seen_puzzles:
            seen_puzzles.add(augmented.hash)
            group.append(augmented)
        if len(group) >= aug_count + 1:
            break

    no_translation_idx = np.random.randint(0, len(puzzle.examples))
    return [
        ARCPuzzle(
            id=puzzle.id,
            split=puzzle.split,
            examples=[
                _apply_translational_augment(example) if idx != no_translation_idx and puzzle.split == "training" else example.pad()
                for idx, example in enumerate(puzzle.examples)
            ],
        )
        for puzzle in group
    ]


def _load_split_raw_puzzles(arc_split: RawArcSplit, dirpath: Path) -> list[ARCPuzzle]:
    puzzles = [
        ARCPuzzle.from_raw_file(
            filepath=Path(filename),
            arc_split=arc_split,
            puzzle_id=Path(filename).stem,
        )
        for filename in Path(dirpath).glob("*.json")
    ]

    random.shuffle(puzzles)
    return puzzles


def _load_all_raw_puzzles(dataset_dir: str | Path) -> list[ARCPuzzle]:
    puzzles = []

    subdirs = (d for d in Path(dataset_dir).iterdir() if d.is_dir())
    for split_dir in subdirs:
        arc_split: RawArcSplit = RawArcSplitAdapter.validate_python(split_dir.name)
        puzzles.extend(_load_split_raw_puzzles(arc_split, split_dir))
    return puzzles

def _process_single_puzzle(puzzle: ARCPuzzle, config: ARCDatasetBuildConfig) -> tuple[RawArcSplit, list[ARCPuzzle]]:
    augmented_puzzles = _generate_puzzle_augmentations(
        puzzle, config.num_aug, config.augment_retries_factor
    )
    return puzzle.split, augmented_puzzles


def _split_puzzle_augmentations(
    original_puzzle_split: RawArcSplit,
    puzzle_augmenations: list[ARCPuzzle]
) -> tuple[list[ARCPuzzle], list[ARCPuzzle]]:
    if original_puzzle_split == "training":
        return puzzle_augmenations, []

    def create_split(example_type: ARCExampleType) -> list[ARCPuzzle]:
        return [
            p.model_copy(update={"examples": examples})
            for p in puzzle_augmenations
            if (examples := [ex for ex in p.examples if ex.example_type == example_type])
        ]

    return create_split("train"), create_split("test")


def _process_arcagi_dataset(
    dataset_paths: list[str | Path], config: ARCDatasetBuildConfig
) -> dict[ProcessedArcSplit, list[list[ARCPuzzle]]]:
    puzzles = []
    for dataset_path in dataset_paths:
        if Path(dataset_path).exists():
            dataset_puzzles = _load_all_raw_puzzles(dataset_path)
            puzzles.extend(dataset_puzzles)
            print(f"[{dataset_path}] loaded {len(dataset_puzzles)} puzzles")
    
    train_groups: list[list[ARCPuzzle]] = []
    test_groups: list[list[ARCPuzzle]] = []
    process_puzzle = partial(_process_single_puzzle, config=config)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_puzzle, puzzle) for puzzle in puzzles]
        progress = tqdm.tqdm(as_completed(futures), total=len(puzzles), desc="Processing ARC puzzles")
        for future in progress:
            original_split, augmented_puzzles = future.result()
            train_batch, test_batch = _split_puzzle_augmentations(original_split, augmented_puzzles)
            train_groups.append(train_batch)
            test_groups.append(test_batch)

    return {"train": train_groups, "test": test_groups}


def _build_identifier_map(data: dict[ProcessedArcSplit, list[list[ARCPuzzle]]]) -> dict[str, int]:
    num_identifiers = 1  # First identifier is reserved for blank puzzle
    identifier_map = {}
    for puzzle in itertools.chain.from_iterable(itertools.chain.from_iterable(data.values())):
        if puzzle.id in identifier_map:
            continue
        identifier_map[puzzle.id] = num_identifiers
        num_identifiers += 1
    return identifier_map


def _save_split_data(split: ProcessedArcSplit, puzzles: list[list[ARCPuzzle]], identifier_map: dict[str, int], output_dir: Path, metadata_filename: str) -> None:
    split_data = DatasetSplit()

    total_examples = 0
    total_puzzles = 0

    for augmented_puzzles in tqdm.tqdm(puzzles, desc=f"Preparing {split} data for saving"):
        for puzzle in augmented_puzzles:
            total_examples += len(puzzle.examples)
            split_data.flattened_inputs.extend(example.input_grid.flatten() for example in puzzle.examples)
            split_data.flattened_labels.extend(example.output_grid.flatten() for example in puzzle.examples)
            split_data.puzzle_indices.append(total_examples)
            split_data.puzzle_identifiers.append(identifier_map[puzzle.id])

        total_puzzles += len(augmented_puzzles)
        split_data.group_indices.append(total_puzzles)

    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)
    _save_dataset_split(split_data, split_output_dir)
    print(f"Saved {split} data to {split_output_dir}!")

    _save_dataset_metadata(
        output_dir=split_output_dir,
        identifier_count=len(identifier_map) + 1,  # +1 for blank puzzle
        n_arc_puzzles=len(puzzles),
        n_aug_puzzles=total_puzzles,
        n_aug_examples=total_examples,
        metadata_filename=metadata_filename,
    )


def _save_dataset_metadata(output_dir: Path, identifier_count: int, n_arc_puzzles: int, n_aug_puzzles: int, n_aug_examples: int, metadata_filename: str) -> None:
    metadata = PuzzleDatasetMetadata(
        seq_len=ARC_MAX_GRID_SIZE * ARC_MAX_GRID_SIZE,
        vocab_size=(ARC_MAX_COLOR_VALUE + 1) + N_PADDING_TOKENS + N_EOS_TOKENS,
        pad_id=PAD_TOKEN,
        ignore_label_id=PAD_TOKEN,
        blank_identifier_id=PAD_TOKEN,
        num_puzzle_identifiers=identifier_count,
        total_groups=n_arc_puzzles,
        mean_puzzle_examples=(n_aug_examples / n_aug_puzzles if n_aug_puzzles > 0 else 0),
        sets=[SET_NAME],
    )
    with open(output_dir / f"{metadata_filename}.json", "w") as f:
        json.dump(metadata.model_dump(), f)


def _save_identifier_list(identifier_map: dict[str, int], output_path: Path) -> None:
    reverse_map = {v: k for k, v in identifier_map.items()}
    identifiers = [reverse_map.get(i, BLANK_PUZZLE_ID) for i in range(len(identifier_map) + 1)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(identifiers, f)


def convert_dataset(config: ARCDatasetBuildConfig) -> None:
    np.random.seed(config.seed)
    random.seed(config.seed)
    output_dir = Path(config.processed_dataset_dir)

    per_split_puzzle_groups = _process_arcagi_dataset(config.raw_dataset_dirs, config)
    print("Building identifier map")
    identifier_map = _build_identifier_map(per_split_puzzle_groups)
    print("Saving identifier list")
    _save_identifier_list(identifier_map, output_dir / f"{config.identifiers_filename}.json")
    for split, puzzles in per_split_puzzle_groups.items():
        _save_split_data(split, puzzles, identifier_map, output_dir, config.metadata_filename)


if __name__ == "__main__":
    from argdantic import ArgParser

    cli = ArgParser()

    @cli.command(singleton=True)
    def main(config: ARCDatasetBuildConfig):
        convert_dataset(ARCDatasetBuildConfig(**config.model_dump()))

    cli()
