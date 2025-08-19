"""Command-line utility for HuggingFace dataset conversion.

To run with default settings:
python convert_dataset.py

You can override any of the defaults by providing them as command-line arguments, for example:
python convert_dataset.py \
    --hf_dataset "imdb" \
    --hf_split "test" \
    --tokenizer_name "bert-base-uncased" \
    --max_seq_len 256 \
    --output_dir "./data/imdb-converted/test" \
    --set_name "imdb_test"

To see all available options and their descriptions::
python convert_dataset.py --help

Adapted from johnnyZeppelin/HRM-Idioma/blob/main/convert_hf_to_puzzle_format.py
"""

import os
import json
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def main(args):
    """Main function to process the dataset and convert it to numpy format."""
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. Load Tokenizer ---
    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    # Set pad token if it's not already set
    if tokenizer.pad_token_id is None:
        print(f"Tokenizer does not have a pad_token_id. Using provided pad_id: {args.pad_id}")
        tokenizer.pad_token_id = args.pad_id

    # --- 2. Load Dataset ---
    print(f"Loading dataset: {args.hf_dataset} (split: {args.hf_split})")
    try:
        dataset = load_dataset(args.hf_dataset, split=args.hf_split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Ensure the dataset has a 'text' column for processing
    if "text" not in dataset.column_names:
        print(f"Error: Dataset '{args.hf_dataset}' does not have a 'text' column.")
        # Attempt to use the first column if 'text' is not found
        if dataset.column_names:
            first_col = dataset.column_names[0]
            print(f"Warning: Using the first column '{first_col}' as the text source.")
            dataset = dataset.rename_column(first_col, 'text')
        else:
            print("Error: Dataset has no columns.")
            return

    # --- 3. Process and Tokenize Data ---
    print("Processing documents and tokenizing...")
    all_inputs = []
    all_labels = []
    puzzle_indices = [0]  # Pointer to where each document's examples start
    puzzle_identifiers = []
    group_indices = [0]   # Only 1 group in this setup
    example_counter = 0
    puzzle_counter = 0

    for doc in dataset["text"]:
        # Skip empty or whitespace-only documents
        if not doc or not doc.strip():
            continue

        # Tokenize the entire document without adding special tokens yet
        tokens = tokenizer.encode(doc, add_special_tokens=False)

        if not tokens:
            continue

        # Break the tokenized document into fixed-length chunks
        for i in range(0, len(tokens), args.max_seq_len):
            chunk = tokens[i:i + args.max_seq_len]

            # Pad the chunk if it's shorter than max_seq_len
            if len(chunk) < args.max_seq_len:
                padding = [tokenizer.pad_token_id] * (args.max_seq_len - len(chunk))
                chunk.extend(padding)

            # For causal language modeling, labels are the same as inputs
            labels = chunk.copy()

            all_inputs.append(chunk)
            all_labels.append(labels)
            puzzle_identifiers.append(puzzle_counter)
            example_counter += 1

        puzzle_counter += 1
        puzzle_indices.append(example_counter)

    group_indices.append(puzzle_counter)

    # --- 4. Convert to Numpy and Save ---
    print("Converting lists to numpy arrays...")
    all_inputs = np.array(all_inputs, dtype=np.int32)
    all_labels = np.array(all_labels, dtype=np.int32)
    puzzle_indices = np.array(puzzle_indices, dtype=np.int64)
    puzzle_identifiers = np.array(puzzle_identifiers, dtype=np.int32)
    group_indices = np.array(group_indices, dtype=np.int64)

    print(f"Saving processed files to {args.output_dir}...")
    np.save(os.path.join(args.output_dir, f"{args.set_name}__inputs.npy"), all_inputs)
    np.save(os.path.join(args.output_dir, f"{args.set_name}__labels.npy"), all_labels)
    np.save(os.path.join(args.output_dir, f"{args.set_name}__puzzle_indices.npy"), puzzle_indices)
    np.save(os.path.join(args.output_dir, f"{args.set_name}__puzzle_identifiers.npy"), puzzle_identifiers)
    np.save(os.path.join(args.output_dir, f"{args.set_name}__group_indices.npy"), group_indices)

    # --- 5. Save Metadata ---
    metadata = {
        "sets": [args.set_name],
        "pad_id": tokenizer.pad_token_id,
        "blank_identifier_id": 0,  # Assuming a default, can be parameterized if needed
        "ignore_label_id": args.ignore_label_id
    }

    metadata_path = os.path.join(args.output_dir, "dataset.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Conversion complete! Saved to {args.output_dir}")
    print(f"   - Examples: {all_inputs.shape[0]}")
    print(f"   - Puzzles (Documents): {len(puzzle_indices)-1}")
    print(f"   - Groups: {len(group_indices)-1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Hugging Face datasets to a specific numpy format for NLP tasks.")

    # --- Add arguments based on the original CONFIG section ---
    parser.add_argument("--hf_dataset", type=str, default="wikitext", help="Name of the Hugging Face dataset to use (e.g., 'wikitext', 'openwebtext').")
    parser.add_argument("--hf_config", type=str, default="wikitext-2-raw-v1", help="Configuration for the Hugging Face dataset (e.g., 'wikitext-2-raw-v1').")
    parser.add_argument("--hf_split", type=str, default="train", help="Dataset split to use (e.g., 'train', 'validation', 'test').")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Name of the Hugging Face tokenizer.")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Maximum sequence length for model inputs.")
    parser.add_argument("--pad_id", type=int, default=50256, help="Padding token ID to use if the tokenizer doesn't have one.")
    parser.add_argument("--ignore_label_id", type=int, default=-100, help="Standard PyTorch loss ignore index for labels.")
    parser.add_argument("--output_dir", type=str, default="./data/nlp-converted/train", help="Directory to save the output .npy files and metadata.")
    parser.add_argument("--set_name", type=str, default="all", help="Name of the output set (used in file naming).")

    # Parse the arguments and run the main function
    args = parser.parse_args()
    main(args)
