# ==============================================================================
# CELL 1: TITLE AND INTRODUCTION (Markdown Cell)
# ==============================================================================

"""
# HRM: Hierarchical Reasoning Model - Inference Demo

This notebook demonstrates how to download and run the Hierarchical Reasoning Model (HRM) from Sapient Inc. for inference on the Abstraction and Reasoning Corpus (ARC).

### What is HRM?
HRM is a novel recurrent architecture inspired by the human brain's hierarchical processing. It's designed for complex reasoning tasks and can solve them in a single forward pass without step-by-step supervision. With only 27 million parameters, it has shown remarkable performance on benchmarks like ARC, Sudoku, and maze-solving.

### What is ARC?
The Abstraction and Reasoning Corpus (ARC) is a benchmark designed to measure artificial general intelligence. It consists of a series of puzzles, where the model must infer an underlying pattern from a few examples and apply it to a new input grid.

### How to Use the Visualizer
The official HRM repository comes with an interactive `puzzle_visualizer.html`. After running the dataset generation step below, you can upload the generated data folder (e.g., `data/arc-2-aug-1000`) to this visualizer to explore the puzzles.
"""

# ==============================================================================
# CELL 2: SETUP - CLONE REPO AND INSTALL DEPENDENCIES (Code Cell)
# ==============================================================================

import os

# Clone the repository
if not os.path.exists("hrm"):
    print("Cloning the HRM repository...")
    !git clone https://github.com/sapient-inc/HRM.git hrm
else:
    print("HRM repository already exists.")

# Navigate into the repo directory
os.chdir("hrm")

# Install required packages
print("Installing dependencies...")
!pip install -r requirements.txt
# Install torch with the correct CUDA version if available, otherwise it will use CPU
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ==============================================================================
# CELL 3: DOWNLOAD HUGGING FACE CHECKPOINT (Code Cell)
# ==============================================================================

import subprocess

def hf_download(repo_id, local_dir):
    """Downloads a Hugging Face repo, skipping large LFS files if needed."""
    if os.path.exists(local_dir):
        print(f"Directory {local_dir} already exists. Skipping download.")
        return
    print(f"Cloning {repo_id} to {local_dir}...")
    # Clone the repo without pulling LFS files initially
    subprocess.run(["git", "clone", f"https://huggingface.co/{repo_id}", local_dir], check=True)
    os.chdir(local_dir)
    # Download the large checkpoint file specifically
    print("Downloading the large checkpoint file via LFS...")
    subprocess.run(["git", "lfs", "pull", "--include", "checkpoint"], check=True)
    os.chdir("..")
    print("Download complete.")

# Define repo and local directory
repo_id = "sapientinc/HRM-checkpoint-ARC-2"
local_dir = "HRM-checkpoint-ARC-2"

# Download the model checkpoint
hf_download(repo_id, local_dir)


# ==============================================================================
# CELL 4: DATASET GENERATION (Markdown Cell)
# ==============================================================================

"""
### Generating the ARC-AGI Dataset

The following cell runs the `build_arc_dataset.py` script from the repository.
This script performs several actions:
1. It initializes the git submodules, which includes the official ARC-AGI repository.
2. It processes the raw JSON files from the ARC dataset.
3. It creates a processed version of the dataset in the `data/arc-2-aug-1000` directory, which is what our model will use for inference.
"""

# ==============================================================================
# CELL 5: RUN DATASET BUILD SCRIPT (Code Cell)
# ==============================================================================

# Initialize submodules to get the dataset source
!git submodule update --init --recursive

# Run the build script for the ARC-2 dataset
# This specific command generates the dataset version the checkpoint was trained on.
!python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000

print("Dataset generation complete.")


# ==============================================================================
# CELL 6: INFERENCE SETUP (Code Cell)
# ==============================================================================

import yaml
import torch
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

# The repository's code is not packaged, so we add it to the Python path
import sys
sys.path.append(os.path.abspath('.'))

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1
from puzzle_dataset import PuzzleDataset, PuzzleDatasetMetadata

# --- Configuration ---
CHECKPOINT_DIR = "HRM-checkpoint-ARC-2"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "checkpoint")
CONFIG_PATH = os.path.join(CHECKPOINT_DIR, "all_config.yaml")
DATA_DIR = "data/arc-2-aug-1000"

# --- Load Model Configuration ---
print("Loading model configuration...")
with open(CONFIG_PATH, "r") as f:
    config_yaml = yaml.safe_load(f)

# The model config is nested under the 'arch' key
model_config = config_yaml['arch']

# Manually add necessary parameters not in the checkpoint's yaml
# These are typically available during training setup but needed for standalone inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# The PuzzleDataset provides this metadata, which we need for model instantiation
train_metadata = PuzzleDatasetMetadata.from_data_path(os.path.join(DATA_DIR, "train"))

model_config['batch_size'] = 1 # Inference on a single sample
model_config['seq_len'] = train_metadata.seq_len
model_config['num_puzzle_identifiers'] = train_metadata.num_puzzle_identifiers
model_config['vocab_size'] = train_metadata.vocab_size
model_config['forward_dtype'] = "float32" # Use float32 for CPU inference

# --- Instantiate Model ---
print("Instantiating the model...")
model = HierarchicalReasoningModel_ACTV1(model_config)
model.to(device)
model.eval()

# --- Load Checkpoint ---
print("Loading checkpoint...")
# Use map_location for CPU or GPU automatically
state_dict = torch.load(CHECKPOINT_PATH, map_location=device)

# The model might be saved with a '_orig_mod.' prefix if compiled
# We need to strip this prefix to load the state dict correctly.
if list(state_dict.keys())[0].startswith('_orig_mod.'):
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

model.load_state_dict(state_dict, assign=True)

print("Model loaded successfully.")


# ==============================================================================
# CELL 7: RUN INFERENCE ON A SAMPLE (Code Cell)
# ==============================================================================

# --- Load Dataset ---
# We'll use the 'test' set for inference
test_dataset = PuzzleDataset(os.path.join(DATA_DIR, "test"), test_set_mode=True)

# --- Select a Sample ---
# Let's pick the first puzzle in the test set
sample_idx = 0
sample = test_dataset[sample_idx]

# The sample contains multiple 'train' examples and one 'test' input
# We will run inference on the first test input
# The model expects a batch, so we add a batch dimension (unsqueeze)
puzz_id = torch.tensor([sample['puzzle_identifier']], device=device)
inputs = torch.tensor(sample['test_inputs'][0], device=device).unsqueeze(0)

# The model processes the puzzle in "cycles" or "steps".
# For ARC evaluation, the model is run for a fixed number of steps (defined in its config).
num_steps = model.config.halt_max_steps
print(f"Running inference for {num_steps} steps...")

# --- Run the Model ---
with torch.no_grad():
    # Initialize the model's recurrent state (called a 'carry')
    batch = {"inputs": inputs, "puzzle_identifiers": puzz_id}
    carry = model.initial_carry(batch)

    for step in range(num_steps):
        carry, outputs = model(carry, batch)
        # On the final step, we get the logits we need
        if step == num_steps - 1:
            final_logits = outputs['logits']

# --- Process the Output ---
# The output logits have dimensions [batch_size, seq_len, vocab_size]
# We get the most likely token for each position in the sequence
predicted_sequence = torch.argmax(final_logits, dim=-1).squeeze(0)

# Reshape the sequence back into a grid
grid_size = int(np.sqrt(inputs.shape[1]))
predicted_grid = predicted_sequence.reshape(grid_size, grid_size).cpu().numpy()
input_grid = inputs.squeeze(0).reshape(grid_size, grid_size).cpu().numpy()

# The ground truth label is also in the sample
label_grid = sample['test_labels'][0]

print("Inference complete.")


# ==============================================================================
# CELL 8: VISUALIZE RESULTS (Code Cell)
# ==============================================================================

def plot_grid(ax, grid, title):
    """Helper function to plot a grid."""
    cmap = plt.get_cmap('viridis', 10) # Color map for 0-9 values
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

plot_grid(axes[0], input_grid, "Test Input")
plot_grid(axes[1], predicted_grid, "Model Prediction")
plot_grid(axes[2], label_grid, "Ground Truth")

plt.tight_layout()
plt.show()

# Check if the prediction is correct
is_correct = np.array_equal(predicted_grid, label_grid)
print(f"\nPrediction Correct: {is_correct}")

# Move back to the root directory
os.chdir("..")
