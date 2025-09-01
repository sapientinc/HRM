# HRM for Vision Tasks

This directory contains an adaptation of the Hierarchical Reasoning Model (HRM) for computer vision tasks, specifically CIFAR-10 and CIFAR-100 classification.

## Overview

The vision HRM adapts the original HRM architecture for image classification by:

1. **Patch-based Processing**: Converts images into sequences of patches (similar to Vision Transformers)
2. **Hierarchical Reasoning**: Uses the same H-level and L-level reasoning modules as the original HRM
3. **Classification Head**: Adds a classification head that uses the class token for final predictions
4. **Simplified ACT**: Uses a single forward pass (no adaptive computation time needed for vision)

## Architecture

### Key Components

- **VisionPatchEmbedding**: Converts 32x32 CIFAR images into 4x4 patches, then to embeddings
- **HierarchicalReasoningModel_VisionV1**: Main model with H-level and L-level reasoning
- **VisionClassificationHead**: Classification head using class token
- **VisionClassificationLossHead**: Loss computation for classification

### Model Configuration

The vision HRM uses these key parameters:
- `hidden_size: 512` - Hidden dimension
- `num_heads: 8` - Number of attention heads
- `H_layers: 4, L_layers: 4` - Number of layers in each reasoning level
- `H_cycles: 2, L_cycles: 2` - Number of reasoning cycles
- `patch_size: 4` - Size of image patches
- `num_classes: 10/100` - Number of classes (CIFAR-10/100)

## Usage

### Quick Start

1. **Build the dataset**:
```bash
python dataset/build_cifar_dataset.py --dataset_name CIFAR10
```

2. **Train the model**:
```bash
python pretrain_vision.py --config-name cfg_vision_pretrain
```

3. **Or use the convenience script**:
```bash
python train_cifar.py --dataset CIFAR10 --epochs 50
```

### Training Options

```bash
# Train on CIFAR-100
python train_cifar.py --dataset CIFAR100 --epochs 100

# Custom hyperparameters
python train_cifar.py --dataset CIFAR10 --epochs 50 --batch-size 128 --lr 2e-4

# Skip dataset building (if already built)
python train_cifar.py --skip-dataset-build
```

### Configuration Files

- `config/arch/hrm_vision_v1.yaml` - Model architecture configuration
- `config/cfg_vision_pretrain.yaml` - Training configuration

## Dataset Format

The CIFAR dataset is converted to the HRM format:

1. **Images** → **Patches**: 32x32 images divided into 4x4 patches (64 patches total)
2. **Patches** → **Sequence**: Each patch flattened to 48 values (4×4×3 RGB)
3. **Sequence Length**: 3072 tokens (64 patches × 48 values)
4. **Vocabulary**: 256 tokens (0-255 for pixel values)

## Key Differences from Original HRM

1. **Input Processing**: Images → patches → embeddings instead of token embeddings
2. **No Puzzle Embeddings**: Simplified for vision tasks
3. **Single Forward Pass**: No adaptive computation time
4. **Classification Output**: Class probabilities instead of sequence generation
5. **Data Augmentation**: Dihedral transforms and color augmentation

## Expected Performance

The vision HRM should achieve:
- **CIFAR-10**: ~85-90% accuracy (comparable to simple CNNs)
- **CIFAR-100**: ~60-70% accuracy

Note: This is primarily a proof-of-concept adaptation. For production vision tasks, specialized architectures like ResNet or Vision Transformers would be more appropriate.

## Files Structure

```
HRM/
├── dataset/
│   └── build_cifar_dataset.py          # CIFAR dataset builder
├── models/
│   ├── hrm/
│   │   └── hrm_vision_v1.py            # Vision HRM architecture
│   └── vision_losses.py                # Vision loss functions
├── config/
│   ├── arch/
│   │   └── hrm_vision_v1.yaml          # Model config
│   └── cfg_vision_pretrain.yaml        # Training config
├── pretrain_vision.py                  # Vision training script
├── train_cifar.py                      # Convenience training script
└── README_VISION.md                    # This file
```

## Dependencies

Make sure you have all the required dependencies from the original HRM:
- PyTorch with CUDA support
- torchvision (for CIFAR dataset)
- wandb (for logging)
- hydra-core (for configuration)
- All other dependencies from `requirements.txt`

## Troubleshooting

1. **CUDA out of memory**: Reduce `global_batch_size` in config
2. **Dataset not found**: Run the dataset builder first
3. **Import errors**: Make sure you're in the HRM directory and all dependencies are installed

## Future Improvements

- Add more sophisticated data augmentation
- Implement attention visualization
- Add support for other vision datasets (ImageNet, etc.)
- Experiment with different patch sizes and architectures
- Add multi-scale reasoning for different image resolutions
