"""
Hierarchical Reasoning Model (HRM)

A novel recurrent neural network architecture for sequential reasoning tasks through
hierarchical processing inspired by the human brain.

This package provides the core HRM model components. For dataset processing, training,
and evaluation utilities, see the scripts/ directory.

Example usage:
    >>> from hierarchical_reasoning_model import HierarchicalReasoningModel_ACTV1Config
    >>> config = HierarchicalReasoningModel_ACTV1Config(
    ...     batch_size=32,
    ...     seq_len=81,
    ...     vocab_size=11,
    ...     num_puzzle_identifiers=1,
    ...     H_cycles=2,
    ...     L_cycles=2,
    ...     H_layers=4,
    ...     L_layers=4,
    ...     hidden_size=512,
    ...     num_heads=8,
    ...     expansion=4.0,
    ...     pos_encodings="rope",
    ...     halt_max_steps=16,
    ...     halt_exploration_prob=0.1,
    ... )
    >>> model = HierarchicalReasoningModel_ACTV1(config.model_dump())
    >>> loss_head = ACTLossHead(model, loss_type="softmax")
"""

__version__ = "0.1.0"

# Core model exports
from hierarchical_reasoning_model.core import (
    ACTLossHead,
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config,
)

__all__ = [
    "__version__",
    # Core models
    "HierarchicalReasoningModel_ACTV1",
    "HierarchicalReasoningModel_ACTV1Config",
    "ACTLossHead",
]
