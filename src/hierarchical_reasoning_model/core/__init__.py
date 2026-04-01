"""Core neural network modules for the Hierarchical Reasoning Model."""

from hierarchical_reasoning_model.core.embeddings import (
    CastedSparseEmbedding,
    CastedSparseEmbeddingSignSGD_Distributed,
)
from hierarchical_reasoning_model.core.layers import (
    Attention,
    CastedEmbedding,
    CastedLinear,
    RotaryEmbedding,
    SwiGLU,
    rms_norm,
)
from hierarchical_reasoning_model.core.losses import (
    IGNORE_LABEL_ID,
    ACTLossHead,
    softmax_cross_entropy,
    stablemax_cross_entropy,
)
from hierarchical_reasoning_model.core.model import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Carry,
    HierarchicalReasoningModel_ACTV1Config,
    HierarchicalReasoningModel_ACTV1InnerCarry,
)

__all__ = [
    # Model
    "HierarchicalReasoningModel_ACTV1",
    "HierarchicalReasoningModel_ACTV1Config",
    "HierarchicalReasoningModel_ACTV1Carry",
    "HierarchicalReasoningModel_ACTV1InnerCarry",
    # Layers
    "Attention",
    "SwiGLU",
    "RotaryEmbedding",
    "CastedLinear",
    "CastedEmbedding",
    "rms_norm",
    # Embeddings
    "CastedSparseEmbedding",
    "CastedSparseEmbeddingSignSGD_Distributed",
    # Losses
    "ACTLossHead",
    "stablemax_cross_entropy",
    "softmax_cross_entropy",
    "IGNORE_LABEL_ID",
]
