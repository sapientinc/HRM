from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.layers import Attention, SwiGLU, RotaryEmbedding
from models.common import CastedEmbedding, CastedLinear, CastedSparseEmbedding
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed


@dataclass
class HierarchicalReasoningModel_VisionV1Config:
    # Model dimensions
    hidden_size: int
    num_heads: int
    expansion: int
    
    # Architecture
    H_layers: int
    L_layers: int
    H_cycles: int
    L_cycles: int
    
    # Vision-specific
    vocab_size: int
    seq_len: int
    num_classes: int
    patch_size: int
    image_size: int
    
    # Puzzle embeddings
    num_puzzle_identifiers: int
    puzzle_emb_ndim: int
    
    # ACT parameters
    halt_exploration_prob: float
    halt_max_steps: int
    
    # Training
    batch_size: int
    forward_dtype: str = "float16"
    
    # Positional encoding
    pos_encodings: str = "rope"
    rope_theta: float = 10000.0


class VisionPatchEmbedding(nn.Module):
    """Convert image patches to embeddings."""
    
    def __init__(self, config: HierarchicalReasoningModel_VisionV1Config):
        super().__init__()
        self.config = config
        
        # Patch embedding layer
        patch_dim = config.patch_size * config.patch_size * 3  # RGB channels
        self.patch_embed = CastedLinear(patch_dim, config.hidden_size, bias=True)
        
        # Positional embedding for patches
        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embed = CastedEmbedding(num_patches, config.hidden_size, init_std=0.02)
        
        # Class token embedding (for classification)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        
    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, seq_len) where seq_len = num_patches * patch_dim
        batch_size = x.shape[0]
        
        # Reshape to patches
        patch_dim = self.config.patch_size * self.config.patch_size * 3
        num_patches = (self.config.image_size // self.config.patch_size) ** 2
        
        # Truncate or pad to exact patch size
        if x.shape[1] > num_patches * patch_dim:
            x = x[:, :num_patches * patch_dim]
        elif x.shape[1] < num_patches * patch_dim:
            # Pad with zeros
            pad_size = num_patches * patch_dim - x.shape[1]
            x = torch.cat([x, torch.zeros(batch_size, pad_size, device=x.device, dtype=x.dtype)], dim=1)
        
        # Reshape to patches
        x = x.view(batch_size, num_patches, patch_dim)
        
        # Normalize pixel values to [0, 1] range
        x = x.float() / 255.0
        
        # Project patches to embeddings
        x = self.patch_embed(x)
        
        # Add positional embeddings
        pos_ids = torch.arange(num_patches, device=x.device).unsqueeze(0)
        x = x + self.pos_embed(pos_ids)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        return x


class VisionClassificationHead(nn.Module):
    """Classification head for vision tasks."""
    
    def __init__(self, config: HierarchicalReasoningModel_VisionV1Config):
        super().__init__()
        self.config = config
        
        # Global average pooling + classification head
        self.norm = nn.LayerNorm(config.hidden_size)
        self.classifier = CastedLinear(config.hidden_size, config.num_classes, bias=True)
        
    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch_size, seq_len, hidden_size)
        # Use class token (first token) for classification
        cls_token = x[:, 0, :]  # (batch_size, hidden_size)
        
        # Normalize and classify
        cls_token = self.norm(cls_token)
        logits = self.classifier(cls_token)
        
        return logits


class HierarchicalReasoningModel_VisionV1Block(nn.Module):
    """Single block for vision HRM."""
    
    def __init__(self, config: HierarchicalReasoningModel_VisionV1Config):
        super().__init__()
        self.config = config
        
        # Attention
        self.attention = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            forward_dtype=config.forward_dtype
        )
        
        # MLP
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
            forward_dtype=config.forward_dtype
        )
        
        # Layer norms
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.mlp_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        # Self-attention
        residual = x
        x = self.attn_norm(x)
        x = self.attention(x, attention_mask=attention_mask)
        x = residual + x
        
        # MLP
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class HierarchicalReasoningModel_VisionV1ReasoningModule(nn.Module):
    """Reasoning module for vision tasks."""
    
    def __init__(self, layers: list):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        
    def forward(self, hidden_states: Tensor, input_injection: Tensor, **kwargs) -> Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        
        # Apply layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        return hidden_states


@dataclass
class HierarchicalReasoningModel_VisionV1InnerCarry:
    """Carry state for vision HRM inner model."""
    H_hidden: Tensor
    L_hidden: Tensor
    step: int


class HierarchicalReasoningModel_VisionV1_Inner(nn.Module):
    """Inner model for vision HRM."""
    
    def __init__(self, config: HierarchicalReasoningModel_VisionV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        
        # Vision-specific components
        self.patch_embedding = VisionPatchEmbedding(config)
        self.classification_head = VisionClassificationHead(config)
        
        # Positional encoding
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + 1,  # +1 for class token
                base=self.config.rope_theta
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + 1,  # +1 for class token
                self.config.hidden_size,
                init_std=0.02,
                cast_to=self.forward_dtype
            )
        else:
            raise NotImplementedError()
        
        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_VisionV1ReasoningModule(
            layers=[HierarchicalReasoningModel_VisionV1Block(self.config) for _ in range(self.config.H_layers)]
        )
        self.L_level = HierarchicalReasoningModel_VisionV1ReasoningModule(
            layers=[HierarchicalReasoningModel_VisionV1Block(self.config) for _ in range(self.config.L_layers)]
        )
        
        # Projection layers
        self.H_proj = CastedLinear(self.config.hidden_size, self.config.hidden_size, bias=False)
        self.L_proj = CastedLinear(self.config.hidden_size, self.config.hidden_size, bias=False)
        
    def empty_carry(self, batch_size: int) -> HierarchicalReasoningModel_VisionV1InnerCarry:
        """Create empty carry state."""
        device = next(self.parameters()).device
        dtype = self.forward_dtype
        
        return HierarchicalReasoningModel_VisionV1InnerCarry(
            H_hidden=torch.zeros((batch_size, self.config.seq_len + 1, self.config.hidden_size), device=device, dtype=dtype),
            L_hidden=torch.zeros((batch_size, self.config.seq_len + 1, self.config.hidden_size), device=device, dtype=dtype),
            step=0
        )
    
    def forward(self, carry: HierarchicalReasoningModel_VisionV1InnerCarry, batch: Dict[str, Tensor]) -> Tuple[HierarchicalReasoningModel_VisionV1InnerCarry, Tensor, Tensor]:
        """Forward pass for vision HRM."""
        # Get input embeddings
        input_embeds = self.patch_embedding(batch["inputs"])
        
        # Apply positional encoding
        if self.config.pos_encodings == "rope":
            cos, sin = self.rotary_emb(input_embeds)
            # Apply rotary embedding to attention layers
            # (This would be handled in the attention layers themselves)
        elif self.config.pos_encodings == "learned":
            pos_ids = torch.arange(input_embeds.shape[1], device=input_embeds.device).unsqueeze(0)
            input_embeds = input_embeds + self.embed_pos(pos_ids)
        
        # Hierarchical reasoning
        for _ in range(self.config.H_cycles):
            # High-level reasoning
            carry.H_hidden = self.H_level(carry.H_hidden, input_embeds)
            
            # Low-level reasoning
            for _ in range(self.config.L_cycles):
                carry.L_hidden = self.L_level(carry.L_hidden, self.H_proj(carry.H_hidden))
        
        # Classification
        logits = self.classification_head(carry.L_hidden)
        
        # Update step
        carry.step += 1
        
        return carry, logits, torch.tensor(0.0, device=logits.device)  # Dummy halt probability


@dataclass
class HierarchicalReasoningModel_VisionV1Carry:
    """Carry state for vision HRM."""
    inner_carry: HierarchicalReasoningModel_VisionV1InnerCarry
    steps: Tensor
    halted: Tensor
    current_data: Dict[str, Tensor]


class HierarchicalReasoningModel_VisionV1(nn.Module):
    """Vision HRM with ACT wrapper."""
    
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_VisionV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_VisionV1_Inner(self.config)
    
    def initial_carry(self, batch: Dict[str, Tensor]):
        """Initialize carry state."""
        batch_size = batch["inputs"].shape[0]
        
        return HierarchicalReasoningModel_VisionV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
    
    def forward(self, carry: HierarchicalReasoningModel_VisionV1Carry, batch: Dict[str, Tensor], return_keys: list = None) -> Tuple[HierarchicalReasoningModel_VisionV1Carry, Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor], bool]:
        """Forward pass for vision HRM."""
        # Update current data
        carry.current_data = batch
        
        # Forward through inner model
        carry.inner_carry, logits, halt_prob = self.inner(carry.inner_carry, batch)
        
        # Compute loss and metrics
        labels = batch["labels"]
        if labels.dim() > 1:
            # If labels are sequences, use the first token (class token position)
            labels = labels[:, 0]
        
        loss = F.cross_entropy(logits, labels.long())
        
        # Compute accuracy
        preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == labels.long()).float().mean()
        
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "count": torch.tensor(1.0, device=loss.device)
        }
        
        # Return predictions
        preds_dict = {"logits": logits, "predictions": preds}
        
        # Always halt for vision tasks (single forward pass)
        all_finish = True
        
        return carry, metrics, preds_dict, {}, all_finish
