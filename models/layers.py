from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.common import trunc_normal_init_

# --- Flash Attention Fallback --- 
# models/layers.py

# --- Flash Attention Fallback ---
# Try to import flash_attn, otherwise fallback to PyTorch's built-in attention
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        # PyTorch 2.0 and later has a built-in, efficient attention implementation
        def flash_attn_func(q, k, v, causal=False):
            # The built-in function expects B, H, S, D format
            # Our input is B, S, H, D, so we need to transpose
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
            # Transpose back to B, S, H, D
            return output.transpose(1, 2)
        HAS_FLASH_ATTN = True
        print("Warning: flash-attn not found. Falling back to PyTorch's built-in scaled_dot_product_attention.")
    else:
        HAS_FLASH_ATTN = False
        print("Warning: flash-attn not found and PyTorch version is < 2.0. Using a basic attention implementation.")

CosSin = Tuple[torch.Tensor, torch.Tensor]

def _find_multiple(a, b):
    return (-(a // -b)) * b

def rotate_half(x: torch.Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5)))
        self.bias = nn.Parameter(torch.zeros((out_features, ))) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)

class CastedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.embedding_weight = nn.Parameter(trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std))
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached

class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if not HAS_FLASH_ATTN:
            # Fallback to a basic attention implementation if no efficient version is available
            # B, S, H, D -> B, H, S, D
            query, key, value = [x.transpose(1, 2) for x in (query, key, value)]
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.causal:
                mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device), diagonal=1)
                scores.masked_fill_(mask, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1).to(value.dtype)
            attn_output = torch.matmul(attn_weights, value) # (B, H, S, D)
            attn_output = attn_output.transpose(1, 2).contiguous() # (B, S, H, D)
        else:
            attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)

        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)
        return self.o_proj(attn_output)

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)