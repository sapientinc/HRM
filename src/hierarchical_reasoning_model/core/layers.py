"""Neural network layers for the Hierarchical Reasoning Model.

This module implements transformer building blocks with optimizations:
- FlashAttention for efficient multi-head attention
- Rotary Position Embeddings (RoPE) for position encoding
- SwiGLU activation for feed-forward networks
- RMS normalization for stable training
- Type-casted layers for mixed-precision training

All layers support bfloat16/float32 computation with automatic type casting.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

# Try to import FlashAttention (requires CUDA)
try:
    from flash_attn_interface import flash_attn_func  # type: ignore[import]

    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    try:
        # Fallback to FlashAttention 2
        from flash_attn import flash_attn_func  # type: ignore[import]

        _FLASH_ATTN_AVAILABLE = True
    except ImportError:
        # No FlashAttention available - use pure PyTorch fallback
        _FLASH_ATTN_AVAILABLE = False

        def flash_attn_func(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            causal: bool = False,
        ) -> torch.Tensor:
            """Pure PyTorch fallback for FlashAttention.

            This is a simple implementation for CPU/environments without FlashAttention.
            It is slower and less memory-efficient than the CUDA implementation.

            Args:
                q: Query tensor (batch, seq_len, num_heads, head_dim)
                k: Key tensor (batch, seq_len, num_heads, head_dim)
                v: Value tensor (batch, seq_len, num_heads, head_dim)
                causal: Whether to apply causal masking

            Returns:
                Attention output (batch, seq_len, num_heads, head_dim)
            """
            batch, seq_len, num_heads, head_dim = q.shape

            # Transpose to (batch, num_heads, seq_len, head_dim)
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)

            # Compute attention scores
            scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / (head_dim**0.5)

            # Apply causal mask if needed
            if causal:
                mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=q.device), diagonal=1
                )
                scores = scores.masked_fill(mask.bool(), float("-inf"))

            # Compute attention weights and output
            attn = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn, v_t)

            # Transpose back to (batch, seq_len, num_heads, head_dim)
            return output.transpose(1, 2).contiguous()


from hierarchical_reasoning_model.core.common import truncated_normal_init_

#: Type alias for precomputed cosine and sine values used in RoPE
CosSin = tuple[torch.Tensor, torch.Tensor]


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round up value to the nearest multiple.

    Performs ceiling division to find the smallest multiple of `multiple`
    that is greater than or equal to `value`.

    Args:
        value: Number to round up
        multiple: Multiple to round to

    Returns:
        Smallest multiple of `multiple` that is >= `value`

    Example:
        >>> _round_up_to_multiple(100, 256)
        256
        >>> _round_up_to_multiple(300, 256)
        512
    """
    return (-(value // -multiple)) * multiple


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions for RoPE.

    Splits the last dimension in half and rotates: [x1, x2] -> [-x2, x1].
    This is a core operation in Rotary Position Embeddings.

    Args:
        x: Input tensor with shape (..., dim)

    Returns:
        Rotated tensor of same shape with second half negated and moved to front
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embeddings to query and key tensors.

    RoPE encodes position information by rotating query and key vectors in
    complex space. This provides better extrapolation to longer sequences
    compared to learned positional embeddings.

    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        k: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim)
        cos: Precomputed cosine values of shape (seq_len, head_dim)
        sin: Precomputed sine values of shape (seq_len, head_dim)

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs
    """
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    """Linear layer with automatic type casting for mixed-precision training.

    Uses truncated LeCun normal initialization for weights (scaled by 1/sqrt(in_features))
    and zero initialization for biases. Automatically casts weights and biases to
    match input dtype during forward pass.

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include bias term

    Attributes:
        weight: Weight parameter of shape (out_features, in_features)
        bias: Optional bias parameter of shape (out_features,)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            truncated_normal_init_(
                torch.empty((out_features, in_features)), std=1.0 / (in_features**0.5)
            )
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features,)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation with automatic type casting.

        Args:
            input: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features) in same dtype as input
        """
        return F.linear(
            input,
            self.weight.to(input.dtype),
            bias=self.bias.to(input.dtype) if self.bias is not None else None,
        )


class CastedEmbedding(nn.Module):
    """Embedding layer with type casting for mixed-precision training.

    Uses truncated normal initialization for embedding weights. Automatically
    casts embeddings to specified dtype during forward pass.

    Args:
        num_embeddings: Size of embedding vocabulary
        embedding_dim: Dimension of embedding vectors
        init_std: Standard deviation for weight initialization
        cast_to: Target dtype to cast embeddings to (e.g., torch.bfloat16)

    Attributes:
        cast_to: Target dtype for embeddings
        embedding_weight: Embedding weight parameter of shape
            (num_embeddings, embedding_dim)
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        init_std: float,
        cast_to: torch.dtype,
    ):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            truncated_normal_init_(
                torch.empty((num_embeddings, embedding_dim)), std=init_std
            )
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Look up embeddings and cast to target dtype.

        Args:
            input: Integer tensor of indices with shape (...)

        Returns:
            Embedded tensor of shape (..., embedding_dim) in cast_to dtype
        """
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) precomputation.

    Precomputes and caches cosine and sine values for rotary position embeddings.
    RoPE allows better extrapolation to longer sequences by encoding absolute
    position information via rotation in complex space.

    Args:
        dim: Dimension of embeddings (typically head_dim)
        max_position_embeddings: Maximum sequence length
        base: Base for inverse frequency computation (typically 10000)
        device: Device to place tensors on (None for CPU)

    Attributes:
        cos_cached: Precomputed cosine values of shape
            (max_position_embeddings, dim)
        sin_cached: Precomputed sine values of shape
            (max_position_embeddings, dim)

    References:
        RoFormer: Enhanced Transformer with Rotary Position Embedding
        https://arxiv.org/abs/2104.09864
    """

    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self) -> CosSin:
        """Return precomputed cosine and sine values.

        Returns:
            Tuple of (cos_cached, sin_cached) tensors, each of shape
            (max_position_embeddings, dim)
        """
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    """Multi-head self-attention using FlashAttention.

    Efficient attention implementation using FlashAttention 2 or 3 for memory
    and speed optimization. Supports:
    - Multi-query attention (MQA) and grouped-query attention (GQA)
    - Optional causal masking
    - RoPE positional encodings
    - Automatic dtype casting

    Args:
        hidden_size: Dimension of input hidden states
        head_dim: Dimension of each attention head
        num_heads: Number of query heads
        num_key_value_heads: Number of key/value heads (for MQA/GQA)
        causal: Whether to apply causal masking (default: False for HRM)

    Attributes:
        hidden_size: Input dimension
        head_dim: Per-head dimension
        output_size: Total output dimension (head_dim * num_heads)
        num_heads: Number of query heads
        num_key_value_heads: Number of key/value heads
        causal: Whether attention is causal
        qkv_proj: Combined query, key, value projection
        o_proj: Output projection
    """

    def __init__(
        self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=False,
        )
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply multi-head self-attention with FlashAttention.

        Args:
            cos_sin: Optional tuple of (cos, sin) for RoPE. If None, no
                positional encoding is applied
            hidden_states: Input tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Attention output of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(
            batch_size,
            seq_len,
            self.num_heads + 2 * self.num_key_value_heads,
            self.head_dim,
        )
        query = qkv[:, :, : self.num_heads]
        key = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads :]

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # flash attn
        attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)
        if isinstance(attn_output, tuple):  # fa2 and fa3 compatibility
            attn_output = attn_output[0]

        attn_output = attn_output.view(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network layer.

    Implements the SwiGLU activation function, which combines Swish (SiLU)
    activation with a gating mechanism. This has been shown to outperform
    standard ReLU and GELU in transformer models.

    The architecture is: down(SiLU(gate(x)) * up(x))

    Args:
        hidden_size: Dimension of input and output
        expansion: Expansion factor for intermediate dimension
            (typically 4.0, rounded to 2/3 * expansion for efficiency)

    Attributes:
        gate_up_proj: Combined gate and up projection
        down_proj: Down projection back to hidden_size

    References:
        GLU Variants Improve Transformer (Shazeer, 2020)
        https://arxiv.org/abs/2002.05202
    """

    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _round_up_to_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation.

        Args:
            x: Input tensor of shape (..., hidden_size)

        Returns:
            Output tensor of shape (..., hidden_size)
        """
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    """Root Mean Square Layer Normalization.

    RMSNorm normalizes by the RMS statistic rather than mean and variance,
    providing similar benefits to LayerNorm but with lower computational cost.
    Computation is done in float32 for numerical stability.

    Args:
        hidden_states: Input tensor of shape (..., hidden_size)
        variance_epsilon: Small constant for numerical stability (typically 1e-5)

    Returns:
        Normalized tensor of same shape and dtype as input

    References:
        Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
        https://arxiv.org/abs/1910.07467
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
