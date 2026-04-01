"""Tests for transformer layers and components."""

import pytest
import torch

from hierarchical_reasoning_model.core.layers import (
    Attention,
    CastedEmbedding,
    CastedLinear,
    RotaryEmbedding,
    SwiGLU,
    _round_up_to_multiple,
    apply_rotary_pos_emb,
    rms_norm,
    rotate_half,
)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_find_multiple_exact(self):
        """Test _round_up_to_multiple with exact multiple."""
        assert _round_up_to_multiple(10, 5) == 10
        assert _round_up_to_multiple(20, 10) == 20

    def test_find_multiple_round_up(self):
        """Test _round_up_to_multiple rounds up."""
        assert _round_up_to_multiple(11, 5) == 15
        assert _round_up_to_multiple(21, 10) == 30
        assert _round_up_to_multiple(1, 256) == 256

    def test_find_multiple_edge_cases(self):
        """Test edge cases."""
        assert _round_up_to_multiple(0, 5) == 0
        assert _round_up_to_multiple(256, 256) == 256

    def test_rotate_half_shape(self):
        """Test rotate_half preserves shape."""
        x = torch.randn(2, 4, 8)
        result = rotate_half(x)
        assert result.shape == x.shape

    def test_rotate_half_computation(self):
        """Test rotate_half correctly rotates."""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = rotate_half(x)

        # Should be [-3, -4, 1, 2]
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        assert torch.allclose(result, expected)

    def test_apply_rotary_pos_emb_shapes(self):
        """Test RoPE preserves shapes."""
        batch, seq_len, num_heads, head_dim = 2, 10, 4, 8
        q = torch.randn(batch, seq_len, num_heads, head_dim)
        k = torch.randn(batch, seq_len, num_heads, head_dim)
        cos = torch.randn(seq_len, head_dim)
        sin = torch.randn(seq_len, head_dim)

        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)

        assert q_embed.shape == q.shape
        assert k_embed.shape == k.shape

    def test_apply_rotary_pos_emb_dtype(self):
        """Test RoPE preserves dtype."""
        q = torch.randn(2, 4, 2, 8, dtype=torch.float32)
        k = torch.randn(2, 4, 2, 8, dtype=torch.float32)
        cos = torch.randn(4, 8, dtype=torch.float16)
        sin = torch.randn(4, 8, dtype=torch.float16)

        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)

        # Should return to original dtype
        assert q_embed.dtype == torch.float32
        assert k_embed.dtype == torch.float32


class TestCastedLinear:
    """Tests for CastedLinear layer."""

    def test_initialization(self):
        """Test layer initialization."""
        layer = CastedLinear(10, 20, bias=True)

        assert layer.weight.shape == (20, 10)
        assert layer.bias is not None
        assert layer.bias.shape == (20,)

    def test_initialization_no_bias(self):
        """Test initialization without bias."""
        layer = CastedLinear(10, 20, bias=False)

        assert layer.weight.shape == (20, 10)
        assert layer.bias is None

    def test_forward_shape(self):
        """Test forward pass output shape."""
        layer = CastedLinear(10, 20, bias=True)
        x = torch.randn(5, 10)

        output = layer(x)

        assert output.shape == (5, 20)

    def test_forward_dtype_casting(self):
        """Test automatic dtype casting."""
        layer = CastedLinear(10, 20, bias=True)
        x = torch.randn(5, 10, dtype=torch.float16)

        output = layer(x)

        # Output should match input dtype
        assert output.dtype == torch.float16


class TestCastedEmbedding:
    """Tests for CastedEmbedding layer."""

    def test_initialization(self):
        """Test embedding initialization."""
        embedding = CastedEmbedding(100, 64, init_std=0.02, cast_to=torch.float32)

        assert embedding.embedding_weight.shape == (100, 64)
        assert embedding.cast_to == torch.float32

    def test_forward_shape(self):
        """Test forward pass output shape."""
        embedding = CastedEmbedding(100, 64, init_std=0.02, cast_to=torch.float32)
        indices = torch.randint(0, 100, (5, 10))

        output = embedding(indices)

        assert output.shape == (5, 10, 64)

    def test_forward_dtype(self):
        """Test output dtype matches cast_to."""
        embedding = CastedEmbedding(100, 64, init_std=0.02, cast_to=torch.bfloat16)
        indices = torch.randint(0, 100, (5, 10))

        output = embedding(indices)

        assert output.dtype == torch.bfloat16


class TestRotaryEmbedding:
    """Tests for RotaryEmbedding."""

    def test_initialization(self):
        """Test RoPE initialization."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512, base=10000)

        assert rope.cos_cached.shape == (512, 64)
        assert rope.sin_cached.shape == (512, 64)

    def test_forward_returns_cached_values(self):
        """Test forward returns precomputed values."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512, base=10000)

        cos, sin = rope()

        assert torch.equal(cos, rope.cos_cached)
        assert torch.equal(sin, rope.sin_cached)

    def test_values_in_valid_range(self):
        """Test cos/sin values are in [-1, 1]."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512, base=10000)

        cos, sin = rope()

        assert torch.all(cos >= -1.0) and torch.all(cos <= 1.0)
        assert torch.all(sin >= -1.0) and torch.all(sin <= 1.0)


class TestAttention:
    """Tests for Attention layer."""

    @pytest.fixture
    def attention_layer(self):
        """Create attention layer for testing."""
        return Attention(
            hidden_size=128,
            head_dim=32,
            num_heads=4,
            num_key_value_heads=4,
            causal=False,
        )

    def test_initialization(self, attention_layer):
        """Test attention initialization."""
        assert attention_layer.hidden_size == 128
        assert attention_layer.head_dim == 32
        assert attention_layer.num_heads == 4
        assert attention_layer.output_size == 128

    def test_forward_shape(self, attention_layer):
        """Test forward pass output shape."""
        batch, seq_len = 2, 10
        hidden_states = torch.randn(batch, seq_len, 128)

        output = attention_layer(cos_sin=None, hidden_states=hidden_states)

        assert output.shape == (batch, seq_len, 128)

    def test_forward_with_rope(self, attention_layer):
        """Test forward with RoPE positional encoding."""
        batch, seq_len = 2, 10
        hidden_states = torch.randn(batch, seq_len, 128)

        rope = RotaryEmbedding(dim=32, max_position_embeddings=10, base=10000)
        cos_sin = rope()

        output = attention_layer(cos_sin=cos_sin, hidden_states=hidden_states)

        assert output.shape == (batch, seq_len, 128)

    def test_causal_attention(self):
        """Test causal attention masking."""
        attention = Attention(
            hidden_size=64,
            head_dim=16,
            num_heads=4,
            num_key_value_heads=4,
            causal=True,
        )

        batch, seq_len = 2, 8
        hidden_states = torch.randn(batch, seq_len, 64)

        output = attention(cos_sin=None, hidden_states=hidden_states)

        assert output.shape == (batch, seq_len, 64)


class TestSwiGLU:
    """Tests for SwiGLU layer."""

    def test_initialization(self):
        """Test SwiGLU initialization."""
        swiglu = SwiGLU(hidden_size=128, expansion=4.0)

        # Should have gate_up_proj and down_proj
        assert hasattr(swiglu, "gate_up_proj")
        assert hasattr(swiglu, "down_proj")

    def test_forward_shape(self):
        """Test forward pass output shape."""
        swiglu = SwiGLU(hidden_size=128, expansion=4.0)

        x = torch.randn(5, 10, 128)
        output = swiglu(x)

        # Output should match input shape
        assert output.shape == x.shape

    def test_forward_computation(self):
        """Test that forward pass produces valid output."""
        swiglu = SwiGLU(hidden_size=64, expansion=2.0)

        x = torch.randn(2, 4, 64)
        output = swiglu(x)

        # Output should not be all zeros or NaN
        assert not torch.allclose(output, torch.zeros_like(output))
        assert not torch.any(torch.isnan(output))


class TestRMSNorm:
    """Tests for RMS normalization."""

    def test_rms_norm_shape(self):
        """Test RMS norm preserves shape."""
        x = torch.randn(2, 4, 8)
        output = rms_norm(x, variance_epsilon=1e-5)

        assert output.shape == x.shape

    def test_rms_norm_dtype(self):
        """Test RMS norm preserves dtype."""
        x = torch.randn(2, 4, 8, dtype=torch.float16)
        output = rms_norm(x, variance_epsilon=1e-5)

        assert output.dtype == torch.float16

    def test_rms_norm_normalization(self):
        """Test that RMS norm approximately normalizes."""
        x = torch.randn(2, 10, 64) * 10  # Large values
        output = rms_norm(x, variance_epsilon=1e-5)

        # RMS should be approximately 1
        rms = torch.sqrt(torch.mean(output**2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_rms_norm_stability(self):
        """Test numerical stability with very small/large values."""
        x_small = torch.randn(2, 4, 8) * 1e-10
        x_large = torch.randn(2, 4, 8) * 1e10

        output_small = rms_norm(x_small, variance_epsilon=1e-5)
        output_large = rms_norm(x_large, variance_epsilon=1e-5)

        # Should not produce NaN or Inf
        assert not torch.any(torch.isnan(output_small))
        assert not torch.any(torch.isinf(output_small))
        assert not torch.any(torch.isnan(output_large))
        assert not torch.any(torch.isinf(output_large))
