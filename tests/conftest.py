"""Pytest configuration and shared fixtures for HRM tests."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Mock FlashAttention for tests (not all test environments have GPU)
# This allows tests to run without requiring CUDA
if "flash_attn" not in sys.modules and "flash_attn_interface" not in sys.modules:
    # Create mock flash_attn module
    flash_attn_mock = MagicMock()

    def mock_flash_attn_func(q, k, v, causal=False):
        """Mock flash attention function.

        Mimics FlashAttention interface with shape (batch, seq_len, num_heads, head_dim).
        """
        # q, k, v shape: (batch, seq_len, num_heads, head_dim)
        batch, seq_len, num_heads, head_dim = q.shape

        # Transpose to (batch, num_heads, seq_len, head_dim) for matmul
        q_t = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / (head_dim**0.5)
        # scores shape: (batch, num_heads, seq_len, seq_len)

        if causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1)
            scores = scores.masked_fill(mask.bool(), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v_t)  # (batch, num_heads, seq_len, head_dim)

        # Transpose back to (batch, seq_len, num_heads, head_dim)
        output = output.transpose(1, 2).contiguous()
        return output

    flash_attn_mock.flash_attn_func = mock_flash_attn_func

    sys.modules["flash_attn"] = flash_attn_mock
    sys.modules["flash_attn_interface"] = flash_attn_mock


@pytest.fixture
def device():
    """Provide CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 4


@pytest.fixture
def seq_len():
    """Default sequence length for tests."""
    return 81  # Sudoku grid size


@pytest.fixture
def vocab_size():
    """Default vocabulary size for tests."""
    return 11  # 0-9 + padding


@pytest.fixture
def hidden_size():
    """Default hidden size for tests."""
    return 128  # Smaller for faster tests


@pytest.fixture
def sample_batch(batch_size, seq_len, vocab_size):
    """Create a sample batch for testing."""
    return {
        "inputs": torch.randint(
            0, vocab_size, (batch_size, seq_len), dtype=torch.int32
        ),
        "labels": torch.randint(
            1, vocab_size, (batch_size, seq_len), dtype=torch.int32
        ),
        "puzzle_identifiers": torch.zeros(batch_size, dtype=torch.int32),
    }


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed
