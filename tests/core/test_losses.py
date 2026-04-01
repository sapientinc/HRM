"""Tests for loss functions and ACTLossHead."""

import pytest
import torch

from hierarchical_reasoning_model.core.losses import (
    IGNORE_LABEL_ID,
    ACTLossHead,
    log_stablemax,
    softmax_cross_entropy,
    stablemax_cross_entropy,
    stablemax_transform,
)


class TestStablemaxFunctions:
    """Tests for stablemax transformation and normalization."""

    def test_s_function_negative(self):
        """Test stablemax_transform() on negative values."""
        x = torch.tensor([-1.0, -2.0, -3.0])
        result = stablemax_transform(x)

        # For negative values, stablemax_transform(x) = 1 / (1 - x + epsilon)
        # Should be in range (0, 1)
        assert torch.all(result > 0)
        assert torch.all(result < 1)

    def test_s_function_positive(self):
        """Test stablemax_transform() on positive values."""
        x = torch.tensor([1.0, 2.0, 3.0])
        result = stablemax_transform(x)

        # For non-negative values, stablemax_transform(x) = x + 1
        expected = x + 1
        assert torch.allclose(result, expected)

    def test_s_function_zero(self):
        """Test stablemax_transform() at zero."""
        x = torch.tensor([0.0])
        result = stablemax_transform(x)

        # At zero, stablemax_transform(0) = 0 + 1 = 1
        assert torch.allclose(result, torch.tensor([1.0]))

    def test_log_stablemax_normalization(self):
        """Test that log_stablemax produces valid log probabilities."""
        x = torch.randn(3, 5)
        log_probs = log_stablemax(x, dim=-1)

        # Log probs should sum to 1 in probability space
        probs = torch.exp(log_probs)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-6)

    def test_log_stablemax_shape(self):
        """Test that log_stablemax preserves shape."""
        x = torch.randn(2, 3, 4)
        result = log_stablemax(x, dim=-1)
        assert result.shape == x.shape


class TestCrossEntropyLosses:
    """Tests for cross-entropy loss functions."""

    def test_stablemax_cross_entropy_basic(self):
        """Test basic stablemax cross-entropy computation."""
        batch_size, seq_len, vocab_size = 2, 4, 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = stablemax_cross_entropy(logits, labels)

        assert loss.shape == (batch_size, seq_len)
        assert torch.all(loss >= 0)  # Loss should be non-negative

    def test_stablemax_cross_entropy_ignore_index(self):
        """Test that ignore_index is properly handled."""
        batch_size, seq_len, vocab_size = 2, 4, 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Set some labels to ignore index
        labels[0, 0] = IGNORE_LABEL_ID
        labels[1, 2] = IGNORE_LABEL_ID

        loss = stablemax_cross_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID)

        # Ignored positions should have zero loss
        assert loss[0, 0] == 0.0
        assert loss[1, 2] == 0.0

    def test_softmax_cross_entropy_basic(self):
        """Test basic softmax cross-entropy computation."""
        batch_size, seq_len, vocab_size = 2, 4, 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        loss = softmax_cross_entropy(logits, labels)

        assert loss.shape == (batch_size, seq_len)
        assert torch.all(loss >= 0)

    def test_softmax_cross_entropy_ignore_index(self):
        """Test that ignore_index is properly handled in softmax."""
        batch_size, seq_len, vocab_size = 2, 4, 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        labels[0, 1] = IGNORE_LABEL_ID

        loss = softmax_cross_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID)

        assert loss[0, 1] == 0.0

    def test_cross_entropy_perfect_prediction(self):
        """Test that perfect predictions give near-zero loss."""
        batch_size, seq_len, vocab_size = 2, 3, 5
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create logits with very high values at correct labels
        logits = torch.full((batch_size, seq_len, vocab_size), -10.0)
        for b in range(batch_size):
            for s in range(seq_len):
                logits[b, s, labels[b, s]] = 10.0

        loss_stablemax = stablemax_cross_entropy(logits, labels)
        loss_softmax = softmax_cross_entropy(logits, labels)

        # Both should give very small loss
        assert torch.all(loss_stablemax < 0.1)
        assert torch.all(loss_softmax < 0.1)


class MockModel(torch.nn.Module):
    """Mock model for testing ACTLossHead."""

    def __init__(self, batch_size, seq_len, vocab_size, hidden_size):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def initial_carry(self, batch):
        """Return mock initial carry."""
        return type(
            "Carry",
            (),
            {
                "current_data": batch,
                "halted": torch.zeros(self.batch_size, dtype=torch.bool),
                "steps": torch.zeros(self.batch_size, dtype=torch.int32),
            },
        )()

    def forward(self, carry, batch):
        """Return mock outputs."""
        logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size)
        q_halt_logits = torch.randn(self.batch_size)
        q_continue_logits = torch.randn(self.batch_size)

        # Update carry
        new_carry = type(
            "Carry",
            (),
            {
                "current_data": batch,
                "halted": torch.ones(self.batch_size, dtype=torch.bool),
                "steps": torch.ones(self.batch_size, dtype=torch.int32),
            },
        )()

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        return new_carry, outputs


class TestACTLossHead:
    """Tests for ACTLossHead wrapper."""

    @pytest.fixture
    def mock_model(self, batch_size, seq_len, vocab_size, hidden_size):
        """Create a mock model for testing."""
        return MockModel(batch_size, seq_len, vocab_size, hidden_size)

    @pytest.fixture
    def act_loss_head(self, mock_model):
        """Create ACTLossHead with mock model."""
        return ACTLossHead(mock_model, loss_type="stablemax_cross_entropy")

    def test_initialization(self, act_loss_head, mock_model):
        """Test ACTLossHead initialization."""
        assert act_loss_head.model is mock_model
        assert act_loss_head.loss_fn is stablemax_cross_entropy

    def test_initialization_with_softmax(self, mock_model):
        """Test initialization with softmax loss."""
        loss_head = ACTLossHead(mock_model, loss_type="softmax_cross_entropy")
        assert loss_head.loss_fn is softmax_cross_entropy

    def test_initial_carry(self, act_loss_head, sample_batch):
        """Test initial_carry passes through to model."""
        carry = act_loss_head.initial_carry(sample_batch)
        assert hasattr(carry, "current_data")
        assert hasattr(carry, "halted")
        assert hasattr(carry, "steps")

    def test_forward_returns_expected_outputs(self, act_loss_head, sample_batch):
        """Test forward returns correct output structure."""
        carry = act_loss_head.initial_carry(sample_batch)

        new_carry, loss, metrics, predictions, all_halted = act_loss_head.forward(
            return_keys=["logits"],
            carry=carry,
            batch=sample_batch,
        )

        # Check outputs
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert isinstance(metrics, dict)
        assert isinstance(all_halted, torch.Tensor)

        # Check metrics
        assert "count" in metrics
        assert "accuracy" in metrics
        assert "exact_accuracy" in metrics
        assert "q_halt_accuracy" in metrics
        assert "steps" in metrics
        assert "lm_loss" in metrics
        assert "q_halt_loss" in metrics

    def test_forward_with_ignore_labels(self, act_loss_head, sample_batch):
        """Test forward with some labels marked as ignore."""
        # Set some labels to ignore
        sample_batch["labels"][0, :10] = IGNORE_LABEL_ID

        carry = act_loss_head.initial_carry(sample_batch)
        new_carry, loss, metrics, predictions, all_halted = act_loss_head.forward(
            return_keys=["logits"],
            carry=carry,
            batch=sample_batch,
        )

        # Should still compute loss without errors
        assert loss.item() >= 0

    def test_metrics_are_detached(self, act_loss_head, sample_batch):
        """Test that returned metrics are detached from computation graph."""
        carry = act_loss_head.initial_carry(sample_batch)
        new_carry, loss, metrics, predictions, all_halted = act_loss_head.forward(
            return_keys=["logits"],
            carry=carry,
            batch=sample_batch,
        )

        # Metrics should not require grad
        for _key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                assert not value.requires_grad

    def test_predictions_are_detached(self, act_loss_head, sample_batch):
        """Test that returned predictions are detached."""
        carry = act_loss_head.initial_carry(sample_batch)
        new_carry, loss, metrics, predictions, all_halted = act_loss_head.forward(
            return_keys=["logits"],
            carry=carry,
            batch=sample_batch,
        )

        if predictions is not None:
            for _key, value in predictions.items():
                if isinstance(value, torch.Tensor):
                    assert not value.requires_grad
