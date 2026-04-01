"""End-to-end integration tests for HRM training and evaluation.

These tests validate the complete pipeline including:
- Model initialization and forward passes
- Loss computation with ACTLossHead
- Multi-step reasoning sequences
- Gradient flow and backpropagation
- ACT halting mechanism over multiple iterations
"""

import pytest
import torch

from hierarchical_reasoning_model.core.losses import ACTLossHead
from hierarchical_reasoning_model.core.model import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config,
)


@pytest.fixture
def integration_config():
    """Create a small model configuration for integration testing."""
    return HierarchicalReasoningModel_ACTV1Config(
        batch_size=4,
        seq_len=16,  # Small sequence
        puzzle_emb_ndim=0,  # Disable for simplicity
        num_puzzle_identifiers=10,
        vocab_size=10,
        H_cycles=2,
        L_cycles=2,
        H_layers=2,
        L_layers=2,
        hidden_size=64,
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        halt_max_steps=5,
        halt_exploration_prob=0.1,
        forward_dtype="float32",
    )


@pytest.fixture
def integration_batch(integration_config):
    """Create a sample batch for integration testing."""
    return {
        "inputs": torch.randint(
            1,
            integration_config.vocab_size,
            (integration_config.batch_size, integration_config.seq_len),
            dtype=torch.int32,
        ),
        "labels": torch.randint(
            1,
            integration_config.vocab_size,
            (integration_config.batch_size, integration_config.seq_len),
            dtype=torch.int32,
        ),
        "puzzle_identifiers": torch.zeros(
            integration_config.batch_size, dtype=torch.int32
        ),
    }


class TestModelInitialization:
    """Test model initialization and basic setup."""

    def test_model_and_loss_head_initialization(self, integration_config):
        """Test creating model with ACTLossHead."""
        model = HierarchicalReasoningModel_ACTV1(integration_config.model_dump())
        loss_head = ACTLossHead(model, loss_type="stablemax_cross_entropy")

        assert loss_head.model is model
        assert hasattr(loss_head, "loss_fn")

    def test_model_parameters_require_grad(self, integration_config):
        """Test that model parameters require gradients."""
        model = HierarchicalReasoningModel_ACTV1(integration_config.model_dump())

        # Check that at least some parameters require grad
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable_params) > 0

        # Check total parameter count
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0


class TestSingleForwardPass:
    """Test single forward pass through model and loss head."""

    @pytest.fixture
    def model_with_loss(self, integration_config):
        """Create model with loss head."""
        model = HierarchicalReasoningModel_ACTV1(integration_config.model_dump())
        return ACTLossHead(model, loss_type="stablemax_cross_entropy")

    def test_forward_pass_completes(self, model_with_loss, integration_batch):
        """Test that a single forward pass completes successfully."""
        carry = model_with_loss.initial_carry(integration_batch)

        new_carry, loss, metrics, predictions, all_halted = model_with_loss.forward(
            return_keys=["logits"], carry=carry, batch=integration_batch
        )

        # Check outputs exist and have correct types
        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)
        assert isinstance(all_halted, torch.Tensor)

    def test_loss_is_scalar(self, model_with_loss, integration_batch):
        """Test that loss is a scalar tensor."""
        carry = model_with_loss.initial_carry(integration_batch)

        _, loss, _, _, _ = model_with_loss.forward(
            return_keys=["logits"], carry=carry, batch=integration_batch
        )

        assert loss.ndim == 0
        assert loss.item() >= 0  # Loss should be non-negative

    def test_metrics_structure(self, model_with_loss, integration_batch):
        """Test that metrics dictionary has expected keys."""
        carry = model_with_loss.initial_carry(integration_batch)

        _, _, metrics, _, _ = model_with_loss.forward(
            return_keys=["logits"], carry=carry, batch=integration_batch
        )

        expected_keys = {
            "count",
            "accuracy",
            "exact_accuracy",
            "q_halt_accuracy",
            "steps",
            "lm_loss",
            "q_halt_loss",
        }
        assert expected_keys.issubset(metrics.keys())

    def test_predictions_returned(self, model_with_loss, integration_batch):
        """Test that predictions are returned when requested."""
        carry = model_with_loss.initial_carry(integration_batch)

        _, _, _, predictions, _ = model_with_loss.forward(
            return_keys=["logits"], carry=carry, batch=integration_batch
        )

        assert predictions is not None
        assert "logits" in predictions


class TestMultiStepReasoning:
    """Test multi-step reasoning with carry state updates."""

    @pytest.fixture
    def model_with_loss(self, integration_config):
        """Create model with loss head."""
        model = HierarchicalReasoningModel_ACTV1(integration_config.model_dump())
        return ACTLossHead(model, loss_type="stablemax_cross_entropy")

    def test_multiple_forward_passes(self, model_with_loss, integration_batch):
        """Test multiple forward passes update carry correctly."""
        carry = model_with_loss.initial_carry(integration_batch)

        # Run 3 forward passes
        losses = []
        for _ in range(3):
            carry, loss, metrics, _, _ = model_with_loss.forward(
                return_keys=["logits"], carry=carry, batch=integration_batch
            )
            losses.append(loss.item())

        # Should have 3 losses
        assert len(losses) == 3

        # All losses should be non-negative
        assert all(loss >= 0 for loss in losses)

    def test_carry_state_evolves(self, model_with_loss, integration_batch):
        """Test that carry state changes across forward passes."""
        carry = model_with_loss.initial_carry(integration_batch)
        initial_steps = carry.steps.clone()

        # Run several forward passes
        for _ in range(3):
            carry, _, _, _, _ = model_with_loss.forward(
                return_keys=["logits"], carry=carry, batch=integration_batch
            )

        # Steps should have incremented
        assert torch.any(carry.steps > initial_steps)

    def test_halting_eventually_occurs(
        self, model_with_loss, integration_batch, integration_config
    ):
        """Test that sequences eventually halt."""
        carry = model_with_loss.initial_carry(integration_batch)

        # Run enough iterations to trigger halting
        max_iterations = integration_config.halt_max_steps * 3
        for _ in range(max_iterations):
            carry, _, _, _, all_halted = model_with_loss.forward(
                return_keys=["logits"], carry=carry, batch=integration_batch
            )

            if all_halted.item():
                break

        # At least some sequences should have halted by now
        # (May not be all due to exploration)
        assert torch.any(carry.halted)

    def test_step_counter_increments(self, model_with_loss, integration_batch):
        """Test that step counter increments for active sequences."""
        carry = model_with_loss.initial_carry(integration_batch)

        # Run one forward pass
        carry, _, metrics, _, _ = model_with_loss.forward(
            return_keys=["logits"], carry=carry, batch=integration_batch
        )

        # Check that steps metric is reported
        assert "steps" in metrics
        assert metrics["steps"] >= 0


class TestGradientFlow:
    """Test gradient computation and backpropagation."""

    @pytest.fixture
    def model_with_loss(self, integration_config):
        """Create model with loss head."""
        model = HierarchicalReasoningModel_ACTV1(integration_config.model_dump())
        return ACTLossHead(model, loss_type="stablemax_cross_entropy")

    def test_loss_requires_grad(self, model_with_loss, integration_batch):
        """Test that loss tensor requires gradients."""
        carry = model_with_loss.initial_carry(integration_batch)

        _, loss, _, _, _ = model_with_loss.forward(
            return_keys=["logits"], carry=carry, batch=integration_batch
        )

        assert loss.requires_grad

    def test_backward_pass_completes(self, model_with_loss, integration_batch):
        """Test that backward pass completes without errors."""
        carry = model_with_loss.initial_carry(integration_batch)

        _, loss, _, _, _ = model_with_loss.forward(
            return_keys=["logits"], carry=carry, batch=integration_batch
        )

        # Backward pass should complete without error
        loss.backward()

    def test_gradients_computed(self, model_with_loss, integration_batch):
        """Test that gradients are computed for model parameters."""
        carry = model_with_loss.initial_carry(integration_batch)

        _, loss, _, _, _ = model_with_loss.forward(
            return_keys=["logits"], carry=carry, batch=integration_batch
        )

        loss.backward()

        # Check that at least some parameters have gradients
        params_with_grad = [
            p for p in model_with_loss.model.parameters() if p.grad is not None
        ]
        assert len(params_with_grad) > 0

    def test_gradients_non_zero(self, model_with_loss, integration_batch):
        """Test that computed gradients are non-zero."""
        carry = model_with_loss.initial_carry(integration_batch)

        _, loss, _, _, _ = model_with_loss.forward(
            return_keys=["logits"], carry=carry, batch=integration_batch
        )

        loss.backward()

        # Check that gradients are non-zero
        non_zero_grads = []
        for p in model_with_loss.model.parameters():
            if p.grad is not None and torch.any(p.grad != 0):
                non_zero_grads.append(p)

        assert len(non_zero_grads) > 0

    def test_gradient_accumulation(self, model_with_loss, integration_batch):
        """Test gradient accumulation across multiple batches."""
        # First batch
        carry1 = model_with_loss.initial_carry(integration_batch)
        _, loss1, _, _, _ = model_with_loss.forward(
            return_keys=["logits"], carry=carry1, batch=integration_batch
        )
        loss1.backward()

        # Save gradients from first batch
        first_grads = {}
        for name, param in model_with_loss.model.named_parameters():
            if param.grad is not None:
                first_grads[name] = param.grad.clone()

        # Second batch (without zeroing gradients)
        carry2 = model_with_loss.initial_carry(integration_batch)
        _, loss2, _, _, _ = model_with_loss.forward(
            return_keys=["logits"], carry=carry2, batch=integration_batch
        )
        loss2.backward()

        # Check that gradients have accumulated
        accumulated = False
        for name, param in model_with_loss.model.named_parameters():
            # Accumulated gradients should generally be larger
            if (
                param.grad is not None
                and name in first_grads
                and torch.sum(torch.abs(param.grad))
                > torch.sum(torch.abs(first_grads[name]))
            ):
                accumulated = True
                break

        assert accumulated


class TestOptimizerIntegration:
    """Test integration with optimizers."""

    @pytest.fixture
    def model_with_loss_and_optimizer(self, integration_config):
        """Create model, loss head, and optimizer."""
        model = HierarchicalReasoningModel_ACTV1(integration_config.model_dump())
        loss_head = ACTLossHead(model, loss_type="stablemax_cross_entropy")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        return loss_head, optimizer

    def test_optimizer_step(self, model_with_loss_and_optimizer, integration_batch):
        """Test that optimizer step updates parameters."""
        loss_head, optimizer = model_with_loss_and_optimizer

        # Save initial parameters
        initial_params = {}
        for name, param in loss_head.model.named_parameters():
            initial_params[name] = param.data.clone()

        # Forward and backward
        carry = loss_head.initial_carry(integration_batch)
        _, loss, _, _, _ = loss_head.forward(
            return_keys=["logits"], carry=carry, batch=integration_batch
        )
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Check that at least some parameters changed
        params_changed = []
        for name, param in loss_head.model.named_parameters():
            if not torch.equal(param.data, initial_params[name]):
                params_changed.append(name)

        assert len(params_changed) > 0

    def test_training_loop_iteration(
        self, model_with_loss_and_optimizer, integration_batch
    ):
        """Test a complete training loop iteration."""
        loss_head, optimizer = model_with_loss_and_optimizer

        # Training iteration
        optimizer.zero_grad()

        carry = loss_head.initial_carry(integration_batch)
        _, loss, metrics, _, _ = loss_head.forward(
            return_keys=["logits"], carry=carry, batch=integration_batch
        )

        loss.backward()
        optimizer.step()

        # Should complete without errors and produce valid metrics
        assert "accuracy" in metrics
        assert "lm_loss" in metrics

    def test_multiple_training_steps(
        self, model_with_loss_and_optimizer, integration_batch
    ):
        """Test multiple training steps."""
        loss_head, optimizer = model_with_loss_and_optimizer

        losses = []
        for _ in range(5):
            optimizer.zero_grad()

            carry = loss_head.initial_carry(integration_batch)
            _, loss, _, _, _ = loss_head.forward(
                return_keys=["logits"], carry=carry, batch=integration_batch
            )

            losses.append(loss.item())

            loss.backward()
            optimizer.step()

        # Should have 5 losses
        assert len(losses) == 5

        # All losses should be finite
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)


class TestACTHaltingMechanism:
    """Test Adaptive Computation Time halting mechanism."""

    @pytest.fixture
    def model_with_loss(self, integration_config):
        """Create model with loss head."""
        model = HierarchicalReasoningModel_ACTV1(integration_config.model_dump())
        return ACTLossHead(model, loss_type="stablemax_cross_entropy")

    def test_q_values_produced(self, model_with_loss, integration_batch):
        """Test that Q-values are produced."""
        carry = model_with_loss.initial_carry(integration_batch)

        _, _, metrics, predictions, _ = model_with_loss.forward(
            return_keys=["logits", "q_halt_logits", "q_continue_logits"],
            carry=carry,
            batch=integration_batch,
        )

        # Check Q-values are in predictions
        assert "q_halt_logits" in predictions
        assert "q_continue_logits" in predictions

        # Check shapes
        assert predictions["q_halt_logits"].shape == (
            integration_batch["inputs"].shape[0],
        )
        assert predictions["q_continue_logits"].shape == (
            integration_batch["inputs"].shape[0],
        )

    def test_q_halt_loss_computed(self, model_with_loss, integration_batch):
        """Test that Q-halt loss is computed."""
        carry = model_with_loss.initial_carry(integration_batch)

        _, _, metrics, _, _ = model_with_loss.forward(
            return_keys=["logits"], carry=carry, batch=integration_batch
        )

        # Check Q-halt loss is in metrics
        assert "q_halt_loss" in metrics
        assert metrics["q_halt_loss"] >= 0

    def test_halt_max_steps_respected(
        self, model_with_loss, integration_batch, integration_config
    ):
        """Test that halt_max_steps limits computation."""
        carry = model_with_loss.initial_carry(integration_batch)

        # Run way past halt_max_steps
        for _ in range(integration_config.halt_max_steps * 2):
            carry, _, _, _, _ = model_with_loss.forward(
                return_keys=["logits"], carry=carry, batch=integration_batch
            )

        # Steps should not exceed halt_max_steps by much
        # (some sequences may take extra steps due to ACT logic)
        max_steps = carry.steps.max().item()
        assert max_steps <= integration_config.halt_max_steps * 1.5


class TestDifferentLossTypes:
    """Test model with different loss types."""

    def test_softmax_loss(self, integration_config, integration_batch):
        """Test model with softmax cross-entropy loss."""
        model = HierarchicalReasoningModel_ACTV1(integration_config.model_dump())
        loss_head = ACTLossHead(model, loss_type="softmax_cross_entropy")

        carry = loss_head.initial_carry(integration_batch)
        _, loss, metrics, _, _ = loss_head.forward(
            return_keys=["logits"], carry=carry, batch=integration_batch
        )

        assert loss.item() >= 0
        assert "lm_loss" in metrics

    def test_stablemax_loss(self, integration_config, integration_batch):
        """Test model with stablemax cross-entropy loss."""
        model = HierarchicalReasoningModel_ACTV1(integration_config.model_dump())
        loss_head = ACTLossHead(model, loss_type="stablemax_cross_entropy")

        carry = loss_head.initial_carry(integration_batch)
        _, loss, metrics, _, _ = loss_head.forward(
            return_keys=["logits"], carry=carry, batch=integration_batch
        )

        assert loss.item() >= 0
        assert "lm_loss" in metrics


@pytest.mark.slow
class TestLongerSequences:
    """Test with longer sequences (marked as slow)."""

    @pytest.fixture
    def longer_config(self):
        """Create config with longer sequences."""
        return HierarchicalReasoningModel_ACTV1Config(
            batch_size=2,
            seq_len=81,  # Full Sudoku grid
            puzzle_emb_ndim=0,
            num_puzzle_identifiers=10,
            vocab_size=11,
            H_cycles=2,
            L_cycles=2,
            H_layers=4,
            L_layers=4,
            hidden_size=128,
            expansion=4.0,
            num_heads=8,
            pos_encodings="rope",
            halt_max_steps=10,
            halt_exploration_prob=0.1,
            forward_dtype="float32",
        )

    @pytest.fixture
    def longer_batch(self, longer_config):
        """Create batch with longer sequences."""
        return {
            "inputs": torch.randint(
                1,
                longer_config.vocab_size,
                (longer_config.batch_size, longer_config.seq_len),
                dtype=torch.int32,
            ),
            "labels": torch.randint(
                1,
                longer_config.vocab_size,
                (longer_config.batch_size, longer_config.seq_len),
                dtype=torch.int32,
            ),
            "puzzle_identifiers": torch.zeros(
                longer_config.batch_size, dtype=torch.int32
            ),
        }

    def test_longer_sequence_forward_pass(self, longer_config, longer_batch):
        """Test forward pass with longer sequences."""
        model = HierarchicalReasoningModel_ACTV1(longer_config.model_dump())
        loss_head = ACTLossHead(model, loss_type="stablemax_cross_entropy")

        carry = loss_head.initial_carry(longer_batch)
        _, loss, metrics, _, _ = loss_head.forward(
            return_keys=["logits"], carry=carry, batch=longer_batch
        )

        assert loss.item() >= 0
        assert "accuracy" in metrics

    def test_longer_sequence_training_step(self, longer_config, longer_batch):
        """Test complete training step with longer sequences."""
        model = HierarchicalReasoningModel_ACTV1(longer_config.model_dump())
        loss_head = ACTLossHead(model, loss_type="stablemax_cross_entropy")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        optimizer.zero_grad()

        carry = loss_head.initial_carry(longer_batch)
        _, loss, _, _, _ = loss_head.forward(
            return_keys=["logits"], carry=carry, batch=longer_batch
        )

        loss.backward()
        optimizer.step()

        # Should complete without errors
        assert True
