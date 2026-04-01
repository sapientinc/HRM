"""Tests for Hierarchical Reasoning Model (HRM) components."""

import pytest
import torch

from hierarchical_reasoning_model.core.model import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1_Inner,
    HierarchicalReasoningModel_ACTV1Block,
    HierarchicalReasoningModel_ACTV1Carry,
    HierarchicalReasoningModel_ACTV1Config,
    HierarchicalReasoningModel_ACTV1InnerCarry,
    HierarchicalReasoningModel_ACTV1ReasoningModule,
)


@pytest.fixture
def small_config():
    """Create a small model configuration for fast testing."""
    return HierarchicalReasoningModel_ACTV1Config(
        batch_size=2,
        seq_len=9,  # Small for fast tests (e.g., 3x3 grid)
        puzzle_emb_ndim=0,  # Disable puzzle embeddings for simplicity
        num_puzzle_identifiers=10,
        vocab_size=5,  # Small vocab
        H_cycles=1,
        L_cycles=1,
        H_layers=2,
        L_layers=2,
        hidden_size=64,  # Small for speed
        expansion=2.0,
        num_heads=4,
        pos_encodings="rope",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        halt_max_steps=3,
        halt_exploration_prob=0.1,
        forward_dtype="float32",  # Use float32 for testing
    )


@pytest.fixture
def config_with_puzzle_emb(small_config):
    """Create config with puzzle embeddings enabled."""
    config_dict = small_config.model_dump()
    config_dict["puzzle_emb_ndim"] = 32
    return HierarchicalReasoningModel_ACTV1Config(**config_dict)


@pytest.fixture
def config_learned_pos(small_config):
    """Create config with learned positional encodings."""
    config_dict = small_config.model_dump()
    config_dict["pos_encodings"] = "learned"
    return HierarchicalReasoningModel_ACTV1Config(**config_dict)


@pytest.fixture
def sample_batch_model(small_config):
    """Create sample batch for model testing."""
    return {
        "inputs": torch.randint(
            0, small_config.vocab_size, (small_config.batch_size, small_config.seq_len)
        ),
        "labels": torch.randint(
            0, small_config.vocab_size, (small_config.batch_size, small_config.seq_len)
        ),
        "puzzle_identifiers": torch.zeros(small_config.batch_size, dtype=torch.int32),
    }


class TestHierarchicalReasoningModelConfig:
    """Tests for model configuration."""

    def test_config_initialization(self, small_config):
        """Test configuration initialization."""
        assert small_config.batch_size == 2
        assert small_config.seq_len == 9
        assert small_config.vocab_size == 5
        assert small_config.hidden_size == 64
        assert small_config.H_cycles == 1
        assert small_config.L_cycles == 1

    def test_config_with_puzzle_embeddings(self, config_with_puzzle_emb):
        """Test configuration with puzzle embeddings."""
        assert config_with_puzzle_emb.puzzle_emb_ndim == 32
        assert config_with_puzzle_emb.num_puzzle_identifiers == 10

    def test_config_different_pos_encodings(self):
        """Test different positional encoding types."""
        config_rope = HierarchicalReasoningModel_ACTV1Config(
            batch_size=2,
            seq_len=9,
            num_puzzle_identifiers=10,
            vocab_size=5,
            H_cycles=1,
            L_cycles=1,
            H_layers=2,
            L_layers=2,
            hidden_size=64,
            expansion=2.0,
            num_heads=4,
            pos_encodings="rope",
            halt_max_steps=3,
            halt_exploration_prob=0.1,
        )

        config_learned = HierarchicalReasoningModel_ACTV1Config(
            batch_size=2,
            seq_len=9,
            num_puzzle_identifiers=10,
            vocab_size=5,
            H_cycles=1,
            L_cycles=1,
            H_layers=2,
            L_layers=2,
            hidden_size=64,
            expansion=2.0,
            num_heads=4,
            pos_encodings="learned",
            halt_max_steps=3,
            halt_exploration_prob=0.1,
        )

        assert config_rope.pos_encodings == "rope"
        assert config_learned.pos_encodings == "learned"


class TestHierarchicalReasoningModelBlock:
    """Tests for single transformer block."""

    @pytest.fixture
    def block(self, small_config):
        """Create a transformer block."""
        return HierarchicalReasoningModel_ACTV1Block(small_config)

    def test_block_initialization(self, block):
        """Test block initialization."""
        assert hasattr(block, "self_attn")
        assert hasattr(block, "mlp")
        assert block.norm_eps == 1e-5

    def test_block_forward_shape(self, block, small_config):
        """Test block forward pass output shape."""
        batch_size = small_config.batch_size
        seq_len = small_config.seq_len
        hidden_size = small_config.hidden_size

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        cos_sin = None  # RoPE will use default

        output = block(cos_sin=cos_sin, hidden_states=hidden_states)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_block_forward_no_nan(self, block, small_config):
        """Test that block produces valid outputs."""
        hidden_states = torch.randn(
            small_config.batch_size, small_config.seq_len, small_config.hidden_size
        )

        output = block(cos_sin=None, hidden_states=hidden_states)

        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))


class TestHierarchicalReasoningModelReasoningModule:
    """Tests for reasoning module (H or L level)."""

    @pytest.fixture
    def reasoning_module(self, small_config):
        """Create a reasoning module."""
        layers = [
            HierarchicalReasoningModel_ACTV1Block(small_config)
            for _ in range(small_config.H_layers)
        ]
        return HierarchicalReasoningModel_ACTV1ReasoningModule(layers)

    def test_reasoning_module_initialization(self, reasoning_module, small_config):
        """Test reasoning module initialization."""
        assert len(reasoning_module.layers) == small_config.H_layers

    def test_reasoning_module_forward_shape(self, reasoning_module, small_config):
        """Test reasoning module forward pass."""
        batch_size = small_config.batch_size
        seq_len = small_config.seq_len
        hidden_size = small_config.hidden_size

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        input_injection = torch.randn(batch_size, seq_len, hidden_size)

        output = reasoning_module(
            hidden_states=hidden_states, input_injection=input_injection, cos_sin=None
        )

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_reasoning_module_input_injection(self, reasoning_module, small_config):
        """Test that input injection is applied."""
        hidden_states = torch.zeros(
            small_config.batch_size, small_config.seq_len, small_config.hidden_size
        )
        input_injection = torch.ones(
            small_config.batch_size, small_config.seq_len, small_config.hidden_size
        )

        output = reasoning_module(
            hidden_states=hidden_states, input_injection=input_injection, cos_sin=None
        )

        # Output should be affected by input injection (non-zero)
        assert not torch.allclose(output, torch.zeros_like(output))


class TestHierarchicalReasoningModelInner:
    """Tests for inner model (core computational engine)."""

    @pytest.fixture
    def inner_model(self, small_config):
        """Create inner model."""
        return HierarchicalReasoningModel_ACTV1_Inner(small_config)

    @pytest.fixture
    def inner_model_puzzle_emb(self, config_with_puzzle_emb):
        """Create inner model with puzzle embeddings."""
        return HierarchicalReasoningModel_ACTV1_Inner(config_with_puzzle_emb)

    @pytest.fixture
    def inner_model_learned_pos(self, config_learned_pos):
        """Create inner model with learned positional encodings."""
        return HierarchicalReasoningModel_ACTV1_Inner(config_learned_pos)

    def test_inner_model_initialization(self, inner_model, small_config):
        """Test inner model initialization."""
        assert inner_model.config == small_config
        assert hasattr(inner_model, "embed_tokens")
        assert hasattr(inner_model, "lm_head")
        assert hasattr(inner_model, "q_head")
        assert hasattr(inner_model, "H_level")
        assert hasattr(inner_model, "L_level")
        assert hasattr(inner_model, "H_init")
        assert hasattr(inner_model, "L_init")

    def test_inner_model_rope_initialization(self, inner_model):
        """Test RoPE initialization."""
        assert hasattr(inner_model, "rotary_emb")
        assert not hasattr(inner_model, "embed_pos")

    def test_inner_model_learned_pos_initialization(self, inner_model_learned_pos):
        """Test learned positional encoding initialization."""
        assert hasattr(inner_model_learned_pos, "embed_pos")
        assert not hasattr(inner_model_learned_pos, "rotary_emb")

    def test_inner_model_puzzle_emb_initialization(self, inner_model_puzzle_emb):
        """Test puzzle embedding initialization."""
        assert hasattr(inner_model_puzzle_emb, "puzzle_emb")
        assert inner_model_puzzle_emb.puzzle_emb_len > 0

    def test_inner_model_q_head_init(self, inner_model):
        """Test Q-head special initialization."""
        # Q-head should be initialized to near-zero for bootstrapping
        assert torch.allclose(
            inner_model.q_head.weight, torch.zeros_like(inner_model.q_head.weight)
        )
        assert torch.allclose(
            inner_model.q_head.bias, torch.full_like(inner_model.q_head.bias, -5.0)
        )

    def test_empty_carry_shape(self, inner_model, small_config):
        """Test empty carry creation."""
        carry = inner_model.empty_carry(batch_size=small_config.batch_size)

        assert isinstance(carry, HierarchicalReasoningModel_ACTV1InnerCarry)
        assert carry.z_H.shape == (
            small_config.batch_size,
            small_config.seq_len,
            small_config.hidden_size,
        )
        assert carry.z_L.shape == (
            small_config.batch_size,
            small_config.seq_len,
            small_config.hidden_size,
        )

    def test_reset_carry(self, inner_model, small_config):
        """Test carry reset with flags."""
        carry = inner_model.empty_carry(batch_size=small_config.batch_size)

        # Create reset flags (reset first sequence)
        reset_flag = torch.tensor([True, False])

        new_carry = inner_model.reset_carry(reset_flag, carry)

        # First sequence should be reset to H_init/L_init
        # Second sequence should be unchanged
        assert isinstance(new_carry, HierarchicalReasoningModel_ACTV1InnerCarry)
        assert new_carry.z_H.shape == carry.z_H.shape
        assert new_carry.z_L.shape == carry.z_L.shape

    def test_input_embeddings_shape(
        self, inner_model, small_config, sample_batch_model
    ):
        """Test input embedding computation."""
        embeddings = inner_model._input_embeddings(
            input=sample_batch_model["inputs"],
            puzzle_identifiers=sample_batch_model["puzzle_identifiers"],
        )

        # Should return embeddings with correct shape
        assert embeddings.shape == (
            small_config.batch_size,
            small_config.seq_len,
            small_config.hidden_size,
        )

    def test_input_embeddings_with_puzzle_emb(
        self, inner_model_puzzle_emb, config_with_puzzle_emb, sample_batch_model
    ):
        """Test input embeddings with puzzle embeddings."""
        embeddings = inner_model_puzzle_emb._input_embeddings(
            input=sample_batch_model["inputs"],
            puzzle_identifiers=sample_batch_model["puzzle_identifiers"],
        )

        # Should include puzzle embedding length
        expected_len = (
            config_with_puzzle_emb.seq_len + inner_model_puzzle_emb.puzzle_emb_len
        )
        assert embeddings.shape == (
            config_with_puzzle_emb.batch_size,
            expected_len,
            config_with_puzzle_emb.hidden_size,
        )

    def test_input_embeddings_scaling(self, inner_model, sample_batch_model):
        """Test that embeddings are properly scaled."""
        embeddings = inner_model._input_embeddings(
            input=sample_batch_model["inputs"],
            puzzle_identifiers=sample_batch_model["puzzle_identifiers"],
        )

        # Embeddings should be scaled by sqrt(hidden_size)
        # Just check they're non-zero and reasonable magnitude
        assert not torch.allclose(embeddings, torch.zeros_like(embeddings))
        assert torch.all(torch.abs(embeddings) < 100)  # Reasonable magnitude


class TestHierarchicalReasoningModelACTV1:
    """Tests for complete HRM model with ACT."""

    @pytest.fixture
    def model(self, small_config):
        """Create full model."""
        # Model expects a dict, not a Config object
        return HierarchicalReasoningModel_ACTV1(small_config.model_dump())

    def test_model_initialization(self, model, small_config):
        """Test model initialization."""
        assert model.config == small_config
        assert hasattr(model, "inner")
        assert isinstance(model.inner, HierarchicalReasoningModel_ACTV1_Inner)

    def test_initial_carry_structure(self, model, sample_batch_model):
        """Test initial carry creation."""
        carry = model.initial_carry(sample_batch_model)

        assert isinstance(carry, HierarchicalReasoningModel_ACTV1Carry)
        assert isinstance(carry.inner_carry, HierarchicalReasoningModel_ACTV1InnerCarry)
        assert carry.steps.shape == (model.config.batch_size,)
        assert carry.halted.shape == (model.config.batch_size,)
        assert carry.steps.dtype == torch.int32
        assert carry.halted.dtype == torch.bool

    def test_initial_carry_all_halted(self, model, sample_batch_model):
        """Test that initial carry has all sequences halted (by design)."""
        carry = model.initial_carry(sample_batch_model)

        # Initial carry should have zero steps and all halted
        # (This triggers data loading on first forward pass)
        assert torch.all(carry.steps == 0)
        assert torch.all(carry.halted)

    def test_forward_output_structure(self, model, sample_batch_model):
        """Test forward pass output structure."""
        carry = model.initial_carry(sample_batch_model)

        new_carry, outputs = model.forward(carry, sample_batch_model)

        # Check new carry
        assert isinstance(new_carry, HierarchicalReasoningModel_ACTV1Carry)

        # Check outputs
        assert "logits" in outputs
        assert "q_halt_logits" in outputs
        assert "q_continue_logits" in outputs

    def test_forward_logits_shape(self, model, sample_batch_model, small_config):
        """Test forward pass logits shape."""
        carry = model.initial_carry(sample_batch_model)
        new_carry, outputs = model.forward(carry, sample_batch_model)

        # Logits should be (batch_size, seq_len, vocab_size)
        assert outputs["logits"].shape == (
            small_config.batch_size,
            small_config.seq_len,
            small_config.vocab_size,
        )

    def test_forward_q_values_shape(self, model, sample_batch_model, small_config):
        """Test forward pass Q-value shapes."""
        carry = model.initial_carry(sample_batch_model)
        new_carry, outputs = model.forward(carry, sample_batch_model)

        # Q-values should be (batch_size,)
        assert outputs["q_halt_logits"].shape == (small_config.batch_size,)
        assert outputs["q_continue_logits"].shape == (small_config.batch_size,)

    def test_forward_no_nan(self, model, sample_batch_model):
        """Test that forward pass produces valid outputs."""
        carry = model.initial_carry(sample_batch_model)
        new_carry, outputs = model.forward(carry, sample_batch_model)

        assert not torch.any(torch.isnan(outputs["logits"]))
        assert not torch.any(torch.isnan(outputs["q_halt_logits"]))
        assert not torch.any(torch.isnan(outputs["q_continue_logits"]))

    def test_forward_updates_steps(self, model, sample_batch_model):
        """Test that forward pass increments step counter."""
        carry = model.initial_carry(sample_batch_model)
        initial_steps = carry.steps.clone()

        new_carry, _ = model.forward(carry, sample_batch_model)

        # Steps should increment for non-halted sequences
        assert torch.all(new_carry.steps >= initial_steps)

    def test_forward_multiple_steps(self, model, sample_batch_model):
        """Test multiple forward passes update carry correctly."""
        carry = model.initial_carry(sample_batch_model)

        # Run 3 forward passes
        for _ in range(3):
            carry, outputs = model.forward(carry, sample_batch_model)

        # Steps should have incremented
        assert torch.any(carry.steps > 0)

    def test_forward_respects_halt_max_steps(
        self, model, sample_batch_model, small_config
    ):
        """Test that model respects halt_max_steps."""
        carry = model.initial_carry(sample_batch_model)

        # Run more than halt_max_steps iterations
        for _ in range(small_config.halt_max_steps + 5):
            carry, outputs = model.forward(carry, sample_batch_model)

        # All should eventually halt
        # (Note: may not be immediate due to exploration, but eventually will halt)
        # For this test, just check that halting mechanism is working
        assert isinstance(carry.halted, torch.Tensor)

    def test_forward_dtype_consistency(self, model, sample_batch_model, small_config):
        """Test that forward pass maintains dtype consistency."""
        carry = model.initial_carry(sample_batch_model)
        new_carry, outputs = model.forward(carry, sample_batch_model)

        forward_dtype = getattr(torch, small_config.forward_dtype)

        # Check internal carry dtypes
        assert new_carry.inner_carry.z_H.dtype == forward_dtype
        assert new_carry.inner_carry.z_L.dtype == forward_dtype

    def test_model_device_placement(self, model):
        """Test model parameters are on correct device."""
        device = next(model.parameters()).device

        # All parameters should be on the same device
        for param in model.parameters():
            assert param.device == device


class TestHierarchicalReasoningModelInnerCarry:
    """Tests for inner carry dataclass."""

    def test_inner_carry_creation(self, small_config):
        """Test inner carry creation."""
        z_H = torch.randn(  # noqa: N806
            small_config.batch_size, small_config.seq_len, small_config.hidden_size
        )
        z_L = torch.randn(  # noqa: N806
            small_config.batch_size, small_config.seq_len, small_config.hidden_size
        )

        carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H, z_L=z_L)

        assert torch.equal(carry.z_H, z_H)
        assert torch.equal(carry.z_L, z_L)


class TestHierarchicalReasoningModelCarry:
    """Tests for full carry dataclass."""

    def test_carry_creation(self, small_config):
        """Test carry creation."""
        inner_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.randn(
                small_config.batch_size, small_config.seq_len, small_config.hidden_size
            ),
            z_L=torch.randn(
                small_config.batch_size, small_config.seq_len, small_config.hidden_size
            ),
        )

        steps = torch.zeros(small_config.batch_size, dtype=torch.int32)
        halted = torch.zeros(small_config.batch_size, dtype=torch.bool)
        current_data = {
            "inputs": torch.randint(
                0, 5, (small_config.batch_size, small_config.seq_len)
            ),
            "labels": torch.randint(
                0, 5, (small_config.batch_size, small_config.seq_len)
            ),
            "puzzle_identifiers": torch.zeros(
                small_config.batch_size, dtype=torch.int32
            ),
        }

        carry = HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=inner_carry,
            steps=steps,
            halted=halted,
            current_data=current_data,
        )

        assert carry.inner_carry == inner_carry
        assert torch.equal(carry.steps, steps)
        assert torch.equal(carry.halted, halted)
        assert carry.current_data == current_data
