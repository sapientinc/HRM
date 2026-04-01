"""Hierarchical Reasoning Model with Adaptive Computation Time (ACT).

This module implements the core HRM architecture featuring two-level hierarchical
processing inspired by human cognition. The model consists of:

- High-level (H) module: Slow, abstract planning and reasoning
- Low-level (L) module: Fast, detailed computations
- ACT mechanism: Q-learning based adaptive halting for dynamic computation

The architecture is particularly effective for sequential reasoning tasks like
ARC puzzles, Sudoku, and maze solving with minimal training data.

References:
    Paper: "Hierarchical Reasoning Model for Sequential Reasoning Tasks"
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from pydantic import BaseModel
from torch import nn

from hierarchical_reasoning_model.core.common import truncated_normal_init_
from hierarchical_reasoning_model.core.embeddings import CastedSparseEmbedding
from hierarchical_reasoning_model.core.layers import (
    Attention,
    CastedEmbedding,
    CastedLinear,
    CosSin,
    RotaryEmbedding,
    SwiGLU,
    rms_norm,
)


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:  # noqa: N801
    """Internal state for the hierarchical reasoning modules.

    Stores the hidden states for both the high-level (H) and low-level (L)
    processing modules during recurrent computation.

    Attributes:
        z_H: High-level hidden state tensor of shape
            (batch_size, seq_len + puzzle_emb_len, hidden_size)
        z_L: Low-level hidden state tensor of shape
            (batch_size, seq_len + puzzle_emb_len, hidden_size)
    """

    z_H: torch.Tensor  # noqa: N815
    z_L: torch.Tensor  # noqa: N815


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:  # noqa: N801
    """Complete carry state for HRM with ACT mechanism.

    Maintains all state information needed across forward passes, including
    internal hidden states, step counts, halting flags, and current data.

    Attributes:
        inner_carry: Internal state containing z_H and z_L hidden states
        steps: Number of computation steps taken for each sequence in batch,
            shape (batch_size,)
        halted: Boolean flags indicating which sequences have halted,
            shape (batch_size,)
        current_data: Dictionary containing current batch data including
            'inputs', 'labels', and 'puzzle_identifiers'
    """

    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):  # noqa: N801
    """Configuration for Hierarchical Reasoning Model with ACT.

    Defines all hyperparameters for the model architecture, training, and
    adaptive computation time mechanism.

    Attributes:
        batch_size: Number of sequences processed in parallel
        seq_len: Maximum sequence length (e.g., 81 for Sudoku, 900 for ARC)
        puzzle_emb_ndim: Dimension of per-puzzle sparse embeddings (0 to disable)
        num_puzzle_identifiers: Total number of unique puzzle types/IDs
        vocab_size: Size of token vocabulary (e.g., 11 for Sudoku: 0-9 + pad)

        H_cycles: Number of high-level reasoning cycles per forward pass
        L_cycles: Number of low-level computation cycles per H cycle

        H_layers: Number of transformer layers in high-level module
        L_layers: Number of transformer layers in low-level module

        hidden_size: Dimension of hidden states and embeddings
        expansion: MLP expansion ratio for SwiGLU (typically 4.0)
        num_heads: Number of attention heads
        pos_encodings: Type of positional encoding ("rope" or "learned")

        rms_norm_eps: Epsilon for RMS normalization stability
        rope_theta: Base frequency for rotary position embeddings

        halt_max_steps: Maximum computation steps before forcing halt
        halt_exploration_prob: Probability of exploration during ACT training

        forward_dtype: Data type for forward pass ("bfloat16" or "float32")
    """

    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"


class HierarchicalReasoningModel_ACTV1Block(nn.Module):  # noqa: N801
    """Single transformer block with post-normalization.

    Implements a standard transformer layer with self-attention and MLP,
    using post-normalization (add-then-norm) and RMS normalization.

    Args:
        config: Model configuration containing architecture hyperparameters

    Attributes:
        self_attn: Multi-head self-attention layer (non-causal)
        mlp: SwiGLU feed-forward layer
        norm_eps: Epsilon for RMS normalization stability
    """

    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply transformer block operations.

        Args:
            cos_sin: Precomputed cosine and sine values for RoPE
            hidden_states: Input hidden states of shape
                (batch_size, seq_len, hidden_size)

        Returns:
            Transformed hidden states of same shape as input
        """
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(
            hidden_states
            + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        # Fully Connected
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps
        )
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):  # noqa: N801
    """Reasoning module for H-level or L-level processing.

    Stacks multiple transformer blocks and applies input injection to enable
    communication between hierarchical levels. Used for both high-level (H)
    and low-level (L) reasoning modules.

    Args:
        layers: List of transformer blocks to stack

    Attributes:
        layers: ModuleList containing all transformer blocks
    """

    def __init__(self, layers: list[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(
        self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Process hidden states with input injection from other level.

        Args:
            hidden_states: Current hidden states of shape
                (batch_size, seq_len, hidden_size)
            input_injection: States from other hierarchical level to inject,
                same shape as hidden_states
            **kwargs: Additional arguments passed to transformer blocks
                (typically cos_sin for RoPE)

        Returns:
            Processed hidden states of same shape as input
        """
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):  # noqa: N801
    """Core inner model implementing hierarchical reasoning with dual recurrence.

    This is the main computational engine featuring:
    - Token and puzzle embeddings with positional encoding
    - Two-level hierarchical processing (H and L modules)
    - Efficient 1-step gradient computation strategy
    - Language modeling and Q-value heads for ACT

    The forward pass uses a clever optimization: run H_cycles × L_cycles iterations
    without gradients, then execute final iteration with gradients. This maintains
    computational depth while enabling stable training.

    Args:
        config: Complete model configuration

    Attributes:
        config: Stored configuration
        forward_dtype: PyTorch dtype for forward pass (bfloat16 or float32)
        embed_scale: Embedding scaling factor (sqrt(hidden_size))
        embed_tokens: Token embedding layer
        lm_head: Language modeling output head
        q_head: Q-value head for halt/continue decisions
        puzzle_emb: Optional sparse puzzle embeddings
        puzzle_emb_len: Length of puzzle embedding sequence
        rotary_emb: Optional RoPE positional embeddings
        embed_pos: Optional learned positional embeddings
        H_level: High-level reasoning module
        L_level: Low-level reasoning module
        H_init: Learned initial state for H module
        L_init: Learned initial state for L module
    """

    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(
            self.config.puzzle_emb_ndim // -self.config.hidden_size
        )  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[
                HierarchicalReasoningModel_ACTV1Block(self.config)
                for _i in range(self.config.H_layers)
            ]
        )
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[
                HierarchicalReasoningModel_ACTV1Block(self.config)
                for _i in range(self.config.L_layers)
            ]
        )

        # Initial states
        self.H_init = nn.Buffer(
            truncated_normal_init_(
                torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1
            ),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            truncated_normal_init_(
                torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1
            ),
            persistent=True,
        )

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(
        self, input: torch.Tensor, puzzle_identifiers: torch.Tensor
    ) -> torch.Tensor:
        """Compute input embeddings with token, puzzle, and position encoding.

        Combines three types of embeddings:
        1. Token embeddings for input sequence
        2. Optional sparse puzzle embeddings prepended to sequence
        3. Positional embeddings (learned or RoPE, applied elsewhere)

        Args:
            input: Input token IDs of shape (batch_size, seq_len)
            puzzle_identifiers: Puzzle type IDs of shape (batch_size,)

        Returns:
            Combined embeddings of shape
            (batch_size, seq_len + puzzle_emb_len, hidden_size),
            scaled by sqrt(hidden_size)
        """
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = (
                self.puzzle_emb_len * self.config.hidden_size
                - puzzle_embedding.shape[-1]
            )
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (
                    puzzle_embedding.view(
                        -1, self.puzzle_emb_len, self.config.hidden_size
                    ),
                    embedding,
                ),
                dim=-2,
            )

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (
                embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)
            )

        # Scale
        return self.embed_scale * embedding

    def empty_carry(
        self, batch_size: int
    ) -> HierarchicalReasoningModel_ACTV1InnerCarry:
        """Create uninitialized carry state tensors.

        Allocates empty tensors for H and L hidden states. These are meant
        to be overwritten immediately by reset_carry() with learned initial states.

        Args:
            batch_size: Number of sequences in batch

        Returns:
            InnerCarry with empty z_H and z_L tensors of shape
            (batch_size, seq_len + puzzle_emb_len, hidden_size)
        """
        # Infer device from H_init buffer
        device = self.H_init.device

        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
            z_L=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
                device=device,
            ),
        )

    def reset_carry(
        self,
        reset_flag: torch.Tensor,
        carry: HierarchicalReasoningModel_ACTV1InnerCarry,
    ) -> HierarchicalReasoningModel_ACTV1InnerCarry:
        """Reset carry state to learned initial values based on flags.

        For sequences marked with reset_flag=True (typically halted sequences
        starting new problems), replace their hidden states with learned initial
        states H_init and L_init. Otherwise preserve existing states.

        Args:
            reset_flag: Boolean tensor of shape (batch_size,) indicating which
                sequences to reset
            carry: Current inner carry state

        Returns:
            Updated InnerCarry with selectively reset states
        """
        # Ensure H_init and L_init are on the same device as the carry tensors
        device = carry.z_H.device
        H_init = self.H_init.to(device)  # noqa: N806
        L_init = self.L_init.to(device)  # noqa: N806

        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), L_init, carry.z_L),
        )

    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV1InnerCarry,
        batch: dict[str, torch.Tensor],
    ) -> tuple[
        HierarchicalReasoningModel_ACTV1InnerCarry,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Execute hierarchical reasoning forward pass with 1-step gradient.

        Implements the core computation strategy:
        1. Run (H_cycles × L_cycles - 1) iterations without gradients
        2. Execute final L-level and H-level pass with gradients
        3. Compute language modeling logits and Q-values

        This approach maintains computational depth (many iterations) while
        keeping gradient computation tractable (only 1-step backprop).

        Args:
            carry: Current inner carry state with z_H and z_L
            batch: Dictionary containing:
                - 'inputs': Token IDs of shape (batch_size, seq_len)
                - 'puzzle_identifiers': Puzzle IDs of shape (batch_size,)

        Returns:
            Tuple containing:
                - new_carry: Updated inner carry (detached from gradients)
                - output: Language modeling logits of shape
                    (batch_size, seq_len, vocab_size)
                - (q_halt_logits, q_continue_logits): Q-values for ACT,
                    each of shape (batch_size,)
        """
        seq_info = {
            "cos_sin": self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        }

        # Input encoding
        input_embeddings = self._input_embeddings(
            batch["inputs"], batch["puzzle_identifiers"]
        )

        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L  # noqa: N806

            for _H_step in range(self.config.H_cycles):  # noqa: N806
                for _L_step in range(self.config.L_cycles):  # noqa: N806
                    if not (
                        (_H_step == self.config.H_cycles - 1)
                        and (_L_step == self.config.L_cycles - 1)
                    ):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)  # noqa: N806

                if _H_step != self.config.H_cycles - 1:
                    z_H = self.H_level(z_H, z_L, **seq_info)  # noqa: N806

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)  # noqa: N806
        z_H = self.H_level(z_H, z_L, **seq_info)  # noqa: N806

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=z_H.detach(), z_L=z_L.detach()
        )  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len :]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):  # noqa: N801
    """Hierarchical Reasoning Model with Adaptive Computation Time.

    Top-level wrapper that adds ACT (Adaptive Computation Time) mechanism
    to the hierarchical reasoning model. Manages:
    - Halting logic via Q-learning
    - Carry state updates
    - New data injection for halted sequences
    - Exploration during training

    This is the main model class users interact with. It wraps the inner
    model and handles the ACT loop, allowing sequences to adaptively decide
    when to stop computation.

    Args:
        config_dict: Dictionary of configuration parameters matching
            HierarchicalReasoningModel_ACTV1Config fields

    Attributes:
        config: Validated configuration object
        inner: Core HierarchicalReasoningModel_ACTV1_Inner instance

    Example:
        >>> config = {
        ...     "batch_size": 4,
        ...     "seq_len": 81,
        ...     "vocab_size": 11,
        ...     "hidden_size": 512,
        ...     "num_heads": 8,
        ...     # ... other config
        ... }
        >>> model = HierarchicalReasoningModel_ACTV1(config)
        >>> carry = model.initial_carry(batch)
        >>> carry, outputs = model(carry=carry, batch=batch)
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        """Access to sparse puzzle embeddings for distributed training."""
        return self.inner.puzzle_emb

    def initial_carry(
        self, batch: dict[str, torch.Tensor]
    ) -> HierarchicalReasoningModel_ACTV1Carry:
        """Initialize carry state for a new batch.

        Creates initial state with:
        - Empty inner carry (will be reset on first forward pass)
        - Zero step counts
        - All sequences marked as halted (triggers data loading)
        - Empty current_data placeholders

        Args:
            batch: Initial batch dictionary containing 'inputs' and other fields

        Returns:
            Initial HierarchicalReasoningModel_ACTV1Carry ready for first forward pass
        """
        batch_size = batch["inputs"].shape[0]
        device = batch["inputs"].device  # Get device from batch

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(
                batch_size
            ),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones(
                (batch_size,), dtype=torch.bool, device=device
            ),  # Default to halted
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV1Carry,
        batch: dict[str, torch.Tensor],
    ) -> tuple[HierarchicalReasoningModel_ACTV1Carry, dict[str, torch.Tensor]]:
        """Execute one ACT step with halting logic.

        Performs:
        1. Reset carry for halted sequences (inject new data)
        2. Run inner model forward pass
        3. Compute halting decisions via Q-learning
        4. Apply exploration during training
        5. Compute bootstrapping target for Q-continue

        The halting mechanism uses Q-values:
        - q_halt_logits: Value of stopping computation now
        - q_continue_logits: Value of continuing computation
        - Halt when q_halt > q_continue (with exploration in training)

        Args:
            carry: Current carry state from previous step
            batch: New batch data to inject for halted sequences, containing:
                - 'inputs': Token IDs of shape (batch_size, seq_len)
                - 'labels': Target labels (same shape)
                - 'puzzle_identifiers': Puzzle IDs of shape (batch_size,)

        Returns:
            Tuple containing:
                - new_carry: Updated carry state with:
                    - Updated inner_carry (z_H, z_L)
                    - Incremented steps
                    - Updated halted flags
                    - Current data
                - outputs: Dictionary with:
                    - 'logits': LM predictions (batch, seq_len, vocab_size)
                    - 'q_halt_logits': Halt Q-values (batch,)
                    - 'q_continue_logits': Continue Q-values (batch,)
                    - 'target_q_continue': Bootstrapping target (training only)
        """
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v
            )
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (
                    torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                ) * torch.randint_like(
                    new_steps, low=2, high=self.config.halt_max_steps + 1
                )

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                next_q_halt_logits, next_q_continue_logits = self.inner(
                    new_inner_carry, new_current_data
                )[-1]

                outputs["target_q_continue"] = torch.sigmoid(
                    torch.where(
                        is_last_step,
                        next_q_halt_logits,
                        torch.maximum(next_q_halt_logits, next_q_continue_logits),
                    )
                )

        return HierarchicalReasoningModel_ACTV1Carry(
            new_inner_carry, new_steps, halted, new_current_data
        ), outputs
