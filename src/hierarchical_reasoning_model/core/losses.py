"""Loss functions and training heads for the Hierarchical Reasoning Model."""

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

#: Special label ID used to ignore certain positions in loss computation
IGNORE_LABEL_ID = -100


def stablemax_transform(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    """Stablemax transformation function.

    Applies a smooth transformation that maps negative values to (0, 1)
    and non-negative values to [1, ∞). This is used as part of the
    stablemax normalization for improved numerical stability.

    Args:
        x: Input tensor of any shape
        epsilon: Small constant for numerical stability (default: 1e-30)

    Returns:
        Transformed tensor of same shape as input
    """
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute log of stablemax normalization.

    The stablemax is an alternative to softmax with better numerical
    properties for certain distributions. This function computes the
    logarithm of the stablemax for numerical stability.

    Args:
        x: Input logits tensor
        dim: Dimension along which to normalize (default: -1)

    Returns:
        Log probabilities tensor of same shape as input
    """
    transformed_x = stablemax_transform(x)
    return torch.log(transformed_x / torch.sum(transformed_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Compute cross-entropy loss using stablemax normalization.

    This provides an alternative to standard softmax cross-entropy with
    improved numerical stability for certain types of predictions.

    Args:
        logits: Predicted logits of shape (batch, seq_len, vocab_size)
        labels: True labels of shape (batch, seq_len)
        ignore_index: Label value to ignore in loss computation (default: -100)

    Returns:
        Per-token loss values of shape (batch, seq_len). Positions with
        labels equal to ignore_index will have loss of 0.
    """
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(
        logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1
    ).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Compute standard softmax cross-entropy loss.

    Args:
        logits: Predicted logits of shape (batch, seq_len, vocab_size)
        labels: True labels of shape (batch, seq_len)
        ignore_index: Label value to ignore in loss computation (default: -100)

    Returns:
        Per-token loss values of shape (batch, seq_len). Positions with
        labels equal to ignore_index will have loss of 0.
    """
    return F.cross_entropy(
        logits.to(torch.float32).view(-1, logits.shape[-1]),
        labels.to(torch.long).view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(labels.shape)


class ACTLossHead(nn.Module):
    """Adaptive Computation Time (ACT) loss head wrapper.

    This module wraps a base model and adds three loss components:
    1. Language modeling loss (token prediction)
    2. Q-halt loss (binary classification for when to stop)
    3. Q-continue loss (reinforcement learning for continuation)

    The ACT mechanism allows the model to dynamically decide how many
    computation steps to perform for each input, using Q-learning.

    Args:
        model: Base HRM model to wrap
        loss_type: Type of language modeling loss to use. Must be either
            "stablemax_cross_entropy" or "softmax_cross_entropy"

    Attributes:
        model: The wrapped base model
        loss_fn: The language modeling loss function
    """

    def __init__(self, model: nn.Module, loss_type: str):
        """Initialize the ACT loss head.

        Args:
            model: Base model implementing forward() and initial_carry()
            loss_type: Name of loss function ("stablemax_cross_entropy" or
                "softmax_cross_entropy")

        Raises:
            KeyError: If loss_type is not a valid loss function name
        """
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

    def initial_carry(self, *args: Any, **kwargs: Any) -> Any:
        """Initialize the model's carry state.

        Args:
            *args: Positional arguments passed to model.initial_carry()
            **kwargs: Keyword arguments passed to model.initial_carry()

        Returns:
            Initial carry state from the wrapped model
        """
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs: Any,
    ) -> tuple[
        Any,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor] | None,
        torch.Tensor,
    ]:
        """Forward pass computing losses and metrics.

        Runs the wrapped model forward pass and computes three loss components:
        1. Language modeling loss for token prediction
        2. Q-halt loss for learning when to stop computation
        3. Q-continue loss for reinforcement learning (if applicable)

        Args:
            return_keys: List of output keys to return in the predictions dict.
                Common keys include: "logits", "q_halt_logits", "q_continue_logits"
            **model_kwargs: Keyword arguments passed to the wrapped model,
                typically including "carry" and "batch"

        Returns:
            A tuple containing:
                - new_carry: Updated model carry state
                - total_loss: Sum of all loss components (scalar tensor)
                - metrics: Dictionary of metrics including accuracy, steps, etc.
                - predictions: Optional dictionary of requested outputs (or None)
                - all_halted: Boolean tensor indicating if all sequences halted

        Example:
            >>> loss_head = ACTLossHead(model, "stablemax_cross_entropy")
            >>> carry = loss_head.initial_carry(batch)
            >>> carry, loss, metrics, preds, halted = loss_head.forward(
            ...     return_keys=["logits"],
            ...     carry=carry,
            ...     batch=batch
            ... )
        """
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Correctness
        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(
                -1
            )  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts

            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": torch.where(
                    valid_metrics,
                    (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                    0,
                ).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "q_halt_accuracy": (
                    valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)
                ).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses
        # FIXME: Assuming the batch is always full
        lm_loss = (
            self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID)
            / loss_divisor
        ).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )

        metrics.update(
            {
                "lm_loss": lm_loss.detach(),
                "q_halt_loss": q_halt_loss.detach(),
            }
        )

        # Q continue (bootstrapping target loss)
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum",
            )

            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return (
            new_carry,
            lm_loss + 0.5 * (q_halt_loss + q_continue_loss),
            metrics,
            detached_outputs,
            new_carry.halted.all(),
        )
