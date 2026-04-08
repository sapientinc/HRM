from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        # Correctness
        with torch.no_grad():
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses
        # FIXME: Assuming the batch is always full
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        # Q continue (bootstrapping target loss)
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()


class MinLossACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits  — B x SeqLen x Vocab
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]   # (B, S, T)
        logits = outputs["logits"]                   # (B, T, V)

        B, num_solutions, seq_len = labels.shape

        # ── Per-token loss for every solution candidate ──
        logits_exp = logits.unsqueeze(1).expand(-1, num_solutions, -1, -1)  # (B, S, T, V)

        per_token_loss = self.loss_fn(
            logits_exp.reshape(B * num_solutions, seq_len, -1),
            labels.reshape(B * num_solutions, seq_len),
            ignore_index=IGNORE_LABEL_ID,
        ).reshape(B, num_solutions, seq_len)  # (B, S, T)

        # ── Normalised per-solution loss ──
        sol_mask   = labels != IGNORE_LABEL_ID          # (B, S, T)
        sol_counts = sol_mask.sum(-1)                    # (B, S)
        sol_valid  = sol_counts > 0                      # (B, S)

        per_sol_loss = per_token_loss.sum(-1) / sol_counts.clamp_min(1)  # (B, S)
        per_sol_loss = torch.where(sol_valid, per_sol_loss, 1e10)

        # ── Select the min-loss solution per example ──
        best_idx = per_sol_loss.argmin(dim=1)            # (B,)
        bi = torch.arange(B, device=labels.device)

        best_labels = labels[bi, best_idx]               # (B, T)
        has_valid   = sol_valid.any(dim=1)                # (B,)
        best_loss   = torch.where(has_valid, per_sol_loss[bi, best_idx], 0.0)

        # ── Metrics ──
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)         # (B, T)

            # Token-level accuracy against best solution
            best_mask    = best_labels != IGNORE_LABEL_ID
            best_counts  = best_mask.sum(-1)
            best_divisor = best_counts.clamp_min(1).unsqueeze(-1)
            is_correct   = best_mask & (preds == best_labels)

            # Exact-match: correct if prediction matches ANY valid solution
            preds_exp    = preds.unsqueeze(1).expand(-1, num_solutions, -1)  # (B, S, T)
            match_counts = ((preds_exp == labels) & sol_mask).sum(-1)        # (B, S)
            any_seq_correct = ((match_counts == sol_counts) & sol_valid).any(dim=1)  # (B,)

            valid_metrics = new_carry.halted & has_valid
            metrics = {
                "count":          valid_metrics.sum(),
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / best_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & any_seq_correct).sum(),
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == any_seq_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # ── Losses ──
        lm_loss    = best_loss.sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            any_seq_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )

        metrics.update({
            "lm_loss":    lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum"
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()
