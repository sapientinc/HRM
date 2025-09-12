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
    # Use float32 for MPS compatibility, float64 for other devices
    dtype = torch.float32 if logits.device.type == 'mps' else torch.float64
    logprobs = log_stablemax(logits.to(dtype), dim=-1)

    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    logits_f32 = logits.to(torch.float32)
    labels_long = labels.to(torch.long)
    
    # Use view for CUDA (fastest), reshape for MPS/CPU (compatibility)
    # view() is faster but requires contiguous tensors
    # reshape() handles all cases but has slight overhead
    if logits.is_cuda and logits_f32.is_contiguous() and labels_long.is_contiguous():
        # CUDA with contiguous tensors: use view for best performance
        return F.cross_entropy(
            logits_f32.view(-1, logits.shape[-1]), 
            labels_long.view(-1), 
            ignore_index=ignore_index, 
            reduction="none"
        ).view(labels.shape)
    else:
        # MPS/CPU or non-contiguous: use reshape for compatibility
        return F.cross_entropy(
            logits_f32.reshape(-1, logits.shape[-1]), 
            labels_long.reshape(-1), 
            ignore_index=ignore_index, 
            reduction="none"
        ).reshape(labels.shape)


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
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), torch.zeros_like((is_correct.to(torch.float32) / loss_divisor).sum(-1))).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, torch.zeros_like(new_carry.steps)).sum(),
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
