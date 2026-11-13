import json
import logging
import math
import os
import random
import shutil
from dataclasses import dataclass
from typing import Optional, Any, Sequence, List

import coolname
import hydra
import numpy as np
import pydantic
import torch
import torch.distributed as dist
import tqdm
import yaml
from adam_atan2 import AdamATan2
from clearml import Task
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from models.ema import EMAHelper
from models.muon import Muon
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path

logging.basicConfig(level=logging.DEBUG)

torch.use_deterministic_algorithms(True, warn_only=True)

os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['HYDRA_RUN_DIR'] = os.path.expanduser("~/hydra_outputs")
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def set_seed(seed: int, rank: int):
    """Fix all sources of randomness for reproducibility."""
    full_seed = seed + rank  # so different ranks get different sequences
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    torch.cuda.manual_seed(full_seed)
    torch.cuda.manual_seed_all(full_seed)

    # Deterministic cuDNN algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int, seed: int, rank: int):
    """Initialize RNG for each DataLoader worker."""
    worker_seed = seed + worker_id + rank
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    loss: LossConfig


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_path: str

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int
    lr_warmup_ratio: Optional[float] = None

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    tags: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    eval_save_outputs: List[str] = ["inputs", "labels", "puzzle_identifiers", "logits", "q_halt_logits",
                                    "q_continue_logits"]  # !!!!!!!!!

    # precision
    use_bf16: bool = False
    use_tf32: bool = False

    # extra_reason
    extra_steps: int = 8
    count_per_step_metrics: bool = True

    # resilience after training halting
    resume: bool = False

    # gradient accumulation
    accum_steps: int = 1

    # Muon optimizer
    use_muon: bool = False

    # ema
    use_ema: bool = False
    ema_rate: float = 0.999
    ema_start_step: int = 0

    # grad_clip
    grad_clip_norm: Optional[float] = 1.0

    # Weight decay scheduler (optional, backward compatible)
    wd_schedule_start_ratio: Optional[float] = None  # a: start of linear growth (0.0–1.0)
    wd_schedule_end_ratio: Optional[float] = None  # b: end of linear growth (0.0–1.0)
    wd_max: Optional[float] = None

    # lr different scheduling
    lr_scheduler: str = "cosine"  # "cosine" | "polynomial" | "exponential"

    # Polynomial scheduler
    lr_power: float = 2.0

    # LookAhead
    use_lookahead: bool = False
    lookahead_k: int = 6
    lookahead_alpha: float = 0.5

    clearml_task_id: Optional[str] = None


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any
    step: int
    total_steps: int
    accum_counter: int = 0
    ema_helper: Optional[EMAHelper] = None


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_path=config.data_path,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=lambda wid: worker_init_fn(wid, config.seed, rank)
    )

    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model, dynamic=False)  # type: ignore

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    if not config.use_muon:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore

                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,

                world_size=world_size
            ),
            AdamATan2(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
    else:
        print('Muon activated')
        adam_params = [p for p in model.parameters() if p.ndim != 2]
        muon_params = [p for p in model.parameters() if p.ndim == 2]

        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            ),

            Muon([
                {
                    "params": muon_params,
                    "use_muon": True,
                    "lr": 0,  # Needs to be set by scheduler
                },
                {
                    "params": adam_params,
                    "use_muon": False,
                    "lr": 0,  # Needs to be set by scheduler
                    "weight_decay": config.weight_decay,
                    "adamw_betas": (config.beta1, config.beta2),
                    "adamw_eps": 1e-8,
                },
            ]),
        ]

    optimizer_lrs = [
        config.puzzle_emb_lr,
        config.lr
    ]

    # LookAhead Wrapper
    if config.use_lookahead:
        from models.lookahead import LookAheadWrapper
        print(f"[LookAhead] Wrapping {len(optimizers)} optimizers (k={config.lookahead_k})")
        # Wrap each optimizer separately, keeping the list length
        optimizers = [
            LookAheadWrapper(opt, k=config.lookahead_k, alpha=config.lookahead_alpha)
            for opt in optimizers
        ]

    return model, optimizers, optimizer_lrs


def cosine_schedule_with_warmup_lr_lambda(
        current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0,
        num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def exponential_schedule_with_warmup_lr_lambda(
        current_step: int, *,
        base_lr: float,
        num_warmup_steps: int,
        num_training_steps: int,
        min_ratio: float = 0.0
):
    """
    Exponential decay with warmup.

    After warmup, LR is multiplied by a constant c each step, such that:
        base_lr * c^N = base_lr * min_ratio, where N = num_training_steps - num_warmup_steps
    => c = min_ratio^(1/N)

    Final formula: lr = base_lr * min_ratio^progress, where progress ∈ [0,1]
    """
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    # Progress after warmup: 0 → 1
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    progress = min(1.0, max(0.0, progress))  # clamp

    # Protection against min_ratio=0 (log(0) undefined)
    effective_min_ratio = max(min_ratio, 1e-7)

    return base_lr * (effective_min_ratio ** progress)


def polynomial_schedule_with_warmup_lr_lambda(
        current_step: int, *,
        base_lr: float,
        num_warmup_steps: int,
        num_training_steps: int,
        min_ratio: float = 0.0,
        power: float = 2.0
):
    """
    Polynomial decay with warmup.

    Interpolate from base_lr to base_lr * min_ratio using:
        lr = base_lr * [min_ratio + (1 - min_ratio) * (1 - progress)^power]

    At progress=0: lr = base_lr * [min_ratio + (1-min_ratio)*1] = base_lr ✓
    At progress=1: lr = base_lr * [min_ratio + (1-min_ratio)*0] = base_lr * min_ratio ✓
    """
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    progress = min(1.0, max(0.0, progress))  # clamp

    # Decay factor
    decay_factor = (1.0 - progress) ** power

    # Interpolate between 1.0 and min_ratio
    lr_ratio = min_ratio + (1.0 - min_ratio) * decay_factor
    return base_lr * lr_ratio


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, world_size: int):
    # Estimated total training steps
    total_steps = int(
        config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, world_size=world_size)

    ema_helper = None
    if config.use_ema:
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(model)

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
        accum_counter=0,
        ema_helper=ema_helper
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    base_path = os.path.join(config.checkpoint_path, f"step_{train_state.step}")
    torch.save(train_state.model.state_dict(), base_path)

    if train_state.ema_helper is not None:
        torch.save(train_state.ema_helper.state_dict(), f"{base_path}_ema")


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    logical_step = train_state.step // config.accum_steps
    logical_total_steps = train_state.total_steps // config.accum_steps

    if config.lr_warmup_ratio is not None:
        # Compute warmup steps as a fraction of total logical steps
        num_warmup_steps = int(logical_total_steps * config.lr_warmup_ratio)
    else:
        # Backward compatibility: use absolute value
        num_warmup_steps = round(config.lr_warmup_steps)

    # Common arguments
    kwargs = dict(
        current_step=logical_step,
        base_lr=base_lr,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=logical_total_steps,
        min_ratio=config.lr_min_ratio
    )

    scheduler_type = getattr(config, 'lr_scheduler', 'cosine')  # backward compat

    if scheduler_type == "cosine":
        return cosine_schedule_with_warmup_lr_lambda(**kwargs)
    elif scheduler_type == "polynomial":
        return polynomial_schedule_with_warmup_lr_lambda(
            **kwargs,
            power=getattr(config, 'lr_power', 2.0)
        )
    elif scheduler_type == "exponential":
        # Optional overridden min_ratio for exponential
        min_ratio = getattr(config, 'lr_exp_min_ratio', None)
        if min_ratio is not None:
            kwargs['min_ratio'] = min_ratio
        return exponential_schedule_with_warmup_lr_lambda(**kwargs)
    else:
        # Fallback + warning
        import logging
        logging.warning(f"Unknown lr_scheduler '{scheduler_type}', falling back to 'cosine'")
        return cosine_schedule_with_warmup_lr_lambda(**kwargs)


def wd_schedule_lambda(
        current_step: int, *,
        base_wd: float,
        num_training_steps: int,
        start_ratio: Optional[float] = None,
        end_ratio: Optional[float] = None,
        wd_max: Optional[float] = None
):
    """
    Weight decay schedule:
    - [0, start_ratio): constant base_wd
    - [start_ratio, end_ratio): linear increase base_wd → wd_max
    - [end_ratio, 1.0]: constant wd_max

    Backward compatible: if any schedule param is None, returns base_wd.
    """
    if start_ratio is None or end_ratio is None or wd_max is None:
        return base_wd

    start_ratio = max(0.0, min(1.0, start_ratio))
    end_ratio = max(0.0, min(1.0, end_ratio))

    progress = current_step / max(1, num_training_steps)

    if progress < start_ratio:
        return base_wd
    elif progress < end_ratio:
        ratio = (progress - start_ratio) / max(1e-8, end_ratio - start_ratio)
        return base_wd + ratio * (wd_max - base_wd)
    else:
        return wd_max


def compute_wd(config: PretrainConfig, train_state: TrainState):
    """
    Compute weight decay for current step using optional linear ramp-up schedule.
    If wd_schedule_* params are not set, returns constant base_wd (backward compatible).
    """
    logical_step = train_state.step // config.accum_steps
    logical_total_steps = train_state.total_steps // config.accum_steps

    return wd_schedule_lambda(
        current_step=logical_step,
        base_wd=config.weight_decay,
        num_training_steps=logical_total_steps,
        start_ratio=config.wd_schedule_start_ratio,
        end_ratio=config.wd_schedule_end_ratio,
        wd_max=config.wd_max
    )


def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int,
                world_size: int):
    train_state.step += 1
    train_state.accum_counter += 1

    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return None

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config.use_bf16):
        train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / (global_batch_size * config.accum_steps)) * loss).backward()

    is_accum_step = (train_state.accum_counter == config.accum_steps)

    lr_this_step = None
    grad_norm = None

    if is_accum_step:
        # Reduce once per optimizer update. Reducing after every microbatch would
        # repeatedly sum gradients already synchronized by earlier microbatches.
        if world_size > 1:
            for param in train_state.model.parameters():
                if param.requires_grad:
                    # Keep collective calls aligned for conditionally unused parameters.
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                    dist.all_reduce(param.grad)

        # Compute gradient norm (and optionally clip)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            train_state.model.parameters(),
            max_norm=config.grad_clip_norm if config.grad_clip_norm is not None else float('inf'),
            norm_type=2.0  # L2 norm
        )

        # Apply optimizer
        for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
            lr_this_step = compute_lr(base_lr, config, train_state)
            wd_this_step = compute_wd(config, train_state)

            for param_group in optim.param_groups:
                param_group['lr'] = lr_this_step
                if 'weight_decay' in param_group:
                    param_group['weight_decay'] = wd_this_step

            optim.step()
            optim.zero_grad()
        train_state.accum_counter = 0

        if train_state.ema_helper is not None and train_state.step >= config.ema_start_step:
            train_state.ema_helper.update(train_state.model)

    # Reduce metrics
    if len(metrics) and is_accum_step:
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in
                               reduced_metrics.items()}

            if lr_this_step is not None:
                reduced_metrics["train/lr"] = lr_this_step
            if grad_norm is not None:
                reduced_metrics["train/grad_norm"] = grad_norm.item()

            return reduced_metrics


def count_batch_metrics(predsn, origs):
    exact = 0

    for i in range(len(origs)):
        mask = origs[i] != -100
        exact += np.all(origs[i][mask] == predsn[i][mask])

    return exact, len(origs)


def evaluate(config: PretrainConfig, train_state: TrainState, eval_loader: torch.utils.data.DataLoader,
             eval_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    with torch.inference_mode():
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        all_preds = {}
        metric_keys = []
        metric_values = None
        metric_global_batch_size = [0 for _ in range(len(set_ids))]

        # Prepare per-step metrics if required
        if config.count_per_step_metrics:
            old_halt_max_steps = train_state.model.model.config.halt_max_steps
            extra_steps = config.extra_steps
            train_state.model.model.config.halt_max_steps += extra_steps
            sets_per_step = {}

        carry = None
        for set_name, batch, global_batch_size in eval_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)

            # Branch forward logic depending on flag
            if config.count_per_step_metrics:
                # Initialize structures for current dataset
                if set_name not in sets_per_step:
                    local_per_step_metrics = {
                        str(i): [0, 0] for i in range(1, train_state.model.model.config.halt_max_steps + 1)
                    }
                    local_per_step_metrics["best_orig"] = [0, 0]
                    local_per_step_metrics["best_full"] = [0, 0]
                    sets_per_step[set_name] = local_per_step_metrics

                batch_size = batch["labels"].shape[0]
                stopped = np.zeros((batch_size,), dtype=bool)
                best_preds = batch["labels"].detach().cpu().numpy().copy()
                origs = batch["labels"].detach().cpu().numpy()

                step_counter = 0
                while True:
                    carry, _, metrics, preds, all_finish = train_state.model(
                        carry=carry, batch=batch, return_keys=config.eval_save_outputs
                    )
                    step_counter += 1

                    try:
                        want_stop = (preds['q_halt_logits'] > preds['q_continue_logits']).detach().cpu().numpy()
                    except Exception:
                        print(preds)
                        print(batch)

                    need_to_stop = (~stopped) & want_stop
                    predsn = preds["logits"].argmax(dim=2).detach().cpu().numpy()
                    best_preds[need_to_stop] = predsn[need_to_stop]
                    stopped |= need_to_stop

                    # Metric at current step
                    exact, N = count_batch_metrics(predsn, origs)
                    key = str(step_counter)
                    if key in sets_per_step[set_name]:
                        sets_per_step[set_name][key][0] += exact
                        sets_per_step[set_name][key][1] += N

                    # Metric at original step limit
                    if step_counter == old_halt_max_steps:
                        need_to_stop_orig = (~stopped)
                        best_preds_orig = best_preds.copy()
                        best_preds_orig[need_to_stop_orig] = predsn[need_to_stop_orig]
                        exact, N = count_batch_metrics(best_preds_orig, origs)
                        sets_per_step[set_name]["best_orig"][0] += exact
                        sets_per_step[set_name]["best_orig"][1] += N

                    if all_finish:
                        need_to_stop = (~stopped)
                        best_preds[need_to_stop] = predsn[need_to_stop]
                        break

                # Metric at full stop (with extra steps)
                exact, N = count_batch_metrics(best_preds, origs)
                sets_per_step[set_name]["best_full"][0] += exact
                sets_per_step[set_name]["best_full"][1] += N

            else:
                # Normal loop without per-step analysis (as in original version)
                while True:
                    carry, _, metrics, preds, all_finish = train_state.model(
                        carry=carry, batch=batch, return_keys=config.eval_save_outputs
                    )
                    if all_finish:
                        break

            # Save predictions (common part for both modes)
            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        all_preds.setdefault(k, []).append(v.cpu())

            del carry, preds, batch, all_finish

            # Aggregate standard model metrics
            set_id = set_ids[set_name]
            if metric_values is None:
                metric_keys = list(sorted(metrics.keys()))
                metric_values = torch.zeros((len(set_ids), len(metrics)), dtype=torch.float32, device="cuda")
            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])
            metric_global_batch_size[set_id] += global_batch_size

        # Restore original step limit if changed
        if config.count_per_step_metrics:
            train_state.model.model.config.halt_max_steps = old_halt_max_steps

        # Save predictions to disk
        if len(all_preds) and config.checkpoint_path is not None:
            all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
            os.makedirs(config.checkpoint_path, exist_ok=True)
            torch.save(all_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}"))

        # Aggregate per-step metrics (only when config.count_per_step_metrics is True)
        new_metrics = None
        if config.count_per_step_metrics:
            if world_size > 1:
                gathered_metrics = [None] * world_size if rank == 0 else None
                dist.gather_object(sets_per_step, gathered_metrics, dst=0)
            else:
                gathered_metrics = [sets_per_step]

            if rank == 0:
                aggregated = {}
                for instance_sets_per_step in gathered_metrics:
                    for set_ in instance_sets_per_step:
                        if set_ not in aggregated:
                            aggregated[set_] = {step: [0, 0] for step in instance_sets_per_step[set_]}
                        for step, values in instance_sets_per_step[set_].items():
                            for i in range(2):
                                aggregated[set_][step][i] += values[i]

                new_metrics = {set_: {} for set_ in aggregated.keys()}
                for set_ in aggregated.keys():
                    for step, (exact, total) in aggregated[set_].items():
                        new_metrics[set_][step] = {
                            "exact": exact / total if total > 0 else 0.0,
                        }

        # Aggregate standard metrics (unchanged)
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }
                for set_name, metrics in reduced_metrics.items():
                    count = metrics.pop("count", 1.0)
                    reduced_metrics[set_name] = {k: v / count for k, v in metrics.items()}

                # Always return tuple (reduced_metrics, new_metrics)
                return reduced_metrics, new_metrics

        # For non-rank 0 or when metrics are absent
        return None, None


def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_path).capitalize()} ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]


def save_resume_checkpoint(config: PretrainConfig, train_state: TrainState, rank: int, dataset_iters: int = 0):
    """
    Save full state for resume.
    """
    if config.checkpoint_path is None or rank != 0:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    checkpoint_path = os.path.join(config.checkpoint_path, "latest_resume.pt")
    temp_path = checkpoint_path + ".tmp"

    try:
        checkpoint = {
            "step": train_state.step,
            "total_steps": train_state.total_steps,
            "model": train_state.model.state_dict(),
            "optimizers": [opt.state_dict() for opt in train_state.optimizers],
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            },
            "dataset_iters": dataset_iters,
            "clearml_task_id": config.clearml_task_id,
        }
        if train_state.ema_helper is not None:
            checkpoint["ema_shadow"] = train_state.ema_helper.state_dict()

        torch.save(checkpoint, temp_path)
        os.replace(temp_path, checkpoint_path)
    except Exception as e:
        print(f"[Rank {rank}] Warning: Failed to save resume checkpoint: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def try_load_resume_checkpoint(config: PretrainConfig, train_state: TrainState, rank: int, world_size: int) \
        -> tuple[bool, int, Any]:
    """
    Attempt to load state. Each GPU loads from disk independently.
    Returns (success: bool, dataset_iters: int).
    """
    if config.checkpoint_path is None:
        print("No checkpoint path")
        return False, 0, None

    checkpoint_path = os.path.join(config.checkpoint_path, "latest_resume.pt")

    if not os.path.exists(checkpoint_path):
        print("No resume")
        return False, 0, None

    try:
        print(f"[Rank {rank}] Loading resume checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    except Exception as e:
        print(f"[Rank {rank}] Warning: Failed to load resume checkpoint: {e}")
        return False, 0, None

    # Restore on each GPU independently
    train_state.step = checkpoint["step"]
    train_state.total_steps = checkpoint["total_steps"]
    train_state.model.load_state_dict(checkpoint["model"])

    for opt, state in zip(train_state.optimizers, checkpoint["optimizers"]):
        opt.load_state_dict(state)

    rng = checkpoint["rng"]
    torch.set_rng_state(rng["torch"].cpu())
    torch.cuda.set_rng_state(rng["cuda"].cpu())
    np.random.set_state(rng["numpy"])

    if "random" in rng:
        random.setstate(rng["random"])

    dataset_iters = checkpoint.get("dataset_iters", 0)
    clearml_task_id = checkpoint.get("clearml_task_id", None)

    if train_state.ema_helper is not None and "ema_shadow" in checkpoint:
        train_state.ema_helper.load_state_dict(checkpoint["ema_shadow"])
        print(f"✓ EMA weights loaded from {checkpoint_path}")

    print(f"[Rank {rank}] ✓ Resumed from step {train_state.step}, dataset_iters={dataset_iters}")
    return True, dataset_iters, clearml_task_id


def load_dataset_metadata(dataset_path: str, split: str) -> PuzzleDatasetMetadata:
    """
    Load only dataset metadata from JSON, without initializing the dataset itself.
    """

    with open(os.path.join(dataset_path, split, "dataset.json"), "r") as f:
        return PuzzleDatasetMetadata(**json.load(f))


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    set_seed(config.seed, RANK)

    if config.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True  # Speeds up residual FP32 operations
        torch.backends.cudnn.allow_tf32 = True

    train_metadata = load_dataset_metadata(config.data_path, "train")

    # Initialize train_state
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    # Attempt to resume (updates train_state.step)
    dataset_iters = 0
    if config.resume:
        loaded, dataset_iters, loaded_task_id = try_load_resume_checkpoint(config, train_state, RANK, WORLD_SIZE)

        if RANK == 0 and loaded_task_id:
            config.clearml_task_id = loaded_task_id

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    steps_per_iter = int(train_epochs_per_iter * train_metadata.total_groups *
                         train_metadata.mean_puzzle_examples / config.global_batch_size)
    start_iter_id = train_state.step // steps_per_iter if steps_per_iter > 0 else 0

    if RANK == 0 and start_iter_id > 0:
        print(f"Skipping processed iterations: starting at iter_id={start_iter_id}/{total_iters}")

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False,
                                                     epochs_per_iter=train_epochs_per_iter,
                                                     global_batch_size=config.global_batch_size, rank=RANK,
                                                     world_size=WORLD_SIZE, initial_iters=dataset_iters)
    eval_loader, eval_metadata = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1,
                                                   global_batch_size=config.global_batch_size, rank=RANK,
                                                   world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        print(config)
        progress_bar = tqdm.tqdm(total=train_state.total_steps, initial=train_state.step)

        task = None

        if config.clearml_task_id:
            # Resume existing task
            try:
                task = Task.get_task(task_id=config.clearml_task_id)
                print(f"✓ Resumed ClearML task {config.clearml_task_id}")
            except Exception as e:
                print(f"Failed to get existing task {config.clearml_task_id}, creating new one. Error: {e}")

        # If task could not be restored, create a new one
        if task is None:
            task = Task.init(
                project_name=config.project_name or "Default Project",
                task_name=config.run_name or f"Run_{coolname.generate_slug(2)}",
                tags=[] if config.tags is None else config.tags.split(),
                auto_connect_frameworks=True,
                auto_resource_monitoring=False
            )
            config.clearml_task_id = task.id

            conveyor_info = dict()
            if total_gpu_number := os.environ.get('MLS_JOB_TOTAL_GPU', None):
                conveyor_info["world_size"] = total_gpu_number
            if region := os.environ.get('MLS_JOB_REGION_NAME', None):
                conveyor_info["region"] = region

            task.connect(
                conveyor_info,
                name="Conveyor_info"
            )

            # Log number of parameters
            task.get_logger().report_scalar(
                title="model",
                series="num_params",
                value=sum(x.numel() for x in train_state.model.parameters()),
                iteration=0
            )

            save_code_and_config(config)

    # Training Loop
    for _iter_id in range(start_iter_id, total_iters):
        print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        train_state.model.train()

        steps_per_epoch = train_state.total_steps / config.epochs
        for batch_num, (set_name, batch, global_batch_size) in enumerate(train_loader):
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0:
                task.get_logger().report_scalar(
                    title="progress",
                    series="epoch",
                    value=int(train_state.step / steps_per_epoch),
                    iteration=train_state.step
                )

            if RANK == 0 and metrics is not None and batch_num % 5 == 0:
                for metric_name, metric_value in metrics.items():
                    task.get_logger().report_scalar(
                        title=metric_name.split('/')[0] if '/' in metric_name else metric_name,
                        series=metric_name,
                        value=metric_value,
                        iteration=train_state.step
                    )

                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore

        # Evaluation
        train_state.model.eval()

        models_to_eval = [("main", train_state.model)]

        if train_state.ema_helper is not None:
            ema_model = train_state.ema_helper.ema_copy(train_state.model)
            ema_model.eval()
            models_to_eval.append(("ema", ema_model))

        for model_prefix, model_to_eval in models_to_eval:
            # Temporarily replace model
            original_model = train_state.model
            train_state.model = model_to_eval

            metrics, new_metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK,
                                            world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                prefix = "" if model_prefix == "main" else f"{model_prefix}/"

                for dataset_name, dataset_metrics in metrics.items():
                    for metric_name, metric_value in dataset_metrics.items():
                        task.get_logger().report_scalar(
                            title=f"{prefix}{dataset_name}",
                            series=metric_name,
                            value=metric_value,
                            iteration=train_state.step
                        )

            if RANK == 0 and new_metrics is not None:
                prefix = "" if model_prefix == "main" else f"{model_prefix}/"

                for set_ in new_metrics.keys():
                    for metric_name in ["exact"]:
                        for step_name, values in new_metrics[set_].items():
                            if "best" in str(step_name):
                                series = step_name
                            else:
                                series = f"step_{step_name}"
                            task.get_logger().report_scalar(
                                title=f"{prefix}{set_}/{metric_name}",
                                series=series,
                                iteration=train_state.step,
                                value=values[metric_name]
                            )

            train_state.model = original_model

            # Cleanup EMA model
            if model_prefix == "ema":
                del ema_model
                torch.cuda.empty_cache()

        # Checkpointing
        if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
            save_train_state(config, train_state)

            # ADD-ON: Save for resume (latest)
            steps_per_iter = int(train_epochs_per_iter * train_metadata.total_groups *
                                 train_metadata.mean_puzzle_examples / config.global_batch_size)
            dataset_iters = train_state.step // steps_per_iter if steps_per_iter > 0 else 0

            save_resume_checkpoint(config, train_state, RANK, dataset_iters=dataset_iters)

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    if RANK == 0:
        task.close()


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~"))
    launch()
