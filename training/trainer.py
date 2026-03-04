"""
Training loop for the heterogeneous MoE.

Device-agnostic (MPS/CUDA/CPU). AdamW with cosine decay + warmup.
wandb logging for loss, aux_loss, routing stats, corruption rate, grad norm.

Includes online difficulty adjustment feedback to the dataset.
"""

import contextlib
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from model.config import ModelConfig
from model.model import HeteroMoETransformer
from training.losses import (
    reconstruction_loss, generative_loss, phase2_mixed_loss,
    compute_per_text_loss,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def get_device() -> torch.device:
    """Select best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int,
                                     total_steps: int, min_lr_ratio: float = 0.1):
    """Cosine decay learning rate schedule with linear warmup."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Training loop with logging, checkpointing, and online difficulty adjustment."""

    def __init__(self,
                 model: HeteroMoETransformer,
                 config: ModelConfig,
                 project_dir: str,
                 lr: float = 1e-4,
                 warmup_steps: int = 1000,
                 total_steps: int = 50000,
                 weight_decay: float = 0.1,
                 grad_clip: float = 1.0,
                 checkpoint_every: int = 5000,
                 log_every: int = 100,
                 eval_every: int = 2500,
                 use_wandb: bool = True,
                 wandb_project: str = 'corpus-training',
                 wandb_run_name: Optional[str] = None,
                 phase: str = 'phase1'):

        self.config = config
        self.project_dir = Path(project_dir)
        self.total_steps = total_steps
        self.grad_clip = grad_clip
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every
        self.eval_every = eval_every
        self.phase = phase
        self.global_step = 0

        # Device
        self.device = get_device()
        print(f"Training on: {self.device}")
        self.model = model.to(self.device)

        # Optimizer
        # Separate weight decay for different param groups
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name or 'embedding' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=lr, betas=(0.9, 0.95))

        # LR schedule
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        # Gradient checkpointing
        if config.gradient_checkpointing:
            # Enable gradient checkpointing on encoder and decoder layers
            for layer in model.decoder_layers:
                if hasattr(layer, 'gradient_checkpointing_enable'):
                    layer.gradient_checkpointing_enable()

        # wandb
        self.use_wandb = use_wandb and HAS_WANDB
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name or f'{phase}-{config.d_model}d',
                config={
                    'phase': phase,
                    'd_model': config.d_model,
                    'n_dec_layers': config.n_dec_layers,
                    'n_experts': config.n_experts,
                    'context_len': config.context_len,
                    'lr': lr,
                    'total_steps': total_steps,
                    'warmup_steps': warmup_steps,
                    'device': str(self.device),
                },
            )

        # Automatic mixed precision (significant speedup on CUDA)
        self.use_amp = self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            print("  AMP enabled (float16 on CUDA)")

        # Checkpoint dir
        self.ckpt_dir = self.project_dir / 'checkpoints' / phase
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.running_loss = 0.0
        self.running_aux_loss = 0.0
        self.running_count = 0

    def train_step(self, batch: dict) -> dict:
        """Single training step. Returns loss dict for logging."""
        self.model.train()

        # Move batch to device
        decoder_input_ids = batch['decoder_input_ids'].to(self.device)
        decoder_targets = batch['decoder_targets'].to(self.device)
        encoder_available = batch['encoder_available'].to(self.device)

        encoder_input_ids = None
        encoder_padding_mask = None
        if batch.get('encoder_input_ids') is not None:
            encoder_input_ids = batch['encoder_input_ids'].to(self.device)
        if batch.get('encoder_padding_mask') is not None:
            encoder_padding_mask = batch['encoder_padding_mask'].to(self.device)

        decoder_padding_mask = batch['decoder_padding_mask'].to(self.device)

        # Forward (with AMP autocast on CUDA)
        amp_ctx = torch.amp.autocast('cuda') if self.use_amp else contextlib.nullcontext()
        with amp_ctx:
            output = self.model(
                decoder_input_ids=decoder_input_ids,
                encoder_input_ids=encoder_input_ids,
                encoder_available=encoder_available,
                decoder_padding_mask=decoder_padding_mask,
                encoder_padding_mask=encoder_padding_mask,
            )

            logits = output['logits']
            aux_loss = output['aux_loss']

            # Compute task loss
            if self.phase == 'phase1':
                task_loss = reconstruction_loss(logits, decoder_targets,
                                                pad_id=self.config.pad_token_id)
            else:
                loss_dict = phase2_mixed_loss(logits, decoder_targets,
                                              encoder_available,
                                              pad_id=self.config.pad_token_id)
                task_loss = loss_dict['loss']

            total_loss = task_loss + aux_loss

        # Backward
        self.optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.optimizer.step()

        self.scheduler.step()

        self.global_step += 1
        self.model.global_step = self.global_step  # sync for alpha ramp

        # Per-text loss for online difficulty adjustment
        text_names = batch.get('text_names', [])
        per_text = {}
        if text_names:
            per_text = compute_per_text_loss(
                logits.detach(), decoder_targets, text_names
            )

        # Tracking
        self.running_loss += task_loss.item()
        self.running_aux_loss += aux_loss.item()
        self.running_count += 1

        return {
            'loss': task_loss.item(),
            'aux_loss': aux_loss.item(),
            'total_loss': total_loss.item(),
            'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'lr': self.scheduler.get_last_lr()[0],
            'per_text_loss': per_text,
        }

    def log(self, step_result: dict):
        """Log metrics to wandb and console."""
        if self.global_step % self.log_every != 0:
            return

        avg_loss = self.running_loss / max(self.running_count, 1)
        avg_aux = self.running_aux_loss / max(self.running_count, 1)

        # Routing entropy for console
        entropy_stats = self.model.moe_manager.get_routing_entropy()
        mean_entropy = sum(entropy_stats.values()) / max(len(entropy_stats), 1) if entropy_stats else 0.0

        # Console
        print(f"  step {self.global_step:>6d} | "
              f"loss {avg_loss:.4f} | "
              f"aux {avg_aux:.4f} | "
              f"lr {step_result['lr']:.2e} | "
              f"grad {step_result['grad_norm']:.2f} | "
              f"ent {mean_entropy:.3f}")

        # wandb
        if self.use_wandb:
            log_dict = {
                'loss': avg_loss,
                'aux_loss': avg_aux,
                'lr': step_result['lr'],
                'grad_norm': step_result['grad_norm'],
                'step': self.global_step,
                'balance_alpha': self.model.moe_manager.get_ramped_alpha(self.global_step),
            }
            # Add per-layer routing entropy (Grok patch)
            entropy_stats = self.model.moe_manager.get_routing_entropy()
            log_dict.update(entropy_stats)
            wandb.log(log_dict, step=self.global_step)

        self.running_loss = 0.0
        self.running_aux_loss = 0.0
        self.running_count = 0

    def save_checkpoint(self, extra: dict = None):
        """Save model checkpoint. Also checks for routing collapse."""
        # Check routing collapse (Grok patch)
        if self.model.moe_manager.check_collapse(threshold=0.4, window=3):
            print("  ⚠ ROUTING COLLAPSE DETECTED: entropy < 0.4 for 3 consecutive checkpoints")
            if self.use_wandb:
                wandb.alert(
                    title="Router Collapse",
                    text=f"Routing entropy below 0.4 for 3 consecutive checkpoints at step {self.global_step}",
                    level=wandb.AlertLevel.WARN,
                )

        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'config': self.config.__dict__,
            'entropy_history': self.model.moe_manager.entropy_history,
        }
        if self.use_amp:
            ckpt['scaler_state_dict'] = self.scaler.state_dict()
        if extra:
            ckpt.update(extra)

        path = self.ckpt_dir / f'step_{self.global_step:06d}.pt'
        torch.save(ckpt, path)
        print(f"  Checkpoint saved: {path}")

        # Also save as 'latest'
        latest_path = self.ckpt_dir / 'latest.pt'
        torch.save(ckpt, latest_path)

    def load_checkpoint(self, path: str):
        """Load a checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.global_step = ckpt['global_step']
        if self.use_amp and 'scaler_state_dict' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        print(f"  Loaded checkpoint from step {self.global_step}")

    def get_routing_stats(self) -> dict:
        """Collect routing statistics from all MoE layers."""
        stats = {}
        for i, layer in enumerate(self.model.decoder_layers):
            router = layer.router
            # We'd need to store the last routing decision to report this
            # For now, just check expert types
            stats[f'layer_{i}'] = {
                'n_experts': self.config.n_experts,
                'expert_types': layer.expert_types,
            }
        return stats
