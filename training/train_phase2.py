#!/usr/bin/env python3
"""
Phase 2: Generative capacity with mixed denoising.

Starts from Phase 1 checkpoint. Autoregressive next-token prediction
with 20% denoising batches to maintain encoder pathway.

The router should learn to switch between reflection (Type A)
and generation (Type B) based on the task.

Usage:
    python -m training.train_phase2                                  # Tiny
    python -m training.train_phase2 --preset small                   # Small
    python -m training.train_phase2 --phase1-ckpt path/to/ckpt.pt   # Specific checkpoint
"""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.config import ModelConfig
from model.model import HeteroMoETransformer
from data.dataloader import create_phase2_loader
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Generative capacity")
    parser.add_argument('--preset', choices=['tiny', 'small'], default='tiny')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--total-steps', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--denoise-ratio', type=float, default=0.20)
    parser.add_argument('--phase1-ckpt', type=str, default=None,
                        help="Phase 1 checkpoint to start from (default: latest)")
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    # Config
    tok_dir = str(PROJECT_DIR / 'tokenizer')
    config = ModelConfig.from_tokenizer(tok_dir, preset=args.preset)

    if args.lr is None:
        args.lr = 5e-5 if args.preset == 'tiny' else 1e-5

    print(f"Phase 2 — {args.preset} model")

    # Load Phase 1 checkpoint
    if args.phase1_ckpt:
        ckpt_path = Path(args.phase1_ckpt)
    else:
        ckpt_path = PROJECT_DIR / 'checkpoints' / 'phase1' / 'latest.pt'

    if not ckpt_path.exists():
        print(f"Error: Phase 1 checkpoint not found: {ckpt_path}")
        print("Run Phase 1 training first.")
        sys.exit(1)

    # Held-out split
    held_out_path = PROJECT_DIR / 'eval' / 'held_out_texts.json'
    with open(held_out_path) as f:
        held_out = set(json.load(f))

    # Data loader
    loader = create_phase2_loader(
        str(PROJECT_DIR), config,
        batch_size=args.batch_size,
        denoise_ratio=args.denoise_ratio,
        held_out=held_out,
        total_steps=args.total_steps,
    )

    # Model from Phase 1 checkpoint
    model = HeteroMoETransformer(config)
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    phase1_step = ckpt.get('global_step', 0)
    print(f"  Loaded Phase 1 checkpoint from step {phase1_step}")
    print(f"  Params: {model.count_parameters()['total_M']:.2f}M")

    # Trainer (fresh optimizer for Phase 2)
    trainer = Trainer(
        model, config, str(PROJECT_DIR),
        lr=args.lr,
        total_steps=args.total_steps,
        use_wandb=not args.no_wandb,
        wandb_run_name=f'phase2-{args.preset}',
        phase='phase2',
    )

    # Training loop
    print(f"\nStarting Phase 2 training ({args.total_steps} steps)")
    print(f"  Denoise ratio: {args.denoise_ratio*100:.0f}%")
    print(f"  LR: {args.lr}")
    print()

    while trainer.global_step < args.total_steps:
        for batch in loader:
            if trainer.global_step >= args.total_steps:
                break

            # Update step per-batch for corruption schedule consistency
            loader.set_step(trainer.global_step)
            result = trainer.train_step(batch)

            # Online difficulty adjustment
            for name, loss in result['per_text_loss'].items():
                loader.generate_dataset.update_online_difficulty(name, loss)

            trainer.log(result)

            if trainer.global_step % trainer.checkpoint_every == 0:
                trainer.save_checkpoint()

    trainer.save_checkpoint()
    print(f"\nPhase 2 training complete. {trainer.global_step} steps.")


if __name__ == '__main__':
    main()
