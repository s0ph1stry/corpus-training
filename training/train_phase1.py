#!/usr/bin/env python3
"""
Phase 1: Denoising pretraining.

T5-style span masking + progressive corruption on the curated corpus.
Encoder-decoder pathway dominant. Router learns to use Type A experts
for reconstruction tasks.

Collapse mitigation: 10% of batches forced to encoder_available=False
to keep Type B routing active.

Usage:
    python -m training.train_phase1                      # Tiny model
    python -m training.train_phase1 --preset small       # Small model
    python -m training.train_phase1 --resume latest      # Resume from checkpoint
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.config import ModelConfig, TinyConfig, SmallConfig
from model.model import HeteroMoETransformer
from data.dataset import CorpusDataset, create_held_out_split
from data.dataloader import create_phase1_loader
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Denoising pretraining")
    parser.add_argument('--preset', choices=['tiny', 'small'], default='tiny')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--total-steps', type=int, default=50000)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help="Checkpoint path or 'latest'")
    parser.add_argument('--weights-only', action='store_true',
                        help="When resuming, only load model weights (reset optimizer/scheduler)")
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help="Override checkpoint directory (e.g. Google Drive path)")
    args = parser.parse_args()

    # Config
    tok_dir = str(PROJECT_DIR / 'tokenizer')
    config = ModelConfig.from_tokenizer(tok_dir, preset=args.preset)

    # Default LR by preset
    if args.lr is None:
        args.lr = 3e-5 if args.preset == 'tiny' else 1e-5

    print(f"Phase 1 — {args.preset} model")
    params = config.estimate_params()
    print(f"  Estimated params: {params['total_M']:.2f}M")
    print(f"  d_model={config.d_model}, dec_layers={config.n_dec_layers}, "
          f"experts={config.n_experts}")

    # Held-out split
    held_out_path = PROJECT_DIR / 'eval' / 'held_out_texts.json'
    if held_out_path.exists():
        with open(held_out_path) as f:
            held_out = set(json.load(f))
        print(f"  Using existing held-out split: {len(held_out)} texts")
    else:
        held_out = create_held_out_split(str(PROJECT_DIR))

    # Data loader
    loader = create_phase1_loader(
        str(PROJECT_DIR), config,
        batch_size=args.batch_size,
        held_out=held_out,
        total_steps=args.total_steps,
    )

    # Model
    model = HeteroMoETransformer(config)
    actual_params = model.count_parameters()
    print(f"  Actual params: {actual_params['total_M']:.2f}M")

    # Trainer
    trainer = Trainer(
        model, config, str(PROJECT_DIR),
        lr=args.lr,
        total_steps=args.total_steps,
        checkpoint_every=args.checkpoint_every,
        use_wandb=not args.no_wandb,
        wandb_run_name=f'phase1-{args.preset}',
        phase='phase1',
        checkpoint_dir=args.checkpoint_dir,
    )

    # Resume if requested
    if args.resume:
        if args.resume == 'latest':
            ckpt_path = PROJECT_DIR / 'checkpoints' / 'phase1' / 'latest.pt'
        else:
            ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            trainer.load_checkpoint(str(ckpt_path), weights_only=args.weights_only)
        else:
            print(f"  Warning: checkpoint not found: {ckpt_path}")

    # Training loop
    print(f"\nStarting Phase 1 training ({args.total_steps} steps)")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}")
    print(f"  UL2 mode mixing: R=50% S=25% X=25%")
    print(f"  (S-mode provides Type B gradient — no separate collapse mitigation needed)")
    print()

    dataset = loader.dataset

    while trainer.global_step < args.total_steps:
        for batch in loader:
            if trainer.global_step >= args.total_steps:
                break

            # Update dataset step for corruption schedule + UL2 mode ratios
            dataset.set_step(trainer.global_step)

            # Train step
            result = trainer.train_step(batch)

            # Online difficulty adjustment
            for name, loss in result['per_text_loss'].items():
                dataset.update_online_difficulty(name, loss)

            # Logging
            trainer.log(result)

            # Checkpointing
            if trainer.global_step % trainer.checkpoint_every == 0:
                trainer.save_checkpoint()

    # Final checkpoint
    trainer.save_checkpoint()
    print(f"\nPhase 1 training complete. {trainer.global_step} steps.")


if __name__ == '__main__':
    main()
