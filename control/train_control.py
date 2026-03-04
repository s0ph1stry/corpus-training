#!/usr/bin/env python3
"""
Train the control model on web text.

Identical architecture, identical training schedule, no author tokens,
no curriculum. Same tokenizer (corpus-trained BPE).

The comparison is the experiment's spine: if the corpus model beats the
control on coherence discrimination, it's evidence that text selection
(not just architecture or training time) matters.

Usage:
    python -m control.train_control
    python -m control.train_control --preset small
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.config import ModelConfig
from model.model import HeteroMoETransformer
from training.trainer import Trainer


class WebTextDataset(Dataset):
    """Simple dataset for web text. No curriculum, no author tokens."""

    def __init__(self, data_path: str, tokenizer_path: str,
                 context_len: int = 512, samples_per_epoch: int = 10000):
        self.context_len = context_len
        self.samples_per_epoch = samples_per_epoch

        # Load tokenizer (base, not with-authors)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_id = self.tokenizer.token_to_id('<pad>')
        self.bos_id = self.tokenizer.token_to_id('<bos>')
        self.eos_id = self.tokenizer.token_to_id('<eos>')

        # Load and tokenize web text
        print(f"Loading web text from {data_path}...")
        text = Path(data_path).read_text(encoding='utf-8')
        encoding = self.tokenizer.encode(text)
        self.token_ids = [t for t in encoding.ids
                          if t not in (self.bos_id, self.eos_id)]
        print(f"  {len(self.token_ids):,} tokens")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        import random
        max_start = max(0, len(self.token_ids) - self.context_len - 1)
        start = random.randint(0, max_start)

        window = self.token_ids[start:start + self.context_len + 1]

        input_ids = window[:-1]
        target_ids = window[1:]

        # Pad if needed
        if len(input_ids) < self.context_len:
            pad_len = self.context_len - len(input_ids)
            input_ids = input_ids + [self.pad_id] * pad_len
            target_ids = target_ids + [-100] * pad_len  # ignore in loss

        return {
            'decoder_input_ids': torch.tensor(input_ids[:self.context_len], dtype=torch.long),
            'decoder_targets': torch.tensor(target_ids[:self.context_len], dtype=torch.long),
            'encoder_available': torch.tensor(0.0),
            'decoder_padding_mask': torch.tensor(
                [1 if t != self.pad_id else 0 for t in input_ids[:self.context_len]],
                dtype=torch.bool
            ),
            'text_names': ['web_text'],
        }


def collate_web(batch):
    """Simple collation for web text batches."""
    return {
        'decoder_input_ids': torch.stack([b['decoder_input_ids'] for b in batch]),
        'decoder_targets': torch.stack([b['decoder_targets'] for b in batch]),
        'encoder_available': torch.stack([b['encoder_available'] for b in batch]),
        'decoder_padding_mask': torch.stack([b['decoder_padding_mask'] for b in batch]),
        'encoder_input_ids': None,
        'encoder_padding_mask': None,
        'text_names': [b['text_names'][0] for b in batch],
    }


def main():
    parser = argparse.ArgumentParser(description="Train control model on web text")
    parser.add_argument('--preset', choices=['tiny', 'small'], default='tiny')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--total-steps', type=int, default=80000)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    # Config (use base tokenizer, not with-authors)
    tok_dir = str(PROJECT_DIR / 'tokenizer')
    config = ModelConfig.from_tokenizer(tok_dir, preset=args.preset)

    if args.lr is None:
        args.lr = 1e-4 if args.preset == 'tiny' else 3e-5

    # Check for web text
    data_path = PROJECT_DIR / 'control' / 'data' / 'web_text.txt'
    if not data_path.exists():
        print("Web text not found. Run: python -m control.download_web_text")
        sys.exit(1)

    tok_path = str(PROJECT_DIR / 'tokenizer' / 'tokenizer.json')

    # Dataset
    dataset = WebTextDataset(
        str(data_path), tok_path,
        context_len=config.context_len,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_web,
        drop_last=True,
    )

    # Model (same architecture, no author tokens)
    model = HeteroMoETransformer(config)
    print(f"Control model params: {model.count_parameters()['total_M']:.2f}M")

    # Trainer
    trainer = Trainer(
        model, config, str(PROJECT_DIR),
        lr=args.lr,
        total_steps=args.total_steps,
        use_wandb=not args.no_wandb,
        wandb_run_name=f'control-{args.preset}',
        phase='control',
    )

    print(f"\nTraining control model ({args.total_steps} steps)")

    while trainer.global_step < args.total_steps:
        for batch in loader:
            if trainer.global_step >= args.total_steps:
                break

            result = trainer.train_step(batch)
            trainer.log(result)

            if trainer.global_step % trainer.checkpoint_every == 0:
                trainer.save_checkpoint()

    trainer.save_checkpoint()
    print(f"\nControl training complete. {trainer.global_step} steps.")


if __name__ == '__main__':
    main()
