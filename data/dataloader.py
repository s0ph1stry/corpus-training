"""
Mixed data loader for Phase 2 training.

Draws from both denoising and generative datasets.
Default 20% denoising batches in Phase 2 to maintain encoder pathway.
Handles collation of mixed encoder_available batches.
"""

import random
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from data.dataset import CorpusDataset


class MixedBatchCollator:
    """Collate function for mixed denoising/generative batches.

    Handles the different return formats:
    - Denoising: encoder_input, decoder_target, encoder_available=1
    - Generative: input_ids, target_ids, encoder_available=0

    Outputs a unified dict compatible with the model's forward().
    """

    def __init__(self, context_len: int, pad_id: int = 0):
        self.context_len = context_len
        self.pad_id = pad_id

    def __call__(self, batch: list) -> dict:
        # Separate by mode
        has_encoder = any(b['encoder_available'].item() > 0.5 for b in batch)
        has_decoder = any(b['encoder_available'].item() < 0.5 for b in batch)

        batch_size = len(batch)
        device = batch[0]['encoder_available'].device if batch else 'cpu'

        # Build unified tensors
        decoder_input_ids = torch.full(
            (batch_size, self.context_len), self.pad_id, dtype=torch.long
        )
        decoder_targets = torch.full(
            (batch_size, self.context_len), -100, dtype=torch.long  # -100 = ignore in CE loss
        )
        encoder_input_ids = torch.full(
            (batch_size, self.context_len), self.pad_id, dtype=torch.long
        )
        encoder_available = torch.zeros(batch_size)
        text_names = []

        for i, sample in enumerate(batch):
            enc_avail = sample['encoder_available'].item()
            encoder_available[i] = enc_avail
            text_names.append(sample.get('text_name', ''))

            if enc_avail > 0.5:
                # Denoising sample
                enc_input = sample['encoder_input']
                dec_target = sample['decoder_target']

                encoder_input_ids[i, :len(enc_input)] = enc_input[:self.context_len]

                # Decoder input = corrupted text (same as encoder input)
                # Decoder target = clean original
                # The model must use the encoder to reconstruct what was corrupted
                decoder_input_ids[i, :len(enc_input)] = enc_input[:self.context_len]
                decoder_targets[i, :len(dec_target)] = dec_target[:self.context_len]
            else:
                # Generative sample
                input_ids = sample['input_ids']
                target_ids = sample['target_ids']

                decoder_input_ids[i, :len(input_ids)] = input_ids[:self.context_len]
                decoder_targets[i, :len(target_ids)] = target_ids[:self.context_len]

        # Padding masks
        decoder_padding_mask = decoder_input_ids != self.pad_id
        encoder_padding_mask = encoder_input_ids != self.pad_id

        # Collect UL2 mode info if available
        ul2_modes = [sample.get('ul2_mode', '') for sample in batch]

        # Convert mode strings to integer IDs for mode-conditioned routing
        # R=0, S=1, X=2 (matches Router.MODE_R/S/X constants)
        mode_map = {'R': 0, 'S': 1, 'X': 2}
        mode_ids = torch.tensor(
            [mode_map.get(m, 0) for m in ul2_modes], dtype=torch.long
        )

        return {
            'decoder_input_ids': decoder_input_ids,
            'decoder_targets': decoder_targets,
            'encoder_input_ids': encoder_input_ids if has_encoder else None,
            'encoder_available': encoder_available,
            'decoder_padding_mask': decoder_padding_mask,
            'encoder_padding_mask': encoder_padding_mask if has_encoder else None,
            'text_names': text_names,
            'ul2_modes': ul2_modes,
            'mode_ids': mode_ids,
        }


class MixedDataLoader:
    """
    Phase 2 data loader that mixes denoising and generative batches.

    Default: 20% denoising to keep the encoder pathway active.
    Phase 1 router collapse mitigation: 10% of denoising batches are forced
    to encoder_available=False to keep Type B routing active.
    """

    def __init__(self,
                 denoise_dataset: CorpusDataset,
                 generate_dataset: CorpusDataset,
                 batch_size: int = 32,
                 denoise_ratio: float = 0.20,
                 num_workers: int = 0,
                 context_len: int = 512,
                 pad_id: int = 0):

        self.denoise_dataset = denoise_dataset
        self.generate_dataset = generate_dataset
        self.batch_size = batch_size
        self.denoise_ratio = denoise_ratio

        collator = MixedBatchCollator(context_len, pad_id)

        self.denoise_loader = DataLoader(
            denoise_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator,
            drop_last=True,
        )
        self.generate_loader = DataLoader(
            generate_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator,
            drop_last=True,
        )

    def __iter__(self):
        denoise_iter = iter(self.denoise_loader)
        generate_iter = iter(self.generate_loader)

        while True:
            try:
                if random.random() < self.denoise_ratio:
                    batch = next(denoise_iter)
                else:
                    batch = next(generate_iter)
                yield batch
            except StopIteration:
                break

    def set_step(self, step: int):
        """Update training step on both datasets."""
        self.denoise_dataset.set_step(step)
        self.generate_dataset.set_step(step)


def create_phase1_loader(project_dir: str, config, batch_size: int = 32,
                          num_workers: int = 0, held_out: set = None,
                          total_steps: int = 50000) -> DataLoader:
    """Create a Phase 1 (denoising) data loader."""
    dataset = CorpusDataset(
        project_dir=project_dir,
        mode='phase1_denoise',
        context_len=config.context_len,
        total_steps=total_steps,
        held_out_texts=held_out,
    )

    collator = MixedBatchCollator(config.context_len, config.pad_token_id)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        drop_last=True,
    )


def create_phase2_loader(project_dir: str, config, batch_size: int = 32,
                          denoise_ratio: float = 0.20,
                          num_workers: int = 0, held_out: set = None,
                          total_steps: int = 30000) -> MixedDataLoader:
    """Create a Phase 2 (mixed denoising + generation) data loader."""
    denoise_ds = CorpusDataset(
        project_dir=project_dir,
        mode='phase1_denoise',
        context_len=config.context_len,
        total_steps=total_steps,
        held_out_texts=held_out,
    )
    generate_ds = CorpusDataset(
        project_dir=project_dir,
        mode='phase2_generate',
        context_len=config.context_len,
        total_steps=total_steps,
        held_out_texts=held_out,
    )

    return MixedDataLoader(
        denoise_ds, generate_ds,
        batch_size=batch_size,
        denoise_ratio=denoise_ratio,
        num_workers=num_workers,
        context_len=config.context_len,
        pad_id=config.pad_token_id,
    )
