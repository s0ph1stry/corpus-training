"""
Corpus dataset with curriculum scheduling and two training modes.

Phase 1 (denoise): sample by curriculum weight → extract context window →
    prepend author token → corrupt → return encoder/decoder pair
Phase 2 (generate): same sampling, no corruption, autoregressive targets.
    Sequence packing for short texts. Includes <complete> tokens.

Curriculum pacing: quadratic over first 15% of training, shifting from
easy-weighted to uniform. After warmup, uniform random or online-adjusted.

Online difficulty adjustment: after curriculum warmup, track per-text
rolling loss and upweight texts in the i+1 zone.
"""

import csv
import json
import math
import random
from pathlib import Path
from typing import Optional, Dict, List

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from data.corruption import (
    init_special_tokens, corruption_rate, sample_strategy,
    apply_corruption, sample_ul2_mode, get_ul2_corruption_config,
    sample_strategy_for_mode,
)


class CorpusDataset(Dataset):
    """
    Main dataset for corpus training.

    Modes:
        'phase1_denoise': denoising pretraining (encoder + decoder)
        'phase2_generate': autoregressive generation (decoder only)
    """

    def __init__(self,
                 project_dir: str,
                 mode: str = 'phase1_denoise',
                 context_len: int = 512,
                 samples_per_epoch: int = 10000,
                 curriculum_warmup_frac: float = 0.15,
                 total_steps: int = 50000,
                 held_out_texts: Optional[set] = None):
        """
        Args:
            project_dir: path to corpus-training/
            mode: 'phase1_denoise' or 'phase2_generate'
            context_len: maximum sequence length
            samples_per_epoch: virtual epoch size
            curriculum_warmup_frac: fraction of training for curriculum pacing
            total_steps: total training steps for corruption schedule
            held_out_texts: set of text names to exclude (eval set)
        """
        self.project_dir = Path(project_dir)
        self.mode = mode
        self.context_len = context_len
        self.samples_per_epoch = samples_per_epoch
        self.curriculum_warmup_frac = curriculum_warmup_frac
        self.total_steps = total_steps

        # Load tokenizer (with author tokens)
        tok_path = self.project_dir / 'tokenizer' / 'tokenizer_with_authors.json'
        if not tok_path.exists():
            # Fall back to base tokenizer
            tok_path = self.project_dir / 'tokenizer' / 'tokenizer.json'
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        init_special_tokens(self.tokenizer)

        # Special token IDs
        self.pad_id = self.tokenizer.token_to_id('<pad>')
        self.bos_id = self.tokenizer.token_to_id('<bos>')
        self.eos_id = self.tokenizer.token_to_id('<eos>')
        self.mask_id = self.tokenizer.token_to_id('<mask>')

        # Load config for complete token
        config_path = self.project_dir / 'tokenizer' / 'config_with_authors.json'
        if config_path.exists():
            with open(config_path) as f:
                tok_config = json.load(f)
            self.complete_id = tok_config.get('complete_token_id')
        else:
            self.complete_id = None

        # Load author map
        author_map_path = self.project_dir / 'tokenizer' / 'author_map.json'
        self.author_map = {}
        if author_map_path.exists():
            with open(author_map_path) as f:
                self.author_map = json.load(f)

        # Load difficulty map
        self._load_difficulty_map()

        # Load per-text sampling weight overrides (e.g. Webster's downweighting)
        self._load_sampling_overrides()

        # Load and tokenize all texts
        self._load_texts(held_out_texts)

        # Initialize curriculum weights
        self._init_curriculum_weights()

        # Online difficulty adjustment state
        self._per_text_loss = {}  # text_name -> rolling average loss
        self._loss_update_count = 0
        self._current_step = 0

    def _load_difficulty_map(self):
        """Load difficulty scores from CSV."""
        self.difficulty = {}
        csv_path = self.project_dir / 'difficulty_map.csv'
        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row['name']
                    # Use combined_research if available, fall back to difficulty
                    if 'combined_research' in row and row['combined_research']:
                        self.difficulty[name] = float(row['combined_research'])
                    else:
                        self.difficulty[name] = float(row['difficulty'])

    def _load_sampling_overrides(self):
        """Load per-text sampling weight multipliers.

        Reads sampling_weights.json: {text_name: multiplier}.
        Multiplier of 1.0 = default, 0.1 = 10% of normal sampling rate.
        Used to downweight dominant texts (e.g. Webster's at 24% of tokens).
        """
        self.sampling_overrides = {}
        override_path = self.project_dir / 'sampling_weights.json'
        if override_path.exists():
            with open(override_path) as f:
                self.sampling_overrides = json.load(f)
            n_overrides = len(self.sampling_overrides)
            if n_overrides > 0:
                print(f"Loaded {n_overrides} sampling weight overrides")

    def _load_texts(self, held_out_texts: Optional[set] = None):
        """Load cleaned texts and tokenize them."""
        cleaned_dir = self.project_dir / 'cleaned'
        self.texts = []  # list of dicts with name, token_ids, author_token_id, difficulty, category

        for f in sorted(cleaned_dir.glob('*.txt')):
            name = f.stem
            if held_out_texts and name in held_out_texts:
                continue

            text = f.read_text(encoding='utf-8')
            encoding = self.tokenizer.encode(text)
            # Strip BOS/EOS added by post-processor for raw token storage
            token_ids = [t for t in encoding.ids
                         if t not in (self.bos_id, self.eos_id)]

            if len(token_ids) < 10:
                continue

            # Look up author token
            author_info = self.author_map.get(name, {})
            author_token_id = author_info.get('token_id')

            # Difficulty score
            diff = self.difficulty.get(name, 0.5)

            self.texts.append({
                'name': name,
                'token_ids': token_ids,
                'author_token_id': author_token_id,
                'difficulty': diff,
                'n_tokens': len(token_ids),
            })

        print(f"Loaded {len(self.texts)} texts "
              f"({sum(t['n_tokens'] for t in self.texts):,} tokens)")

    def _init_curriculum_weights(self):
        """Initialize sampling weights based on difficulty scores and overrides."""
        self.base_weights = []
        for t in self.texts:
            # Start heavily weighted toward easy texts
            # (will shift toward uniform during warmup)
            w = 1.0 - t['difficulty']

            # Apply per-text sampling override (e.g. Webster's at 0.1x)
            override = self.sampling_overrides.get(t['name'], 1.0)
            w *= override

            self.base_weights.append(w)

        total = sum(self.base_weights)
        self.base_weights = [w / total for w in self.base_weights]

        # Active weights (modified by online adjustment)
        self.active_weights = list(self.base_weights)

    def _get_sampling_weights(self, step: int) -> List[float]:
        """Get curriculum-adjusted sampling weights for current step.

        Quadratic pacing: during warmup, interpolate from easy-biased to uniform.
        After warmup, use online-adjusted weights.
        """
        warmup_steps = int(self.total_steps * self.curriculum_warmup_frac)

        if step < warmup_steps:
            # Quadratic interpolation from curriculum to uniform
            progress = step / warmup_steps
            alpha = progress ** 2  # quadratic pacing

            # Uniform weights still respect sampling overrides
            overridden_uniform = []
            for t in self.texts:
                override = self.sampling_overrides.get(t['name'], 1.0)
                overridden_uniform.append(override)
            u_total = sum(overridden_uniform)
            overridden_uniform = [w / u_total for w in overridden_uniform]

            weights = [
                (1 - alpha) * cw + alpha * uw
                for cw, uw in zip(self.base_weights, overridden_uniform)
            ]
        else:
            # After warmup: use online-adjusted weights
            weights = list(self.active_weights)

        return weights

    def update_online_difficulty(self, text_name: str, loss: float):
        """Update per-text loss tracking for online difficulty adjustment.

        Called by the trainer after each batch.
        """
        alpha = 0.1  # exponential moving average
        if text_name in self._per_text_loss:
            self._per_text_loss[text_name] = (
                alpha * loss + (1 - alpha) * self._per_text_loss[text_name]
            )
        else:
            self._per_text_loss[text_name] = loss

        self._loss_update_count += 1

        # Recompute weights every 500 loss updates
        if self._loss_update_count % 500 == 0:
            self._recompute_weights()

    def _recompute_weights(self):
        """Recompute sampling weights based on per-text loss (i+1 zone)."""
        if len(self._per_text_loss) < len(self.texts) * 0.5:
            return  # not enough data yet

        losses = list(self._per_text_loss.values())
        if not losses:
            return
        global_mean = sum(losses) / len(losses)

        if global_mean == 0:
            return

        new_weights = []
        for t in self.texts:
            name = t['name']
            if name not in self._per_text_loss:
                new_weights.append(1.0)
                continue

            loss = self._per_text_loss[name]
            ratio = loss / global_mean

            if 0.8 < ratio < 1.5:  # i+1 zone
                new_weights.append(1.3)
            elif ratio > 2.0:  # too hard right now
                new_weights.append(0.7)
            elif ratio < 0.5:  # already mastered
                new_weights.append(0.8)
            else:
                new_weights.append(1.0)

        total = sum(new_weights)
        self.active_weights = [w / total for w in new_weights]

    def set_step(self, step: int):
        """Update current training step (for corruption schedule etc.)."""
        self._current_step = step

    def _sample_text(self) -> dict:
        """Sample a text using current curriculum weights."""
        weights = self._get_sampling_weights(self._current_step)
        idx = random.choices(range(len(self.texts)), weights=weights, k=1)[0]
        return self.texts[idx]

    def _extract_window(self, token_ids: List[int], max_len: int) -> List[int]:
        """Extract a random context window from token IDs."""
        if len(token_ids) <= max_len:
            return list(token_ids)
        start = random.randint(0, len(token_ids) - max_len)
        return token_ids[start:start + max_len]

    def _get_donor_text(self, exclude_name: str) -> List[int]:
        """Get token IDs from a different text for cross-text corruption."""
        candidates = [t for t in self.texts if t['name'] != exclude_name]
        if not candidates:
            return []
        donor = random.choice(candidates)
        return donor['token_ids']

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        if self.mode == 'phase1_denoise':
            return self._get_denoise_sample()
        elif self.mode == 'phase2_generate':
            return self._get_generate_sample()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    @property
    def phase(self) -> str:
        """Extract phase name from mode string."""
        return self.mode.split('_')[0]

    def _get_denoise_sample(self) -> dict:
        """UL2-style denoising sample. Samples R/S/X mode, prepends mode token.

        R-mode: short spans, low corruption → encoder_available=True
        S-mode: autoregressive, no corruption → encoder_available=False
        X-mode: high corruption, long spans → encoder_available=True
        """
        # Sample UL2 mode
        mode = sample_ul2_mode(self._current_step, self.total_steps,
                               phase=self.mode.split('_')[0])
        mode_config = get_ul2_corruption_config(
            mode, self._current_step, self.total_steps
        )

        text = self._sample_text()

        # Reserve space for mode token + author token
        max_content = self.context_len - 3
        window = self._extract_window(text['token_ids'], max_content)

        # Prepend author token if available
        if text['author_token_id'] is not None:
            window = [text['author_token_id']] + window

        if mode == 'S':
            # S-mode: autoregressive generation, no corruption
            # Prepend mode token, then standard next-token prediction
            window = [mode_config['mode_token_id']] + window
            input_ids = window[:-1]
            target_ids = window[1:]

            input_ids = self._pad_or_truncate(input_ids, self.context_len)
            target_ids = self._pad_or_truncate(target_ids, self.context_len)

            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'target_ids': torch.tensor(target_ids, dtype=torch.long),
                'encoder_available': torch.tensor(0.0),
                'text_name': text['name'],
                'ul2_mode': mode,
            }
        else:
            # R or X mode: corruption + reconstruction
            strategy = sample_strategy_for_mode(mode_config)
            rate = mode_config['corruption_rate']

            donor_ids = None
            if strategy == 'cross_text_insert':
                donor_ids = self._get_donor_text(text['name'])

            result = apply_corruption(
                window, strategy, rate,
                tokenizer=self.tokenizer,
                donor_ids=donor_ids,
            )

            corrupted = result['corrupted_ids']

            # Prepend mode token to both corrupted and clean
            corrupted = [mode_config['mode_token_id']] + corrupted
            clean = [mode_config['mode_token_id']] + window

            # Pad/truncate to context_len
            encoder_input = self._pad_or_truncate(corrupted, self.context_len)
            decoder_target = self._pad_or_truncate(clean, self.context_len)

            return {
                'encoder_input': torch.tensor(encoder_input, dtype=torch.long),
                'decoder_target': torch.tensor(decoder_target, dtype=torch.long),
                'encoder_available': torch.tensor(1.0),
                'text_name': text['name'],
                'ul2_mode': mode,
            }

    def _get_generate_sample(self) -> dict:
        """Phase 2: autoregressive generation sample with UL2 mode mixing.

        Samples UL2 mode — mostly S-mode (generation) with some R/X
        maintenance denoising to keep the encoder pathway alive.
        """
        # In Phase 2, UL2 mode sampling shifts toward S-mode
        mode = sample_ul2_mode(self._current_step, self.total_steps,
                               phase='phase2')
        mode_config = get_ul2_corruption_config(
            mode, self._current_step, self.total_steps
        )

        text = self._sample_text()

        if mode == 'S':
            # S-mode: autoregressive generation
            max_content = self.context_len - 3
            window = self._extract_window(text['token_ids'], max_content)

            if text['author_token_id'] is not None:
                window = [text['author_token_id']] + window

            window = [mode_config['mode_token_id']] + window

            input_ids = window[:-1]
            target_ids = window[1:]

            input_ids = self._pad_or_truncate(input_ids, self.context_len)
            target_ids = self._pad_or_truncate(target_ids, self.context_len)

            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'target_ids': torch.tensor(target_ids, dtype=torch.long),
                'encoder_available': torch.tensor(0.0),
                'text_name': text['name'],
                'ul2_mode': mode,
            }
        else:
            # R or X mode: denoising maintenance in Phase 2
            max_content = self.context_len - 3
            window = self._extract_window(text['token_ids'], max_content)

            if text['author_token_id'] is not None:
                window = [text['author_token_id']] + window

            strategy = sample_strategy_for_mode(mode_config)
            rate = mode_config['corruption_rate']

            donor_ids = None
            if strategy == 'cross_text_insert':
                donor_ids = self._get_donor_text(text['name'])

            result = apply_corruption(
                window, strategy, rate,
                tokenizer=self.tokenizer,
                donor_ids=donor_ids,
            )

            corrupted = result['corrupted_ids']
            corrupted = [mode_config['mode_token_id']] + corrupted
            clean = [mode_config['mode_token_id']] + window

            encoder_input = self._pad_or_truncate(corrupted, self.context_len)
            decoder_target = self._pad_or_truncate(clean, self.context_len)

            return {
                'encoder_input': torch.tensor(encoder_input, dtype=torch.long),
                'decoder_target': torch.tensor(decoder_target, dtype=torch.long),
                'encoder_available': torch.tensor(1.0),
                'text_name': text['name'],
                'ul2_mode': mode,
            }

    def _pad_or_truncate(self, ids: List[int], length: int) -> List[int]:
        """Pad with pad_id or truncate to exact length."""
        if len(ids) >= length:
            return ids[:length]
        return ids + [self.pad_id] * (length - len(ids))


def create_held_out_split(project_dir: str, frac: float = 0.10,
                           seed: int = 42) -> set:
    """Create stratified held-out split based on difficulty tiers.

    Saves to eval/held_out_texts.json and returns the set of text names.
    """
    project_dir = Path(project_dir)

    # Load difficulty map for stratification
    difficulty = {}
    csv_path = project_dir / 'difficulty_map.csv'
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                difficulty[row['name']] = {
                    'difficulty': float(row['difficulty']),
                    'tier': int(row.get('tier', 1)),
                }

    # Get all cleaned text names
    cleaned_dir = project_dir / 'cleaned'
    all_names = sorted(f.stem for f in cleaned_dir.glob('*.txt'))

    # Stratified sampling by tier
    rng = random.Random(seed)
    tiers = {}
    for name in all_names:
        tier = difficulty.get(name, {}).get('tier', 1)
        tiers.setdefault(tier, []).append(name)

    held_out = set()
    for tier, names in sorted(tiers.items()):
        n_hold = max(1, int(len(names) * frac))
        rng.shuffle(names)
        held_out.update(names[:n_hold])

    # Save
    eval_dir = project_dir / 'eval'
    eval_dir.mkdir(exist_ok=True)
    out_path = eval_dir / 'held_out_texts.json'
    with open(out_path, 'w') as f:
        json.dump(sorted(held_out), f, indent=2)

    print(f"Held out {len(held_out)} texts ({len(held_out)/len(all_names)*100:.1f}%)")
    for tier in sorted(tiers):
        tier_held = [n for n in held_out if difficulty.get(n, {}).get('tier') == tier]
        print(f"  Tier {tier}: {len(tier_held)}/{len(tiers[tier])}")

    return held_out
