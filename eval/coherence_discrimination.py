"""
Coherence discrimination — the PRIMARY evaluation metric.

Tests whether the model can distinguish original passages from degraded ones.
If the model has learned structural coherence (not just fluency), it should
assign lower loss to the original than to structurally degraded versions.

Degradation types:
  - structural_removal: remove a load-bearing element
  - unearned_conclusion: replace conclusion with non-sequitur
  - register_shift: inject text from wrong register
  - cross_text_insert: insert foreign passage (no <corrupt> marker at eval)
  - paragraph_swap: swap paragraph order

The model's preference = lower loss on original.
Run on both corpus model and control for comparison.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Tuple

import torch
from tokenizers import Tokenizer

from model.model import HeteroMoETransformer
from model.config import ModelConfig


def _paragraphs(text: str) -> List[str]:
    """Split text into paragraphs."""
    paras = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
    return paras


def degrade_structural_removal(paragraphs: List[str]) -> str:
    """Remove a paragraph from the middle (structural element removal)."""
    if len(paragraphs) < 3:
        return '\n\n'.join(paragraphs)
    # Remove a middle paragraph
    mid = len(paragraphs) // 2
    degraded = paragraphs[:mid] + paragraphs[mid + 1:]
    return '\n\n'.join(degraded)


def degrade_unearned_conclusion(paragraphs: List[str],
                                 all_texts: List[str]) -> str:
    """Replace the final paragraph with one from a different text."""
    if len(paragraphs) < 2 or not all_texts:
        return '\n\n'.join(paragraphs)
    donor = random.choice(all_texts)
    donor_paras = _paragraphs(donor)
    if not donor_paras:
        return '\n\n'.join(paragraphs)
    replacement = random.choice(donor_paras)
    return '\n\n'.join(paragraphs[:-1] + [replacement])


def degrade_register_shift(paragraphs: List[str],
                            all_texts: List[str]) -> str:
    """Replace a middle paragraph with one from a very different register."""
    if len(paragraphs) < 3 or not all_texts:
        return '\n\n'.join(paragraphs)
    mid = len(paragraphs) // 2
    donor = random.choice(all_texts)
    donor_paras = _paragraphs(donor)
    if not donor_paras:
        return '\n\n'.join(paragraphs)
    replacement = random.choice(donor_paras)
    degraded = paragraphs[:mid] + [replacement] + paragraphs[mid + 1:]
    return '\n\n'.join(degraded)


def degrade_cross_text_insert(paragraphs: List[str],
                                all_texts: List[str]) -> str:
    """Insert a foreign paragraph without any marker (harder at eval)."""
    if not paragraphs or not all_texts:
        return '\n\n'.join(paragraphs)
    donor = random.choice(all_texts)
    donor_paras = _paragraphs(donor)
    if not donor_paras:
        return '\n\n'.join(paragraphs)
    insert = random.choice(donor_paras)
    pos = random.randint(1, max(1, len(paragraphs) - 1))
    degraded = paragraphs[:pos] + [insert] + paragraphs[pos:]
    return '\n\n'.join(degraded)


def degrade_paragraph_swap(paragraphs: List[str]) -> str:
    """Swap two non-adjacent paragraphs."""
    if len(paragraphs) < 3:
        return '\n\n'.join(paragraphs)
    indices = list(range(len(paragraphs)))
    i, j = random.sample(indices, 2)
    degraded = list(paragraphs)
    degraded[i], degraded[j] = degraded[j], degraded[i]
    return '\n\n'.join(degraded)


DEGRADATION_FNS = {
    'structural_removal': degrade_structural_removal,
    'unearned_conclusion': degrade_unearned_conclusion,
    'register_shift': degrade_register_shift,
    'cross_text_insert': degrade_cross_text_insert,
    'paragraph_swap': degrade_paragraph_swap,
}


def _compute_passage_loss(model: HeteroMoETransformer, tokenizer: Tokenizer,
                           text: str, context_len: int,
                           device: torch.device) -> float:
    """Compute mean token loss for a passage."""
    encoding = tokenizer.encode(text)
    bos_id = tokenizer.token_to_id('<bos>')
    eos_id = tokenizer.token_to_id('<eos>')
    token_ids = [t for t in encoding.ids if t not in (bos_id, eos_id)]

    if len(token_ids) < 2:
        return float('inf')

    # Truncate to context length
    token_ids = token_ids[:context_len]

    input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
    target_ids = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        output = model(decoder_input_ids=input_ids)
        logits = output['logits']
        B, S, V = logits.shape
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, V), target_ids.view(-1), reduction='mean'
        )

    return loss.item()


def evaluate_coherence_discrimination(
        model: HeteroMoETransformer,
        config: ModelConfig,
        project_dir: str,
        device: torch.device,
        n_pairs: int = 50,
        seed: int = 42) -> dict:
    """
    Run coherence discrimination evaluation.

    Creates original/degraded pairs and checks if the model prefers the original.
    """
    project_dir = Path(project_dir)
    rng = random.Random(seed)

    # Load tokenizer
    tok_path = project_dir / 'tokenizer' / 'tokenizer_with_authors.json'
    if not tok_path.exists():
        tok_path = project_dir / 'tokenizer' / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tok_path))

    # Load held-out texts
    held_out_path = project_dir / 'eval' / 'held_out_texts.json'
    with open(held_out_path) as f:
        held_out_names = json.load(f)

    # Load texts
    all_held_out_texts = []
    held_out_by_name = {}
    for name in held_out_names:
        path = project_dir / 'cleaned' / f'{name}.txt'
        if path.exists():
            text = path.read_text(encoding='utf-8')
            all_held_out_texts.append(text)
            held_out_by_name[name] = text

    # Also load some non-held-out texts for donor passages
    all_texts = []
    for f in sorted((project_dir / 'cleaned').glob('*.txt')):
        all_texts.append(f.read_text(encoding='utf-8'))

    results_by_type = {dtype: {'correct': 0, 'total': 0, 'margins': []}
                       for dtype in DEGRADATION_FNS}

    pairs_per_type = max(1, n_pairs // len(DEGRADATION_FNS))

    for dtype, degrade_fn in DEGRADATION_FNS.items():
        for _ in range(pairs_per_type):
            # Pick a random held-out text
            name = rng.choice(held_out_names)
            if name not in held_out_by_name:
                continue
            text = held_out_by_name[name]
            paragraphs = _paragraphs(text)

            if len(paragraphs) < 3:
                continue

            # Select a passage (3-6 paragraphs)
            n_paras = min(rng.randint(3, 6), len(paragraphs))
            start = rng.randint(0, len(paragraphs) - n_paras)
            passage_paras = paragraphs[start:start + n_paras]

            original = '\n\n'.join(passage_paras)

            # Degrade
            if dtype in ('unearned_conclusion', 'register_shift', 'cross_text_insert'):
                degraded = degrade_fn(passage_paras, all_texts)
            else:
                degraded = degrade_fn(passage_paras)

            if original == degraded:
                continue

            # Compare losses
            orig_loss = _compute_passage_loss(
                model, tokenizer, original, config.context_len, device
            )
            deg_loss = _compute_passage_loss(
                model, tokenizer, degraded, config.context_len, device
            )

            correct = orig_loss < deg_loss
            margin = deg_loss - orig_loss

            results_by_type[dtype]['total'] += 1
            if correct:
                results_by_type[dtype]['correct'] += 1
            results_by_type[dtype]['margins'].append(margin)

    # Aggregate
    total_correct = sum(r['correct'] for r in results_by_type.values())
    total_pairs = sum(r['total'] for r in results_by_type.values())
    accuracy = total_correct / total_pairs if total_pairs > 0 else 0

    print(f"\nCoherence Discrimination Results:")
    print(f"  Overall accuracy: {accuracy:.1%} ({total_correct}/{total_pairs})")
    for dtype, r in results_by_type.items():
        acc = r['correct'] / r['total'] if r['total'] > 0 else 0
        avg_margin = sum(r['margins']) / len(r['margins']) if r['margins'] else 0
        print(f"  {dtype:<25s}  {acc:.1%} ({r['correct']}/{r['total']})  "
              f"avg margin: {avg_margin:.4f}")

    return {
        'overall_accuracy': accuracy,
        'total_pairs': total_pairs,
        'per_type': {
            dtype: {
                'accuracy': r['correct'] / r['total'] if r['total'] > 0 else 0,
                'correct': r['correct'],
                'total': r['total'],
                'mean_margin': sum(r['margins']) / len(r['margins']) if r['margins'] else 0,
            }
            for dtype, r in results_by_type.items()
        }
    }
