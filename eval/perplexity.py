"""
Perplexity evaluation on held-out texts.

Per-text and mean perplexity using sliding window with 50% overlap.
Reports alongside difficulty tier for analysis.
"""

import json
import math
from pathlib import Path
from typing import Optional

import torch
from tokenizers import Tokenizer

from model.model import HeteroMoETransformer
from model.config import ModelConfig


def compute_perplexity(model: HeteroMoETransformer,
                       token_ids: list,
                       context_len: int,
                       device: torch.device,
                       stride: Optional[int] = None) -> float:
    """
    Compute perplexity using sliding window with overlap.

    stride: window step size. Default = context_len // 2 (50% overlap).
    """
    if stride is None:
        stride = context_len // 2

    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for start in range(0, len(token_ids) - 1, stride):
            end = min(start + context_len, len(token_ids))
            input_ids = torch.tensor([token_ids[start:end - 1]], dtype=torch.long, device=device)
            target_ids = torch.tensor([token_ids[start + 1:end]], dtype=torch.long, device=device)

            output = model(decoder_input_ids=input_ids)
            logits = output['logits']

            # Cross-entropy loss
            B, S, V = logits.shape
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, V), target_ids.view(-1), reduction='sum'
            )

            # Only count tokens not in overlap region (except first window)
            if start == 0:
                n_tokens = end - start - 1
            else:
                n_tokens = min(stride, end - start - 1)
                # Only count the non-overlapping portion
                loss_per_token = torch.nn.functional.cross_entropy(
                    logits.view(-1, V), target_ids.view(-1), reduction='none'
                )
                # Take loss from the stride portion only
                overlap = (end - start - 1) - stride
                if overlap > 0:
                    loss = loss_per_token[overlap:].sum()
                    n_tokens = len(loss_per_token) - overlap

            total_nll += loss.item()
            total_tokens += n_tokens

            if end >= len(token_ids):
                break

    if total_tokens == 0:
        return float('inf')

    avg_nll = total_nll / total_tokens
    return math.exp(avg_nll)


def evaluate_perplexity(model: HeteroMoETransformer,
                         config: ModelConfig,
                         project_dir: str,
                         device: torch.device) -> dict:
    """Run perplexity evaluation on held-out texts.

    Returns dict with per-text and aggregate results.
    """
    project_dir = Path(project_dir)

    # Load held-out texts
    held_out_path = project_dir / 'eval' / 'held_out_texts.json'
    with open(held_out_path) as f:
        held_out_names = json.load(f)

    # Load tokenizer
    tok_path = project_dir / 'tokenizer' / 'tokenizer_with_authors.json'
    if not tok_path.exists():
        tok_path = project_dir / 'tokenizer' / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tok_path))

    bos_id = tokenizer.token_to_id('<bos>')
    eos_id = tokenizer.token_to_id('<eos>')

    # Load difficulty map for tier info
    difficulty = {}
    csv_path = project_dir / 'difficulty_map.csv'
    if csv_path.exists():
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                difficulty[row['name']] = {
                    'difficulty': float(row['difficulty']),
                    'tier': int(row.get('tier', 1)),
                }

    results = {}
    total_ppl = 0.0
    count = 0

    for name in held_out_names:
        text_path = project_dir / 'cleaned' / f'{name}.txt'
        if not text_path.exists():
            continue

        text = text_path.read_text(encoding='utf-8')
        encoding = tokenizer.encode(text)
        token_ids = [t for t in encoding.ids if t not in (bos_id, eos_id)]

        if len(token_ids) < 10:
            continue

        ppl = compute_perplexity(model, token_ids, config.context_len, device)

        tier = difficulty.get(name, {}).get('tier', 0)
        diff = difficulty.get(name, {}).get('difficulty', 0)

        results[name] = {
            'perplexity': ppl,
            'tier': tier,
            'difficulty': diff,
            'n_tokens': len(token_ids),
        }

        total_ppl += ppl
        count += 1
        print(f"  {name[:50]:<50s}  ppl={ppl:>8.2f}  tier={tier}")

    mean_ppl = total_ppl / count if count > 0 else float('inf')

    # Per-tier averages
    tier_ppls = {}
    for name, r in results.items():
        tier = r['tier']
        tier_ppls.setdefault(tier, []).append(r['perplexity'])

    tier_means = {t: sum(ppls) / len(ppls) for t, ppls in tier_ppls.items()}

    return {
        'per_text': results,
        'mean_perplexity': mean_ppl,
        'per_tier_mean': tier_means,
        'n_texts': count,
    }
