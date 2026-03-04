"""
Author discrimination evaluation.

Given a passage, rank author tokens by P(passage | author).
Accuracy = true author ranked first.

Tests whether the model has learned to associate structural/stylistic
patterns with specific authors beyond surface vocabulary.
"""

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.model import HeteroMoETransformer
from model.config import ModelConfig


def evaluate_author_discrimination(model: HeteroMoETransformer,
                                    config: ModelConfig,
                                    project_dir: str,
                                    device: torch.device,
                                    n_samples: int = 50,
                                    passage_len: int = 128,
                                    seed: int = 42) -> dict:
    """
    For each passage: prepend each author token, compute loss, rank.

    The model should assign lowest loss when the correct author token
    is prepended.
    """
    project_dir = Path(project_dir)
    rng = random.Random(seed)

    # Load tokenizer
    tok_path = project_dir / 'tokenizer' / 'tokenizer_with_authors.json'
    if not tok_path.exists():
        print("Author tokenizer not found. Run add_author_tokens.py first.")
        return {}
    tokenizer = Tokenizer.from_file(str(tok_path))
    bos_id = tokenizer.token_to_id('<bos>')
    eos_id = tokenizer.token_to_id('<eos>')

    # Load author map
    author_map_path = project_dir / 'tokenizer' / 'author_map.json'
    if not author_map_path.exists():
        return {}
    with open(author_map_path) as f:
        author_map = json.load(f)

    # Get unique author token IDs
    author_tokens = {}  # token_id -> token_string
    for info in author_map.values():
        tid = info.get('token_id')
        token = info.get('token')
        if tid is not None and token:
            author_tokens[tid] = token

    if len(author_tokens) < 2:
        print("Not enough author tokens for discrimination.")
        return {}

    author_ids = sorted(author_tokens.keys())

    # Load held-out texts
    held_out_path = project_dir / 'eval' / 'held_out_texts.json'
    with open(held_out_path) as f:
        held_out_names = json.load(f)

    # Prepare samples: (text_name, token_ids, true_author_token_id)
    samples = []
    for name in held_out_names:
        if name not in author_map:
            continue
        true_author_id = author_map[name].get('token_id')
        if true_author_id is None:
            continue

        path = project_dir / 'cleaned' / f'{name}.txt'
        if not path.exists():
            continue

        text = path.read_text(encoding='utf-8')
        enc = tokenizer.encode(text)
        ids = [t for t in enc.ids if t not in (bos_id, eos_id)]

        if len(ids) < passage_len + 10:
            continue

        samples.append((name, ids, true_author_id))

    if not samples:
        return {}

    rng.shuffle(samples)
    samples = samples[:n_samples]

    model.eval()
    correct = 0
    total = 0
    top3_correct = 0
    results = []

    with torch.no_grad():
        for name, text_ids, true_author_id in samples:
            # Extract a random passage
            start = rng.randint(0, len(text_ids) - passage_len)
            passage = text_ids[start:start + passage_len]

            # For each author token, compute loss on the passage
            losses = {}
            for aid in author_ids:
                # Prepend author token to passage
                input_seq = [aid] + passage[:-1]
                target_seq = passage

                input_t = torch.tensor([input_seq], dtype=torch.long, device=device)
                target_t = torch.tensor([target_seq], dtype=torch.long, device=device)

                output = model(decoder_input_ids=input_t)
                logits = output['logits']

                loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    target_t.view(-1),
                    reduction='mean',
                )
                losses[aid] = loss.item()

            # Rank by loss (lower = better)
            ranked = sorted(losses.items(), key=lambda x: x[1])
            predicted_author = ranked[0][0]
            top3_ids = [r[0] for r in ranked[:3]]

            is_correct = predicted_author == true_author_id
            is_top3 = true_author_id in top3_ids

            if is_correct:
                correct += 1
            if is_top3:
                top3_correct += 1
            total += 1

            true_rank = next(
                i for i, (aid, _) in enumerate(ranked) if aid == true_author_id
            ) + 1

            results.append({
                'text': name,
                'true_author': author_tokens.get(true_author_id, '?'),
                'predicted_author': author_tokens.get(predicted_author, '?'),
                'correct': is_correct,
                'true_rank': true_rank,
            })

    accuracy = correct / total if total > 0 else 0
    top3_accuracy = top3_correct / total if total > 0 else 0
    mean_rank = sum(r['true_rank'] for r in results) / len(results) if results else 0

    print(f"\nAuthor Discrimination:")
    print(f"  Top-1 accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"  Top-3 accuracy: {top3_accuracy:.1%} ({top3_correct}/{total})")
    print(f"  Mean true rank: {mean_rank:.1f} / {len(author_ids)}")

    return {
        'top1_accuracy': accuracy,
        'top3_accuracy': top3_accuracy,
        'mean_true_rank': mean_rank,
        'n_authors': len(author_ids),
        'n_samples': total,
        'per_sample': results,
    }
