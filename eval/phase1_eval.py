"""
Phase 1 evaluation: denoising-specific metrics.

1. Reconstruction accuracy at different corruption rates
2. Loss breakdown by corruption strategy
3. Routing behavior by text/author
4. Loss correlation with rubric scores
"""

import csv
import json
import math
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.model import HeteroMoETransformer
from model.config import ModelConfig
from data.corruption import (
    init_special_tokens, apply_corruption,
)
from eval.routing_analysis import RoutingTracker, compute_routing_entropy


def reconstruction_accuracy(model, config, project_dir, device,
                            corruption_rates=(0.15, 0.30, 0.50, 0.80),
                            n_samples=100, seed=42):
    """Measure exact token reconstruction accuracy at different corruption rates.

    For each rate, corrupt held-out text and check how many tokens
    the model reconstructs exactly.
    """
    project_dir = Path(project_dir)
    rng = random.Random(seed)

    tok_path = project_dir / 'tokenizer' / 'tokenizer_with_authors.json'
    if not tok_path.exists():
        tok_path = project_dir / 'tokenizer' / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tok_path))
    init_special_tokens(tokenizer)

    bos_id = tokenizer.token_to_id('<bos>')
    eos_id = tokenizer.token_to_id('<eos>')
    pad_id = tokenizer.token_to_id('<pad>')

    # Load held-out texts
    held_out_path = project_dir / 'eval' / 'held_out_texts.json'
    with open(held_out_path) as f:
        held_out_names = json.load(f)

    texts = []
    for name in held_out_names:
        path = project_dir / 'cleaned' / f'{name}.txt'
        if path.exists():
            text = path.read_text(encoding='utf-8')
            enc = tokenizer.encode(text)
            ids = [t for t in enc.ids if t not in (bos_id, eos_id)]
            if len(ids) > config.context_len:
                texts.append({'name': name, 'ids': ids})

    results = {}
    model.eval()

    for rate in corruption_rates:
        correct = 0
        total = 0
        total_loss = 0.0
        n_batches = 0

        for _ in range(n_samples):
            text = rng.choice(texts)
            start = rng.randint(0, len(text['ids']) - config.context_len)
            window = text['ids'][start:start + config.context_len]

            # Corrupt with span_mask strategy at fixed rate
            result = apply_corruption(
                window, 'span_mask', rate, tokenizer=tokenizer
            )
            corrupted = result['corrupted_ids']

            # Pad
            def pad(ids, length):
                if len(ids) >= length:
                    return ids[:length]
                return ids + [pad_id] * (length - len(ids))

            enc_input = torch.tensor([pad(corrupted, config.context_len)],
                                     dtype=torch.long, device=device)
            dec_target = torch.tensor([pad(window, config.context_len)],
                                      dtype=torch.long, device=device)
            enc_avail = torch.tensor([1.0], device=device)

            with torch.no_grad():
                output = model(
                    decoder_input_ids=dec_target,
                    encoder_input_ids=enc_input,
                    encoder_available=enc_avail,
                )
                logits = output['logits']

                # Token accuracy
                preds = logits.argmax(dim=-1)  # (1, seq_len)
                mask = dec_target != pad_id
                correct += (preds == dec_target).masked_select(mask).sum().item()
                total += mask.sum().item()

                # Loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    dec_target.view(-1),
                    ignore_index=pad_id,
                    reduction='mean',
                )
                total_loss += loss.item()
                n_batches += 1

        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / n_batches if n_batches > 0 else 0

        results[f'rate_{rate}'] = {
            'corruption_rate': rate,
            'token_accuracy': accuracy,
            'mean_loss': avg_loss,
            'n_tokens': total,
        }
        print(f"  corruption={rate:.0%}  accuracy={accuracy:.1%}  loss={avg_loss:.3f}")

    return results


def loss_by_corruption_strategy(model, config, project_dir, device,
                                 n_samples=50, seed=42):
    """Measure loss for each corruption strategy separately."""
    project_dir = Path(project_dir)
    rng = random.Random(seed)

    tok_path = project_dir / 'tokenizer' / 'tokenizer_with_authors.json'
    if not tok_path.exists():
        tok_path = project_dir / 'tokenizer' / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tok_path))
    init_special_tokens(tokenizer)

    bos_id = tokenizer.token_to_id('<bos>')
    eos_id = tokenizer.token_to_id('<eos>')
    pad_id = tokenizer.token_to_id('<pad>')

    held_out_path = project_dir / 'eval' / 'held_out_texts.json'
    with open(held_out_path) as f:
        held_out_names = json.load(f)

    texts = []
    for name in held_out_names:
        path = project_dir / 'cleaned' / f'{name}.txt'
        if path.exists():
            text = path.read_text(encoding='utf-8')
            enc = tokenizer.encode(text)
            ids = [t for t in enc.ids if t not in (bos_id, eos_id)]
            if len(ids) > config.context_len:
                texts.append({'name': name, 'ids': ids})

    strategies = ['span_mask', 'sentence_shuffle', 'span_deletion']
    results = {}
    model.eval()

    for strategy in strategies:
        total_loss = 0.0
        count = 0

        for _ in range(n_samples):
            text = rng.choice(texts)
            start = rng.randint(0, len(text['ids']) - config.context_len)
            window = text['ids'][start:start + config.context_len]

            result = apply_corruption(
                window, strategy, 0.30, tokenizer=tokenizer
            )
            corrupted = result['corrupted_ids']

            def pad(ids, length):
                if len(ids) >= length:
                    return ids[:length]
                return ids + [pad_id] * (length - len(ids))

            enc_input = torch.tensor([pad(corrupted, config.context_len)],
                                     dtype=torch.long, device=device)
            dec_target = torch.tensor([pad(window, config.context_len)],
                                      dtype=torch.long, device=device)
            enc_avail = torch.tensor([1.0], device=device)

            with torch.no_grad():
                output = model(
                    decoder_input_ids=dec_target,
                    encoder_input_ids=enc_input,
                    encoder_available=enc_avail,
                )
                loss = F.cross_entropy(
                    output['logits'].view(-1, output['logits'].shape[-1]),
                    dec_target.view(-1),
                    ignore_index=pad_id,
                    reduction='mean',
                )
                total_loss += loss.item()
                count += 1

        avg_loss = total_loss / count if count > 0 else 0
        results[strategy] = {'mean_loss': avg_loss, 'n_samples': count}
        print(f"  {strategy:<25s}  loss={avg_loss:.3f}")

    return results


def routing_by_text(model, config, project_dir, device,
                    n_windows=10, seed=42):
    """Analyze routing decisions per text — do different texts route differently?"""
    project_dir = Path(project_dir)
    rng = random.Random(seed)

    tok_path = project_dir / 'tokenizer' / 'tokenizer_with_authors.json'
    if not tok_path.exists():
        tok_path = project_dir / 'tokenizer' / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tok_path))
    bos_id = tokenizer.token_to_id('<bos>')
    eos_id = tokenizer.token_to_id('<eos>')
    pad_id = tokenizer.token_to_id('<pad>')

    held_out_path = project_dir / 'eval' / 'held_out_texts.json'
    with open(held_out_path) as f:
        held_out_names = json.load(f)

    # Track per-text routing
    text_routing = {}
    tracker = RoutingTracker(model)
    tracker.register_hooks()
    model.eval()

    for name in held_out_names:
        path = project_dir / 'cleaned' / f'{name}.txt'
        if not path.exists():
            continue
        text = path.read_text(encoding='utf-8')
        enc = tokenizer.encode(text)
        ids = [t for t in enc.ids if t not in (bos_id, eos_id)]
        if len(ids) < config.context_len:
            continue

        tracker.reset()

        with torch.no_grad():
            for _ in range(n_windows):
                start = rng.randint(0, len(ids) - config.context_len)
                window = ids[start:start + config.context_len]
                input_ids = torch.tensor([window], dtype=torch.long, device=device)
                enc_avail = torch.tensor([1.0], device=device)
                model(
                    decoder_input_ids=input_ids,
                    encoder_input_ids=input_ids,
                    encoder_available=enc_avail,
                )

        # Compute per-layer Type A fraction for this text
        type_a_fracs = {}
        for layer_idx in sorted(tracker.stats.keys()):
            counts = tracker.stats[layer_idx]['expert_counts']
            total = sum(counts.values())
            if total > 0:
                # Expert 0 is Type A
                type_a_frac = counts.get(0, 0) / total
                type_a_fracs[f'layer_{layer_idx}'] = type_a_frac

        mean_type_a = sum(type_a_fracs.values()) / len(type_a_fracs) if type_a_fracs else 0
        text_routing[name] = {
            'per_layer': type_a_fracs,
            'mean_type_a_frac': mean_type_a,
        }

    tracker.remove_hooks()

    # Sort by Type A preference
    sorted_texts = sorted(text_routing.items(), key=lambda x: x[1]['mean_type_a_frac'])

    print(f"\nRouting by text (Type A fraction, higher = more reflection):")
    for name, info in sorted_texts:
        bar = '█' * int(info['mean_type_a_frac'] * 40)
        print(f"  {name[:45]:<45s}  {info['mean_type_a_frac']:.1%}  {bar}")

    # Variance across texts — high variance means the router is differentiating
    fracs = [v['mean_type_a_frac'] for v in text_routing.values()]
    if len(fracs) > 1:
        mean_frac = sum(fracs) / len(fracs)
        variance = sum((f - mean_frac) ** 2 for f in fracs) / len(fracs)
        print(f"\n  Mean Type A: {mean_frac:.1%}")
        print(f"  Variance:   {variance:.4f}")
        print(f"  {'Router IS differentiating by text' if variance > 0.005 else 'Router treats all texts similarly'}")

    return {
        'per_text': text_routing,
        'routing_variance': variance if len(fracs) > 1 else 0,
    }


def loss_by_rubric_score(model, config, project_dir, device,
                          n_windows=5, seed=42):
    """Correlate model loss with rubric scores.

    If the model finds high-rubric texts easier, it's learning what we're teaching.
    """
    project_dir = Path(project_dir)
    rng = random.Random(seed)

    tok_path = project_dir / 'tokenizer' / 'tokenizer_with_authors.json'
    if not tok_path.exists():
        tok_path = project_dir / 'tokenizer' / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tok_path))
    bos_id = tokenizer.token_to_id('<bos>')
    eos_id = tokenizer.token_to_id('<eos>')
    pad_id = tokenizer.token_to_id('<pad>')

    # Load rubric scores
    rubric = {}
    rubric_path = project_dir / 'rubric_scores.csv'
    if rubric_path.exists():
        with open(rubric_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['name']
                scores = []
                for dim in ['structural_integrity', 'cohesiveness',
                           'artistic_merit', 'teachability']:
                    if dim in row and row[dim]:
                        scores.append(float(row[dim]))
                if scores:
                    rubric[name] = sum(scores) / len(scores)

    # Use training texts (not held-out) so rubric scores are available
    results = []
    model.eval()

    cleaned_dir = project_dir / 'cleaned'
    for name, mean_score in sorted(rubric.items(), key=lambda x: x[1]):
        path = cleaned_dir / f'{name}.txt'
        if not path.exists():
            continue
        text = path.read_text(encoding='utf-8')
        enc = tokenizer.encode(text)
        ids = [t for t in enc.ids if t not in (bos_id, eos_id)]
        if len(ids) < config.context_len:
            continue

        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for _ in range(n_windows):
                start = rng.randint(0, len(ids) - config.context_len)
                window = ids[start:start + config.context_len]
                input_ids = torch.tensor([window[:-1]], dtype=torch.long, device=device)
                targets = torch.tensor([window[1:]], dtype=torch.long, device=device)

                output = model(decoder_input_ids=input_ids)
                loss = F.cross_entropy(
                    output['logits'].view(-1, output['logits'].shape[-1]),
                    targets.view(-1),
                    ignore_index=pad_id,
                    reduction='mean',
                )
                total_loss += loss.item()
                count += 1

        avg_loss = total_loss / count if count > 0 else 0
        results.append({
            'name': name,
            'rubric_mean': mean_score,
            'loss': avg_loss,
        })

    # Compute correlation
    if len(results) > 5:
        scores = [r['rubric_mean'] for r in results]
        losses = [r['loss'] for r in results]
        n = len(scores)
        mean_s = sum(scores) / n
        mean_l = sum(losses) / n
        cov = sum((s - mean_s) * (l - mean_l) for s, l in zip(scores, losses)) / n
        std_s = (sum((s - mean_s) ** 2 for s in scores) / n) ** 0.5
        std_l = (sum((l - mean_l) ** 2 for l in losses) / n) ** 0.5
        correlation = cov / (std_s * std_l) if std_s > 0 and std_l > 0 else 0

        print(f"\nLoss vs Rubric Score:")
        print(f"  Correlation: {correlation:.3f}")
        if correlation < -0.1:
            print(f"  Higher-scored texts have lower loss — model aligns with rubric")
        elif correlation > 0.1:
            print(f"  Higher-scored texts have HIGHER loss — model finds quality harder")
        else:
            print(f"  No clear relationship between rubric score and loss")

        # Show extremes
        results.sort(key=lambda r: r['loss'])
        print(f"\n  Easiest for model:")
        for r in results[:5]:
            print(f"    {r['name'][:45]:<45s}  loss={r['loss']:.3f}  rubric={r['rubric_mean']:.1f}")
        print(f"  Hardest for model:")
        for r in results[-5:]:
            print(f"    {r['name'][:45]:<45s}  loss={r['loss']:.3f}  rubric={r['rubric_mean']:.1f}")

        return {
            'correlation': correlation,
            'per_text': results,
        }

    return {'per_text': results}


def run_phase1_eval(checkpoint_path, preset='tiny'):
    """Run all Phase 1 evaluations."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from training.trainer import get_device

    device = get_device()
    project_dir = str(Path(__file__).parent.parent)
    tok_dir = str(Path(project_dir) / 'tokenizer')
    config = ModelConfig.from_tokenizer(tok_dir, preset=preset)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = HeteroMoETransformer(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Step: {ckpt['global_step']}")
    print(f"  Device: {device}")

    # Entropy from checkpoint
    if 'entropy_history' in ckpt and ckpt['entropy_history']:
        print(f"\n{'='*60}")
        print(f"  ROUTING ENTROPY HISTORY")
        print(f"{'='*60}")
        for i, e in enumerate(ckpt['entropy_history']):
            status = '✓' if e > 0.4 else '⚠'
            print(f"  checkpoint {i}: {e:.4f} {status}")

    print(f"\n{'='*60}")
    print(f"  1. RECONSTRUCTION ACCURACY")
    print(f"{'='*60}")
    recon = reconstruction_accuracy(model, config, project_dir, device)

    print(f"\n{'='*60}")
    print(f"  2. LOSS BY CORRUPTION STRATEGY")
    print(f"{'='*60}")
    strat = loss_by_corruption_strategy(model, config, project_dir, device)

    print(f"\n{'='*60}")
    print(f"  3. ROUTING BY TEXT")
    print(f"{'='*60}")
    routing = routing_by_text(model, config, project_dir, device)

    print(f"\n{'='*60}")
    print(f"  4. LOSS vs RUBRIC SCORE")
    print(f"{'='*60}")
    rubric = loss_by_rubric_score(model, config, project_dir, device)

    return {
        'reconstruction': recon,
        'corruption_strategies': strat,
        'routing_by_text': routing,
        'rubric_correlation': rubric,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/phase1/latest.pt')
    parser.add_argument('--preset', type=str, default='tiny')
    args = parser.parse_args()
    run_phase1_eval(args.checkpoint, args.preset)
