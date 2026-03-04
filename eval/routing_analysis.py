"""
Routing analysis: per-layer expert fractions, encoder_available correlation,
routing entropy. Flag collapse (>80% to one expert).
"""

import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.model import HeteroMoETransformer
from model.config import ModelConfig


class RoutingTracker:
    """Hook-based tracker for routing decisions across layers."""

    def __init__(self, model: HeteroMoETransformer):
        self.model = model
        self.stats = defaultdict(lambda: {
            'expert_counts': defaultdict(int),
            'expert_counts_enc_avail': defaultdict(int),
            'expert_counts_no_enc': defaultdict(int),
            'total_tokens': 0,
            'total_enc_avail': 0,
            'total_no_enc': 0,
        })
        self._hooks = []

    def register_hooks(self):
        """Register forward hooks on all MoE layers to capture routing."""
        for i, layer in enumerate(self.model.decoder_layers):
            hook = layer.router.register_forward_hook(
                self._make_hook(i)
            )
            self._hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            expert_indices, expert_weights, router_logits = output
            hidden_states, enc_avail = input

            enc_mask = enc_avail.squeeze(-1) > 0.5

            for k in range(expert_indices.shape[1]):
                for idx in expert_indices[:, k].tolist():
                    self.stats[layer_idx]['expert_counts'][idx] += 1

                # Separate by encoder availability
                enc_indices = expert_indices[enc_mask, k]
                no_enc_indices = expert_indices[~enc_mask, k]

                for idx in enc_indices.tolist():
                    self.stats[layer_idx]['expert_counts_enc_avail'][idx] += 1
                for idx in no_enc_indices.tolist():
                    self.stats[layer_idx]['expert_counts_no_enc'][idx] += 1

            self.stats[layer_idx]['total_tokens'] += expert_indices.shape[0]
            self.stats[layer_idx]['total_enc_avail'] += enc_mask.sum().item()
            self.stats[layer_idx]['total_no_enc'] += (~enc_mask).sum().item()

        return hook

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def reset(self):
        self.stats = defaultdict(lambda: {
            'expert_counts': defaultdict(int),
            'expert_counts_enc_avail': defaultdict(int),
            'expert_counts_no_enc': defaultdict(int),
            'total_tokens': 0,
            'total_enc_avail': 0,
            'total_no_enc': 0,
        })


def compute_routing_entropy(expert_counts: dict, n_experts: int) -> float:
    """Compute entropy of routing distribution. Max = log(n_experts)."""
    total = sum(expert_counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in expert_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * torch.tensor(p).log().item()

    return entropy


def evaluate_routing(model: HeteroMoETransformer,
                     config: ModelConfig,
                     project_dir: str,
                     device: torch.device,
                     n_batches: int = 50,
                     batch_size: int = 8) -> dict:
    """
    Run routing analysis on held-out texts.

    Returns per-layer routing fractions, entropy, and collapse warnings.
    """
    import math
    project_dir = Path(project_dir)

    # Load tokenizer
    tok_path = project_dir / 'tokenizer' / 'tokenizer_with_authors.json'
    if not tok_path.exists():
        tok_path = project_dir / 'tokenizer' / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tok_path))
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
            texts.append(ids)

    if not texts:
        return {}

    # Set up tracking
    tracker = RoutingTracker(model)
    tracker.register_hooks()
    model.eval()

    import random
    rng = random.Random(42)

    with torch.no_grad():
        for _ in range(n_batches):
            # Build a batch
            batch_ids = []
            enc_available = []

            for _ in range(batch_size):
                text_ids = rng.choice(texts)
                start = rng.randint(0, max(0, len(text_ids) - config.context_len))
                window = text_ids[start:start + config.context_len]
                # Pad
                if len(window) < config.context_len:
                    window = window + [pad_id] * (config.context_len - len(window))
                batch_ids.append(window)
                # Alternate encoder availability
                enc_available.append(float(rng.random() < 0.5))

            input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)
            enc_avail = torch.tensor(enc_available, device=device)

            # Also create dummy encoder input for enc-available samples
            encoder_input_ids = None
            if any(e > 0.5 for e in enc_available):
                encoder_input_ids = input_ids.clone()

            model(
                decoder_input_ids=input_ids,
                encoder_input_ids=encoder_input_ids,
                encoder_available=enc_avail,
            )

    tracker.remove_hooks()

    # Compile results
    max_entropy = math.log(config.n_experts) if config.n_experts > 1 else 1.0
    results = {}
    collapse_warnings = []

    for layer_idx in sorted(tracker.stats.keys()):
        layer_stats = tracker.stats[layer_idx]
        total = layer_stats['total_tokens']
        if total == 0:
            continue

        expert_fracs = {}
        expert_types = model.decoder_layers[layer_idx].expert_types

        for expert_idx in range(config.n_experts):
            count = layer_stats['expert_counts'].get(expert_idx, 0)
            frac = count / total
            expert_fracs[f'expert_{expert_idx}_{expert_types[expert_idx]}'] = frac

            # Collapse check
            if frac > 0.8:
                collapse_warnings.append(
                    f"Layer {layer_idx}: expert {expert_idx} "
                    f"({expert_types[expert_idx]}) receives {frac:.1%} of tokens"
                )

        entropy = compute_routing_entropy(
            layer_stats['expert_counts'], config.n_experts
        )
        normalized_entropy = entropy / max_entropy

        results[f'layer_{layer_idx}'] = {
            'expert_fractions': expert_fracs,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'total_tokens': total,
        }

    print(f"\nRouting Analysis:")
    for layer_name, layer_result in results.items():
        print(f"  {layer_name}:")
        for expert, frac in layer_result['expert_fractions'].items():
            print(f"    {expert}: {frac:.1%}")
        print(f"    entropy: {layer_result['normalized_entropy']:.3f} "
              f"(1.0 = perfectly balanced)")

    if collapse_warnings:
        print(f"\n  COLLAPSE WARNINGS:")
        for w in collapse_warnings:
            print(f"    {w}")

    return {
        'per_layer': results,
        'collapse_warnings': collapse_warnings,
    }
