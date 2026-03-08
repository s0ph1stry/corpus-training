"""
Gradient covariance probe: test whether different corruption levels produce
qualitatively different learning signals (orthogonal gradients) or just
quantitatively different ones (aligned gradients of different magnitude).

Inspired by the helical engine analogy: if gradients from high-corruption
(X-mode) and low-corruption (R-mode) batches are partially orthogonal,
the model is learning genuinely different things from different corruption
levels. The corruption schedule would then be doing something structurally
interesting — creating a "helical" path through weight space where different
phases push the model in different directions, and the net movement is
something neither phase could achieve alone.

Usage:
    python -m eval.gradient_covariance --checkpoint path/to/checkpoint.pt \
        --data-dir path/to/pretokenized/ --num-batches 50

Outputs:
    - Cosine similarity between mean gradient vectors per mode (R vs S vs X)
    - Gradient magnitude ratio between modes
    - Per-layer breakdown of alignment
    - Projection of each mode's gradient onto principal components
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ModelConfig
from model.model import HeteroMoETransformer
from data.dataset import Phase1Dataset
from data.corruption import init_special_tokens, apply_corruption, UL2Mode
from training.losses import reconstruction_loss


def collect_gradients_by_mode(model, dataset, tokenizer, device,
                               num_batches=50, context_len=512):
    """
    Run forward+backward passes on batches separated by UL2 mode.
    Collect the gradient vector for each batch, grouped by mode.

    Returns dict: {'R': [grad_vectors], 'S': [grad_vectors], 'X': [grad_vectors]}
    """
    from data.corruption import (
        MODE_R_ID, MODE_S_ID, MODE_X_ID,
        init_special_tokens as init_st,
    )
    init_st(tokenizer)

    # Define forced modes
    modes = {
        'R': UL2Mode(
            name='R', token_id=MODE_R_ID, encoder_available=True,
            corruption_rate_range=(0.15, 0.30), avg_span_length=3,
            strategies=['span_mask'],
        ),
        'S': UL2Mode(
            name='S', token_id=MODE_S_ID, encoder_available=False,
            corruption_rate_range=(0.0, 0.0), avg_span_length=0,
            strategies=['sequential'],
        ),
        'X': UL2Mode(
            name='X', token_id=MODE_X_ID, encoder_available=True,
            corruption_rate_range=(0.50, 0.80), avg_span_length=12,
            strategies=['span_mask', 'span_deletion'],
        ),
    }

    gradients_by_mode = defaultdict(list)
    pad_id = tokenizer.token_to_id("<pad>")

    model.eval()  # BatchNorm/Dropout off, but we still compute gradients

    for mode_name, mode in modes.items():
        print(f"  Collecting gradients for mode {mode_name}...")
        for i in range(num_batches):
            # Get a random sample from the dataset
            idx = torch.randint(len(dataset), (1,)).item()
            sample = dataset[idx]

            # Apply corruption with forced mode
            corrupted = apply_corruption(
                sample['token_ids'][:context_len],
                tokenizer=tokenizer,
                forced_mode=mode,
                corruption_rate=sum(mode.corruption_rate_range) / 2,
            )

            # Build batch tensors (batch_size=1)
            dec_ids = torch.tensor([corrupted['decoder_input_ids']], device=device)
            dec_targets = torch.tensor([corrupted['decoder_targets']], device=device)
            enc_available = torch.tensor([corrupted['encoder_available']], device=device)

            enc_ids = None
            enc_mask = None
            if corrupted.get('encoder_input_ids') is not None:
                enc_ids = torch.tensor([corrupted['encoder_input_ids']], device=device)

            dec_mask = (dec_ids == pad_id)

            # Forward
            model.zero_grad()
            output = model(
                decoder_input_ids=dec_ids,
                encoder_input_ids=enc_ids,
                encoder_available=enc_available,
                decoder_padding_mask=dec_mask,
                encoder_padding_mask=enc_mask,
            )

            loss = reconstruction_loss(output['logits'], dec_targets, pad_id=pad_id)
            loss.backward()

            # Collect gradient as a single flat vector
            grad_vec = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_vec.append(param.grad.detach().flatten().cpu())
            grad_vec = torch.cat(grad_vec)
            gradients_by_mode[mode_name].append(grad_vec)

    return gradients_by_mode


def analyze_gradient_covariance(gradients_by_mode):
    """
    Compute cosine similarity, magnitude ratio, and PCA projection
    between gradient distributions for each mode pair.
    """
    results = {}

    # Stack and compute mean gradient per mode
    mean_grads = {}
    for mode, grads in gradients_by_mode.items():
        stacked = torch.stack(grads)
        mean_grads[mode] = stacked.mean(dim=0)
        results[f'{mode}_grad_norm'] = mean_grads[mode].norm().item()
        results[f'{mode}_grad_std'] = stacked.std(dim=0).mean().item()
        results[f'{mode}_num_batches'] = len(grads)

    # Pairwise cosine similarity between mean gradients
    mode_names = list(mean_grads.keys())
    for i, m1 in enumerate(mode_names):
        for m2 in mode_names[i+1:]:
            cos_sim = torch.nn.functional.cosine_similarity(
                mean_grads[m1].unsqueeze(0),
                mean_grads[m2].unsqueeze(0),
            ).item()
            results[f'cosine_{m1}_vs_{m2}'] = cos_sim

            # Magnitude ratio
            norm_ratio = (mean_grads[m1].norm() / mean_grads[m2].norm()).item()
            results[f'norm_ratio_{m1}_vs_{m2}'] = norm_ratio

    # Per-batch cosine similarity distribution (R vs X is the key comparison)
    if 'R' in gradients_by_mode and 'X' in gradients_by_mode:
        r_grads = torch.stack(gradients_by_mode['R'])
        x_grads = torch.stack(gradients_by_mode['X'])
        n = min(len(r_grads), len(x_grads))

        pairwise_cos = []
        for j in range(n):
            cs = torch.nn.functional.cosine_similarity(
                r_grads[j].unsqueeze(0), x_grads[j].unsqueeze(0)
            ).item()
            pairwise_cos.append(cs)

        results['R_vs_X_pairwise_cosine_mean'] = np.mean(pairwise_cos)
        results['R_vs_X_pairwise_cosine_std'] = np.std(pairwise_cos)
        results['R_vs_X_pairwise_cosine_min'] = np.min(pairwise_cos)
        results['R_vs_X_pairwise_cosine_max'] = np.max(pairwise_cos)

    # PCA: project all gradients into 2D to visualize mode clustering
    all_grads = []
    all_labels = []
    for mode, grads in gradients_by_mode.items():
        for g in grads:
            all_grads.append(g)
            all_labels.append(mode)

    all_grads = torch.stack(all_grads)
    # Reduce dimensionality for PCA (use random projection if too large)
    if all_grads.shape[1] > 10000:
        proj = torch.randn(all_grads.shape[1], 1000) / (1000 ** 0.5)
        all_grads_reduced = all_grads @ proj
    else:
        all_grads_reduced = all_grads

    # SVD-based PCA
    centered = all_grads_reduced - all_grads_reduced.mean(dim=0)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    pca_coords = (centered @ Vh[:2].T).numpy()

    results['pca_coords'] = {
        label: pca_coords[i].tolist()
        for i, label in enumerate(all_labels)
    }
    results['pca_explained_variance'] = (S[:2] ** 2 / (S ** 2).sum()).tolist()

    return results


def main():
    parser = argparse.ArgumentParser(description='Gradient covariance probe')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', required=True, help='Path to pretokenized data')
    parser.add_argument('--tokenizer', default=None, help='Path to tokenizer (default: project tokenizer)')
    parser.add_argument('--num-batches', type=int, default=50, help='Batches per mode')
    parser.add_argument('--output', default=None, help='Output JSON path')
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent

    # Load tokenizer
    from tokenizers import Tokenizer
    tok_path = args.tokenizer or str(project_dir / 'tokenizer.json')
    tokenizer = Tokenizer.from_file(tok_path)

    # Load model from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', None)
    if config is None:
        config = ModelConfig()
    elif isinstance(config, dict):
        config = ModelConfig(**config)

    model = HeteroMoETransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load dataset
    dataset = Phase1Dataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        context_len=config.context_len,
    )

    print(f"Loaded model from {args.checkpoint}")
    print(f"Dataset: {len(dataset)} samples")
    print(f"Collecting {args.num_batches} gradient samples per mode...")

    # Collect gradients
    with torch.enable_grad():
        grads = collect_gradients_by_mode(
            model, dataset, tokenizer, device,
            num_batches=args.num_batches,
        )

    # Analyze
    print("\nAnalyzing gradient covariance...")
    results = analyze_gradient_covariance(grads)

    # Print summary
    print("\n" + "=" * 60)
    print("GRADIENT COVARIANCE ANALYSIS")
    print("=" * 60)

    for key in ['R_grad_norm', 'S_grad_norm', 'X_grad_norm']:
        if key in results:
            print(f"  {key}: {results[key]:.6f}")

    print()
    for key in sorted(results.keys()):
        if 'cosine' in key and 'pairwise' not in key:
            print(f"  {key}: {results[key]:.4f}")

    print()
    if 'R_vs_X_pairwise_cosine_mean' in results:
        print(f"  R vs X pairwise cosine: {results['R_vs_X_pairwise_cosine_mean']:.4f} "
              f"± {results['R_vs_X_pairwise_cosine_std']:.4f} "
              f"[{results['R_vs_X_pairwise_cosine_min']:.4f}, "
              f"{results['R_vs_X_pairwise_cosine_max']:.4f}]")

    if 'pca_explained_variance' in results:
        print(f"\n  PCA explained variance (first 2 components): "
              f"{results['pca_explained_variance']}")

    print()
    print("INTERPRETATION:")
    cos_rx = results.get('cosine_R_vs_X', 0)
    if cos_rx > 0.9:
        print("  R and X gradients are highly aligned — corruption level is")
        print("  mainly a difficulty knob, not a qualitative shift.")
    elif cos_rx > 0.5:
        print("  R and X gradients are partially aligned — there is some")
        print("  orthogonal component. The model learns partly different")
        print("  things from different corruption levels.")
    else:
        print("  R and X gradients are substantially orthogonal — the model")
        print("  learns genuinely different things from different corruption")
        print("  levels. The corruption schedule creates a 'helical' path")
        print("  through weight space.")

    # Save
    out_path = args.output or str(project_dir / 'eval' / 'results' / 'gradient_covariance.json')
    # Remove non-serializable items
    save_results = {k: v for k, v in results.items() if k != 'pca_coords'}
    save_results['pca_mode_centroids'] = {}
    if 'pca_coords' in results:
        from collections import defaultdict
        mode_coords = defaultdict(list)
        for label, coord in results['pca_coords'].items():
            mode_coords[label].append(coord)
        for mode, coords in mode_coords.items():
            save_results['pca_mode_centroids'][mode] = np.mean(coords, axis=0).tolist()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
