#!/usr/bin/env python3
"""
Run the full evaluation suite and generate summary.

Usage:
    python -m eval.run_eval --checkpoint checkpoints/phase2/latest.pt
    python -m eval.run_eval --checkpoint checkpoints/phase2/latest.pt --control checkpoints/control/latest.pt
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.config import ModelConfig
from model.model import HeteroMoETransformer
from training.trainer import get_device

from eval.perplexity import evaluate_perplexity
from eval.coherence_discrimination import evaluate_coherence_discrimination
from eval.routing_analysis import evaluate_routing
from eval.generation_quality import evaluate_generation_quality
from eval.author_discrimination import evaluate_author_discrimination


def load_model(checkpoint_path: str, config: ModelConfig,
               device: torch.device) -> HeteroMoETransformer:
    """Load a model from checkpoint."""
    model = HeteroMoETransformer(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def run_suite(model: HeteroMoETransformer, config: ModelConfig,
              project_dir: str, device: torch.device,
              label: str = 'corpus') -> dict:
    """Run full evaluation suite on a model."""
    results = {'label': label, 'timestamp': datetime.now().isoformat()}

    print(f"\n{'='*60}")
    print(f"  Evaluating: {label}")
    print(f"{'='*60}")

    # 1. Perplexity
    print("\n[1/5] Perplexity")
    results['perplexity'] = evaluate_perplexity(model, config, project_dir, device)

    # 2. Coherence discrimination (PRIMARY)
    print("\n[2/5] Coherence Discrimination")
    results['coherence'] = evaluate_coherence_discrimination(
        model, config, project_dir, device
    )

    # 3. Routing analysis
    print("\n[3/5] Routing Analysis")
    results['routing'] = evaluate_routing(model, config, project_dir, device)

    # 4. Generation quality
    print("\n[4/5] Generation Quality")
    results['generation'] = evaluate_generation_quality(
        model, config, project_dir, device
    )

    # 5. Author discrimination
    print("\n[5/5] Author Discrimination")
    results['author'] = evaluate_author_discrimination(
        model, config, project_dir, device
    )

    return results


def generate_summary(corpus_results: dict, control_results: dict = None,
                     output_dir: str = None) -> str:
    """Generate markdown summary comparing corpus vs control."""
    lines = [
        f"# Evaluation Summary",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # Perplexity
    lines.append("## Perplexity")
    c_ppl = corpus_results['perplexity']['mean_perplexity']
    lines.append(f"- Corpus model: **{c_ppl:.2f}**")
    if control_results:
        ctrl_ppl = control_results['perplexity']['mean_perplexity']
        lines.append(f"- Control model: **{ctrl_ppl:.2f}**")
    lines.append("")

    # Coherence discrimination (PRIMARY)
    lines.append("## Coherence Discrimination (PRIMARY METRIC)")
    c_acc = corpus_results['coherence']['overall_accuracy']
    lines.append(f"- Corpus model: **{c_acc:.1%}**")
    if control_results:
        ctrl_acc = control_results['coherence']['overall_accuracy']
        lines.append(f"- Control model: **{ctrl_acc:.1%}**")
        diff = c_acc - ctrl_acc
        lines.append(f"- Delta: **{diff:+.1%}** {'(corpus wins)' if diff > 0 else '(control wins)'}")
    lines.append("")

    # Per-type breakdown
    lines.append("### Per-type breakdown")
    lines.append("| Type | Corpus | Control | Delta |")
    lines.append("|------|--------|---------|-------|")
    for dtype in corpus_results['coherence']['per_type']:
        c_val = corpus_results['coherence']['per_type'][dtype]['accuracy']
        if control_results and dtype in control_results['coherence']['per_type']:
            ctrl_val = control_results['coherence']['per_type'][dtype]['accuracy']
            delta = c_val - ctrl_val
            lines.append(f"| {dtype} | {c_val:.1%} | {ctrl_val:.1%} | {delta:+.1%} |")
        else:
            lines.append(f"| {dtype} | {c_val:.1%} | — | — |")
    lines.append("")

    # Routing
    lines.append("## Routing Analysis")
    warnings = corpus_results['routing'].get('collapse_warnings', [])
    if warnings:
        lines.append("**WARNINGS:**")
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("No collapse detected.")
    lines.append("")

    # Generation
    lines.append("## Generation Quality")
    gen = corpus_results['generation']
    lines.append(f"- Mean self-perplexity: {gen.get('mean_self_perplexity', 0):.1f}")
    lines.append(f"- Mean compression ratio: {gen.get('mean_compression_ratio', 0):.2f}")
    lines.append(f"- Mean generation length: {gen.get('mean_gen_length', 0):.0f} tokens")
    lines.append("")

    # Author discrimination
    lines.append("## Author Discrimination")
    auth = corpus_results['author']
    if auth:
        lines.append(f"- Top-1 accuracy: {auth.get('top1_accuracy', 0):.1%}")
        lines.append(f"- Top-3 accuracy: {auth.get('top3_accuracy', 0):.1%}")
        lines.append(f"- Mean true rank: {auth.get('mean_true_rank', 0):.1f} / {auth.get('n_authors', 0)}")
    lines.append("")

    summary = '\n'.join(lines)

    if output_dir:
        out_path = Path(output_dir) / 'summary.md'
        out_path.write_text(summary)
        print(f"\nSummary written to {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run evaluation suite")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--control', type=str, default=None,
                        help="Control model checkpoint for comparison")
    parser.add_argument('--preset', choices=['tiny', 'small'], default='tiny')
    args = parser.parse_args()

    device = get_device()
    tok_dir = str(PROJECT_DIR / 'tokenizer')
    config = ModelConfig.from_tokenizer(tok_dir, preset=args.preset)

    # Load corpus model
    print(f"Loading corpus model from {args.checkpoint}")
    corpus_model = load_model(args.checkpoint, config, device)

    # Run corpus eval
    corpus_results = run_suite(
        corpus_model, config, str(PROJECT_DIR), device, label='corpus'
    )

    # Save results
    results_dir = PROJECT_DIR / 'eval' / 'results'
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    with open(results_dir / f'corpus_{timestamp}.json', 'w') as f:
        # Convert non-serializable items
        json.dump(corpus_results, f, indent=2, default=str)

    # Control model (if provided)
    control_results = None
    if args.control:
        print(f"\nLoading control model from {args.control}")
        control_model = load_model(args.control, config, device)
        control_results = run_suite(
            control_model, config, str(PROJECT_DIR), device, label='control'
        )
        with open(results_dir / f'control_{timestamp}.json', 'w') as f:
            json.dump(control_results, f, indent=2, default=str)

    # Generate summary
    summary = generate_summary(corpus_results, control_results, str(results_dir))
    print(f"\n{'='*60}")
    print(summary)


if __name__ == '__main__':
    main()
