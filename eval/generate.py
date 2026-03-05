#!/usr/bin/env python3
"""
Generate text from a trained checkpoint.

Usage:
    python -m eval.generate --checkpoint path/to/ckpt.pt --prompt "The sea"
    python -m eval.generate --checkpoint path/to/ckpt.pt --prompt "In the beginning"
    python -m eval.generate --checkpoint path/to/ckpt.pt  # random start
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.config import ModelConfig
from model.model import HeteroMoETransformer
from data.corruption import init_special_tokens
from data import corruption as corruption_mod
from tokenizers import Tokenizer


def generate(model, tokenizer, config, device,
             prompt_ids=None, max_tokens=200,
             temperature=0.8, top_k=50, top_p=0.9):
    """Autoregressive generation in S-mode (decoder-only)."""
    model.eval()

    # Start with S-mode token + optional prompt
    mode_token = corruption_mod.MODE_S_ID
    if prompt_ids is None:
        # Start with just the mode token
        ids = [mode_token]
    else:
        ids = [mode_token] + list(prompt_ids)

    generated = []

    with torch.no_grad():
        for _ in range(max_tokens):
            # Truncate to context window
            input_ids = ids[-(config.context_len):]
            x = torch.tensor([input_ids], dtype=torch.long, device=device)
            enc_avail = torch.tensor([0.0], device=device)  # S-mode: no encoder

            output = model(
                decoder_input_ids=x,
                encoder_available=enc_avail,
            )

            # Get logits for last position
            logits = output['logits'][0, -1, :].float()

            # Temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = torch.topk(logits, top_k)
                logits[logits < topk_vals[-1]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift so first token above threshold is kept
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            # Stop on EOS
            eos_id = tokenizer.token_to_id('<eos>')
            if next_token == eos_id:
                break

            ids.append(next_token)
            generated.append(next_token)

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate text from checkpoint")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--preset', type=str, default='tiny')
    parser.add_argument('--prompt', type=str, default=None,
                        help="Text prompt to start generation")
    parser.add_argument('--max-tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--n', type=int, default=3,
                        help="Number of samples to generate")
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    # Setup
    tok_dir = str(PROJECT_DIR / 'tokenizer')
    config = ModelConfig.from_tokenizer(tok_dir, preset=args.preset)

    tok_path = PROJECT_DIR / 'tokenizer' / 'tokenizer_with_authors.json'
    tokenizer = Tokenizer.from_file(str(tok_path))
    init_special_tokens(tokenizer)

    bos_id = tokenizer.token_to_id('<bos>')
    eos_id = tokenizer.token_to_id('<eos>')

    if args.cpu:
        device = torch.device('cpu')
    else:
        from training.trainer import get_device
        device = get_device()

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = HeteroMoETransformer(config)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Step: {ckpt['global_step']}, Device: {device}")

    # Encode prompt
    prompt_ids = None
    if args.prompt:
        enc = tokenizer.encode(args.prompt)
        prompt_ids = [t for t in enc.ids if t not in (bos_id, eos_id)]
        print(f"  Prompt: \"{args.prompt}\" ({len(prompt_ids)} tokens)")

    # Generate
    print(f"\n{'='*60}")
    for i in range(args.n):
        tokens = generate(
            model, tokenizer, config, device,
            prompt_ids=prompt_ids,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        text = tokenizer.decode(tokens)
        prefix = args.prompt or ""
        print(f"\n--- Sample {i+1} ---")
        print(f"{prefix}{text}")

    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
