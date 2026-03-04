"""
Generation quality evaluation.

Automated metrics:
  - Self-perplexity: model's perplexity on its own output
  - Compression ratio: bytes per token in generated text
  - Corruption detectability: can the model detect its own output as non-corpus?

Manual evaluation (for key runs):
  20 passages rated on the corpus quality rubric.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.model import HeteroMoETransformer
from model.config import ModelConfig


def generate(model: HeteroMoETransformer,
             tokenizer: Tokenizer,
             prompt_ids: list,
             max_new_tokens: int = 256,
             temperature: float = 0.8,
             top_k: int = 50,
             top_p: float = 0.9,
             device: torch.device = None,
             complete_token_id: int = None) -> list:
    """
    Autoregressive generation with top-k/top-p sampling.

    Stops at <eos>, <complete>, or max_new_tokens.
    """
    model.eval()
    eos_id = tokenizer.token_to_id('<eos>')
    context_len = model.config.context_len

    input_ids = list(prompt_ids)
    generated = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Truncate to context window
            window = input_ids[-(context_len - 1):]
            x = torch.tensor([window], dtype=torch.long, device=device)

            output = model(decoder_input_ids=x)
            logits = output['logits'][:, -1, :]  # last token logits

            # Temperature
            if temperature > 0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Stop conditions
            if next_token == eos_id:
                break
            if complete_token_id is not None and next_token == complete_token_id:
                break

            generated.append(next_token)
            input_ids.append(next_token)

    return generated


def evaluate_generation_quality(model: HeteroMoETransformer,
                                 config: ModelConfig,
                                 project_dir: str,
                                 device: torch.device,
                                 n_samples: int = 20,
                                 seed: int = 42) -> dict:
    """
    Evaluate quality of generated text.

    Generates passages from held-out text prompts, then measures:
    1. Self-perplexity (model's own assessment of its output)
    2. Compression ratio (information density)
    3. Average generation length before stopping
    """
    import random
    project_dir = Path(project_dir)
    rng = random.Random(seed)

    # Load tokenizer
    tok_path = project_dir / 'tokenizer' / 'tokenizer_with_authors.json'
    if not tok_path.exists():
        tok_path = project_dir / 'tokenizer' / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tok_path))
    bos_id = tokenizer.token_to_id('<bos>')
    eos_id = tokenizer.token_to_id('<eos>')

    # Complete token
    config_path = project_dir / 'tokenizer' / 'config_with_authors.json'
    complete_id = None
    if config_path.exists():
        with open(config_path) as f:
            tok_config = json.load(f)
        complete_id = tok_config.get('complete_token_id')

    # Load held-out texts for prompts
    held_out_path = project_dir / 'eval' / 'held_out_texts.json'
    with open(held_out_path) as f:
        held_out_names = json.load(f)

    texts = {}
    for name in held_out_names:
        path = project_dir / 'cleaned' / f'{name}.txt'
        if path.exists():
            text = path.read_text(encoding='utf-8')
            enc = tokenizer.encode(text)
            ids = [t for t in enc.ids if t not in (bos_id, eos_id)]
            if len(ids) > 50:
                texts[name] = ids

    if not texts:
        return {}

    results = []
    total_self_ppl = 0.0
    total_compression = 0.0
    total_length = 0

    names = list(texts.keys())
    for i in range(min(n_samples, len(names))):
        name = names[i % len(names)]
        text_ids = texts[name]

        # Use first 32-64 tokens as prompt
        prompt_len = rng.randint(32, min(64, len(text_ids) // 2))
        prompt = text_ids[:prompt_len]

        # Generate
        generated = generate(
            model, tokenizer, prompt,
            max_new_tokens=256,
            device=device,
            complete_token_id=complete_id,
        )

        if len(generated) < 5:
            continue

        # Decode for display and compression ratio
        gen_text = tokenizer.decode(generated)
        gen_bytes = len(gen_text.encode('utf-8'))
        compression = gen_bytes / len(generated) if generated else 0

        # Self-perplexity: model's loss on its own generated text
        all_ids = prompt + generated
        input_ids = torch.tensor([all_ids[:-1]], dtype=torch.long, device=device)
        target_ids = torch.tensor([all_ids[1:]], dtype=torch.long, device=device)

        model.eval()
        with torch.no_grad():
            output = model(decoder_input_ids=input_ids[:, :config.context_len])
            logits = output['logits']
            target = target_ids[:, :logits.shape[1]]
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                target.view(-1),
                reduction='mean',
            )

        self_ppl = math.exp(loss.item())

        results.append({
            'source': name,
            'prompt_len': prompt_len,
            'gen_len': len(generated),
            'self_perplexity': self_ppl,
            'compression_ratio': compression,
            'sample': gen_text[:200],
        })

        total_self_ppl += self_ppl
        total_compression += compression
        total_length += len(generated)

        print(f"  {name[:40]:<40s}  len={len(generated):>4d}  "
              f"self_ppl={self_ppl:>7.1f}  comp={compression:.2f}")

    n = len(results)
    return {
        'samples': results,
        'mean_self_perplexity': total_self_ppl / n if n else 0,
        'mean_compression_ratio': total_compression / n if n else 0,
        'mean_gen_length': total_length / n if n else 0,
        'n_samples': n,
    }
