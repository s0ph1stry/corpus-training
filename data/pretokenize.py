#!/usr/bin/env python3
"""
Pre-tokenize all cleaned texts to numpy uint16 arrays.

Reduces memory from ~1.4GB (Python list of ints) to ~76MB (numpy uint16).
Enables memmap loading for near-zero RAM usage.

Usage:
    python -m data.pretokenize
    python -m data.pretokenize --check  # verify existing cache
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

PROJECT_DIR = Path(__file__).parent.parent


def pretokenize(project_dir: Path = PROJECT_DIR):
    """Tokenize all cleaned texts and save as .npy files."""
    tok_path = project_dir / 'tokenizer' / 'tokenizer_with_authors.json'
    if not tok_path.exists():
        tok_path = project_dir / 'tokenizer' / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tok_path))

    # Get special token IDs to strip
    bos_id = tokenizer.token_to_id('<bos>')
    eos_id = tokenizer.token_to_id('<eos>')

    cleaned_dir = project_dir / 'cleaned'
    cache_dir = project_dir / 'data' / 'token_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}  # name -> {n_tokens, file}
    total_tokens = 0

    for f in sorted(cleaned_dir.glob('*.txt')):
        name = f.stem
        text = f.read_text(encoding='utf-8')
        encoding = tokenizer.encode(text)
        token_ids = [t for t in encoding.ids if t not in (bos_id, eos_id)]

        if len(token_ids) < 10:
            print(f"  Skipping {name} ({len(token_ids)} tokens)")
            continue

        # Verify all IDs fit in uint16
        max_id = max(token_ids)
        assert max_id < 65536, f"{name} has token ID {max_id} >= 65536"

        arr = np.array(token_ids, dtype=np.uint16)
        npy_path = cache_dir / f'{name}.npy'
        np.save(str(npy_path), arr)

        manifest[name] = {
            'n_tokens': len(token_ids),
            'file': f'{name}.npy',
        }
        total_tokens += len(token_ids)

    # Save manifest
    manifest_path = cache_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Pre-tokenized {len(manifest)} texts ({total_tokens:,} tokens)")
    print(f"Cache size: {sum(p.stat().st_size for p in cache_dir.glob('*.npy')) / 1e6:.1f} MB")
    print(f"Saved to: {cache_dir}")
    return manifest


def check_cache(project_dir: Path = PROJECT_DIR):
    """Verify token cache matches cleaned texts."""
    cache_dir = project_dir / 'data' / 'token_cache'
    manifest_path = cache_dir / 'manifest.json'

    if not manifest_path.exists():
        print("No token cache found. Run: python -m data.pretokenize")
        return False

    with open(manifest_path) as f:
        manifest = json.load(f)

    cleaned_dir = project_dir / 'cleaned'
    cleaned_names = {f.stem for f in cleaned_dir.glob('*.txt')}
    cached_names = set(manifest.keys())

    missing = cleaned_names - cached_names
    if missing:
        print(f"Missing from cache: {len(missing)} texts")
        for name in sorted(missing)[:5]:
            print(f"  {name}")
        return False

    # Spot-check a few files
    for name in list(manifest.keys())[:3]:
        npy_path = cache_dir / manifest[name]['file']
        arr = np.load(str(npy_path))
        assert arr.dtype == np.uint16
        assert len(arr) == manifest[name]['n_tokens']

    print(f"Cache OK: {len(manifest)} texts, "
          f"{sum(m['n_tokens'] for m in manifest.values()):,} tokens")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()

    if args.check:
        check_cache()
    else:
        pretokenize()
