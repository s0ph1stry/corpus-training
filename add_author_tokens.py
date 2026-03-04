#!/usr/bin/env python3
"""
Extend the trained tokenizer with author-specific special tokens
and the structural completion token <complete>.

Reads author names from sources.csv, creates tokens like <author_Tolstoy>,
and adds them to the tokenizer vocabulary using add_special_tokens().
This expands vocab without retraining the BPE model.

Must run BEFORE any model initialization — vocab_size depends on this.

Outputs:
  - tokenizer/tokenizer_with_authors.json — extended tokenizer
  - tokenizer/author_map.json — filename → author_token mapping
  - tokenizer/config_with_authors.json — updated config with new vocab_size
"""

import csv
import json
import re
from pathlib import Path

from tokenizers import Tokenizer

PROJECT_DIR = Path(__file__).parent
TOKENIZER_DIR = PROJECT_DIR / "tokenizer"
SOURCES_CSV = PROJECT_DIR / "sources.csv"
CLEANED_DIR = PROJECT_DIR / "cleaned"


def normalize_author_name(author: str) -> str:
    """Convert author name to token-safe format.

    'Leo Tolstoy' -> 'Tolstoy'
    'Fyodor Dostoevsky' -> 'Dostoevsky'
    'Brothers Grimm' -> 'Grimm'
    '—' or '' -> 'anonymous'
    """
    if not author or author.strip() in ('—', '-', '', 'unknown', 'Unknown'):
        return 'anonymous'

    # Use last name (last space-separated component)
    parts = author.strip().split()
    last = parts[-1]

    # Clean: remove any non-alphanumeric chars except hyphens
    last = re.sub(r'[^a-zA-Z\-]', '', last)

    if not last:
        return 'anonymous'

    return last


def build_author_tokens():
    """Read sources.csv and build author token list + filename mapping."""
    authors = set()
    filename_to_author = {}

    with open(SOURCES_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            author_raw = row.get('Author', '').strip()
            filename = row.get('Filename', '').strip()
            cleaned = row.get('Cleaned', '').strip()

            author_name = normalize_author_name(author_raw)
            authors.add(author_name)

            if filename:
                # Map the cleaned filename (without extension) to author token
                stem = Path(filename).stem
                filename_to_author[stem] = f"<author_{author_name}>"

    # Also check cleaned/ directory for files not in sources.csv
    if CLEANED_DIR.exists():
        for f in sorted(CLEANED_DIR.glob("*.txt")):
            stem = f.stem
            if stem not in filename_to_author:
                # Try to infer author from filename pattern "Author - Title"
                if ' - ' in stem:
                    author_part = stem.split(' - ')[0].strip()
                    author_name = normalize_author_name(author_part)
                else:
                    author_name = 'anonymous'
                filename_to_author[stem] = f"<author_{author_name}>"

    # Build sorted token list
    author_tokens = sorted(f"<author_{name}>" for name in authors)

    return author_tokens, filename_to_author


def main():
    # Load base tokenizer
    base_path = TOKENIZER_DIR / "tokenizer.json"
    if not base_path.exists():
        print("No trained tokenizer found. Run train_tokenizer.py first.")
        return

    tokenizer = Tokenizer.from_file(str(base_path))
    base_vocab_size = tokenizer.get_vocab_size()
    print(f"Base tokenizer vocab size: {base_vocab_size}")

    # Build author tokens
    author_tokens, filename_to_author = build_author_tokens()
    print(f"Found {len(author_tokens)} unique author tokens")

    # The structural completion token
    complete_token = "<complete>"

    # Combine all new special tokens
    new_tokens = author_tokens + [complete_token]

    # Filter out any that already exist
    existing_vocab = tokenizer.get_vocab()
    new_tokens = [t for t in new_tokens if t not in existing_vocab]
    print(f"Adding {len(new_tokens)} new special tokens")

    # Add to tokenizer
    tokenizer.add_special_tokens(new_tokens)

    new_vocab_size = tokenizer.get_vocab_size()
    print(f"New vocab size: {new_vocab_size} (+{new_vocab_size - base_vocab_size})")

    # Save extended tokenizer
    out_path = TOKENIZER_DIR / "tokenizer_with_authors.json"
    tokenizer.save(str(out_path))
    print(f"Saved: {out_path}")

    # Save author map
    author_map_path = TOKENIZER_DIR / "author_map.json"
    # Also include token IDs for easy lookup
    author_map_with_ids = {}
    for stem, token in filename_to_author.items():
        tid = tokenizer.token_to_id(token)
        author_map_with_ids[stem] = {
            'token': token,
            'token_id': tid
        }

    with open(author_map_path, 'w', encoding='utf-8') as f:
        json.dump(author_map_with_ids, f, indent=2, ensure_ascii=False)
    print(f"Saved: {author_map_path}")

    # Save updated config
    base_config_path = TOKENIZER_DIR / "config.json"
    with open(base_config_path) as f:
        config = json.load(f)

    config['vocab_size'] = new_vocab_size
    config['author_tokens'] = author_tokens
    config['complete_token'] = complete_token
    config['complete_token_id'] = tokenizer.token_to_id(complete_token)
    config['n_author_tokens'] = len(author_tokens)

    new_config_path = TOKENIZER_DIR / "config_with_authors.json"
    with open(new_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Saved: {new_config_path}")

    # Print summary
    print(f"\nAuthor token examples:")
    for token in author_tokens[:10]:
        tid = tokenizer.token_to_id(token)
        print(f"  {token}: id={tid}")
    if len(author_tokens) > 10:
        print(f"  ... and {len(author_tokens) - 10} more")

    complete_id = tokenizer.token_to_id(complete_token)
    print(f"\nCompletion token: {complete_token}: id={complete_id}")

    # Verify round-trip
    test_text = "The quick brown fox"
    enc = tokenizer.encode(test_text)
    dec = tokenizer.decode(enc.ids)
    print(f"\nRound-trip test: '{test_text}' -> {len(enc.ids)} tokens -> '{dec}'")


if __name__ == '__main__':
    main()
