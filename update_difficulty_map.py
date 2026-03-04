#!/usr/bin/env python3
"""
Update the difficulty map with research-backed difficulty signals.

Adds three new columns to difficulty_map.csv:
  - compression_ratio: bytes per token (high = dense/complex)
  - lexical_diversity_mtld: Measure of Textual Lexical Diversity
  - flesch_reading_ease: standard readability (inverted: lower = harder)
  - combined_research: weighted combination of research-backed signals

The curriculum scheduler should use combined_research as primary difficulty
signal. Existing columns preserved for backward compatibility.

Based on arXiv 2506.11300: compression ratio is the best offline difficulty
signal for curriculum learning at sub-160M scale.

Usage:
    python update_difficulty_map.py          # Update CSV
    python update_difficulty_map.py --dry    # Print without saving
"""

import argparse
import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

PROJECT_DIR = Path(__file__).parent
CLEANED_DIR = PROJECT_DIR / "cleaned"
TOKENIZER_DIR = PROJECT_DIR / "tokenizer"


def load_tokenizer():
    tok_path = TOKENIZER_DIR / "tokenizer.json"
    if not tok_path.exists():
        print("No trained tokenizer found.")
        sys.exit(1)
    return Tokenizer.from_file(str(tok_path))


def compression_ratio(text: str, tokenizer) -> float:
    """Bytes per token. Higher = denser/more complex text."""
    encoding = tokenizer.encode(text)
    n_tokens = len(encoding.ids)
    if n_tokens == 0:
        return 0.0
    n_bytes = len(text.encode('utf-8'))
    return n_bytes / n_tokens


def lexical_diversity_mtld(text: str, threshold: float = 0.72) -> float:
    """
    Measure of Textual Lexical Diversity (MTLD).

    Counts the number of sequential word segments where the type-token ratio
    (TTR) stays above the threshold. Higher MTLD = more lexically diverse.

    McCarthy & Jarvis (2010).
    """
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if len(words) < 10:
        return 0.0

    def _mtld_forward(words):
        factors = 0.0
        types = set()
        token_count = 0

        for word in words:
            types.add(word)
            token_count += 1
            ttr = len(types) / token_count

            if ttr <= threshold:
                factors += 1
                types = set()
                token_count = 0

        # Partial factor for remaining words
        if token_count > 0:
            ttr = len(types) / token_count
            if ttr < 1.0:
                factors += (1 - ttr) / (1 - threshold)

        return len(words) / factors if factors > 0 else len(words)

    # Bidirectional MTLD (average forward and backward)
    forward = _mtld_forward(words)
    backward = _mtld_forward(list(reversed(words)))
    return (forward + backward) / 2


def flesch_reading_ease(text: str) -> float:
    """
    Flesch Reading Ease score.

    Higher = easier. Standard scale: 0-100.
    Academic text: 30-50. Simple English: 60-70.

    Score = 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    words = re.findall(r'\b[a-zA-Z]+\b', text)

    if not sentences or not words:
        return 0.0

    n_sentences = len(sentences)
    n_words = len(words)

    # Syllable estimation (simple heuristic)
    def count_syllables(word):
        word = word.lower()
        if len(word) <= 3:
            return 1
        # Remove trailing silent e
        if word.endswith('e'):
            word = word[:-1]
        # Count vowel groups
        count = len(re.findall(r'[aeiouy]+', word))
        return max(1, count)

    n_syllables = sum(count_syllables(w) for w in words)

    score = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / n_words)
    return score


def normalize_column(values: list) -> list:
    """Normalize to 0-1 range."""
    vmin, vmax = min(values), max(values)
    vrange = vmax - vmin if vmax > vmin else 1
    return [(v - vmin) / vrange for v in values]


def main():
    parser = argparse.ArgumentParser(description="Update difficulty map with research signals")
    parser.add_argument('--dry', action='store_true', help="Print without saving")
    args = parser.parse_args()

    tokenizer = load_tokenizer()

    # Load existing difficulty map
    csv_path = PROJECT_DIR / "difficulty_map.csv"
    existing = {}
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row['name']] = row

    # Process all cleaned texts
    files = sorted(CLEANED_DIR.glob("*.txt"))
    print(f"Processing {len(files)} texts...")

    results = []
    for f in files:
        name = f.stem
        text = f.read_text(encoding='utf-8')

        cr = compression_ratio(text, tokenizer)
        mtld = lexical_diversity_mtld(text)
        fre = flesch_reading_ease(text)

        results.append({
            'name': name,
            'compression_ratio': cr,
            'lexical_diversity_mtld': mtld,
            'flesch_reading_ease': fre,
        })

    # Normalize each new metric to 0-1
    cr_values = [r['compression_ratio'] for r in results]
    mtld_values = [r['lexical_diversity_mtld'] for r in results]
    fre_values = [r['flesch_reading_ease'] for r in results]

    cr_norm = normalize_column(cr_values)
    mtld_norm = normalize_column(mtld_values)
    fre_norm = normalize_column(fre_values)

    for i, r in enumerate(results):
        r['compression_ratio_norm'] = cr_norm[i]
        r['lexical_diversity_norm'] = mtld_norm[i]
        # Invert Flesch: lower score = harder = higher difficulty
        r['flesch_difficulty_norm'] = 1.0 - fre_norm[i]

        # Combined research score (compression ratio weighted highest per literature)
        r['combined_research'] = (
            0.50 * cr_norm[i] +          # compression ratio (best signal)
            0.25 * mtld_norm[i] +         # lexical diversity
            0.25 * r['flesch_difficulty_norm']  # readability (inverted)
        )

    # Merge with existing data
    merged = []
    for r in results:
        name = r['name']
        if name in existing:
            row = dict(existing[name])
            row.update(r)
        else:
            row = r
        merged.append(row)

    # Sort by curriculum order if available, otherwise by combined_research
    merged.sort(key=lambda x: float(x.get('combined_research', 0)))

    if args.dry:
        print(f"\nTop 10 easiest (combined_research):")
        for r in merged[:10]:
            print(f"  {r['name'][:50]:<50s}  "
                  f"cr={r['compression_ratio']:.2f}  "
                  f"mtld={r['lexical_diversity_mtld']:.0f}  "
                  f"fre={r['flesch_reading_ease']:.1f}  "
                  f"combined={r['combined_research']:.3f}")
        print(f"\nTop 10 hardest:")
        for r in merged[-10:]:
            print(f"  {r['name'][:50]:<50s}  "
                  f"cr={r['compression_ratio']:.2f}  "
                  f"mtld={r['lexical_diversity_mtld']:.0f}  "
                  f"fre={r['flesch_reading_ease']:.1f}  "
                  f"combined={r['combined_research']:.3f}")
        return

    # Determine all fieldnames (preserve existing + add new)
    all_fields = set()
    for row in merged:
        all_fields.update(row.keys())

    # Ordered fieldnames
    priority = ['curriculum_order', 'tier', 'name', 'n_tokens', 'difficulty',
                'combined_research', 'compression_ratio', 'lexical_diversity_mtld',
                'flesch_reading_ease', 'compression_ratio_norm',
                'lexical_diversity_norm', 'flesch_difficulty_norm',
                'familiarity_norm', 'complexity_norm']
    fieldnames = [f for f in priority if f in all_fields]
    fieldnames += sorted(f for f in all_fields if f not in fieldnames)

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in merged:
            formatted = {k: (f"{v:.4f}" if isinstance(v, float) else v)
                         for k, v in row.items()}
            writer.writerow(formatted)

    print(f"\nUpdated {csv_path}")
    print(f"  {len(merged)} texts, {len(fieldnames)} columns")
    print(f"  New columns: compression_ratio, lexical_diversity_mtld, "
          f"flesch_reading_ease, combined_research")


if __name__ == '__main__':
    main()
