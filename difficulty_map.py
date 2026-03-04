#!/usr/bin/env python3
"""
Corpus Difficulty Map — two-dimensional scoring for curriculum design.

Scores each text on two axes:
  1. Token familiarity: How much of the text uses common vs rare vocabulary?
     Measured by the proportion of tokens in the top frequency tiers of the
     trained tokenizer's vocabulary across the whole corpus.
  2. Structural complexity: How syntactically complex is the text?
     Measured by average sentence length, sentence length variance,
     and punctuation density (semicolons, colons, em-dashes = clause nesting).

Outputs:
  - CSV with per-text scores
  - Terminal scatter plot (ASCII)
  - Suggested curriculum tiers

Usage:
    python difficulty_map.py              # Generate map
    python difficulty_map.py --csv        # Output CSV only
    python difficulty_map.py --anchors    # Suggest anchor texts
"""

import argparse
import csv
import json
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
    """Load the trained tokenizer."""
    tok_path = TOKENIZER_DIR / "tokenizer.json"
    if not tok_path.exists():
        print("No trained tokenizer found. Run train_tokenizer.py first.")
        sys.exit(1)
    return Tokenizer.from_file(str(tok_path))


def get_corpus_files():
    """Get all cleaned text files."""
    files = sorted(CLEANED_DIR.glob("*.txt"))
    if not files:
        print("No cleaned texts found. Run clean_corpus.py first.")
        sys.exit(1)
    return files


def build_corpus_frequency(tokenizer, files):
    """
    Build a global token frequency distribution across the entire corpus.
    Returns a dict mapping token_id -> frequency rank percentile (0-1).
    0.0 = most common, 1.0 = rarest.
    """
    global_counts = Counter()
    for f in files:
        text = f.read_text(encoding='utf-8')
        encoding = tokenizer.encode(text)
        global_counts.update(encoding.ids)

    # Rank tokens by frequency (most common first)
    sorted_tokens = global_counts.most_common()
    total_unique = len(sorted_tokens)

    # Map each token to its percentile rank
    rank_map = {}
    for rank, (token_id, count) in enumerate(sorted_tokens):
        rank_map[token_id] = rank / total_unique if total_unique > 0 else 0

    return rank_map, global_counts


def score_token_familiarity(tokenizer, text, rank_map):
    """
    Score token familiarity for a text.

    Returns a value between 0 (all common tokens) and 1 (all rare tokens).
    Also returns:
      - fragmentation: average number of tokens per whitespace-separated word
      - unknown_ratio: proportion of tokens the model would need to assemble
        from subword pieces (tokens in the bottom 25% of frequency)
    """
    encoding = tokenizer.encode(text)
    token_ids = encoding.ids

    if not token_ids:
        return 0.0, 1.0, 0.0

    # Average rarity (rank percentile) of tokens in this text
    rarities = [rank_map.get(tid, 1.0) for tid in token_ids]
    avg_rarity = sum(rarities) / len(rarities)

    # Fragmentation: tokens per word
    words = text.split()
    n_tokens_no_special = len([t for t in token_ids
                               if t >= 7])  # skip special token IDs 0-6
    fragmentation = n_tokens_no_special / len(words) if words else 1.0

    # Proportion of rare tokens (bottom 25% of corpus frequency)
    rare_threshold = 0.75
    rare_count = sum(1 for r in rarities if r > rare_threshold)
    rare_ratio = rare_count / len(rarities)

    return avg_rarity, fragmentation, rare_ratio


def score_structural_complexity(text):
    """
    Score structural complexity of a text.

    Components:
      - avg_sentence_length: mean words per sentence
      - sentence_variance: coefficient of variation of sentence lengths
      - clause_density: rate of clause-nesting punctuation (;:—) per sentence
      - max_sentence_length: longest sentence (indicator of embedded complexity)

    Returns a composite score between 0 (simple) and 1 (complex),
    plus component values.
    """
    # Split into sentences (approximate — handles ., !, ?)
    sentences = re.split(r'[.!?]+(?:\s|$)', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

    if not sentences:
        return 0.0, {}

    # Words per sentence
    lengths = [len(s.split()) for s in sentences]
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)

    # Coefficient of variation (normalized variance)
    if avg_len > 0:
        variance = math.sqrt(sum((l - avg_len) ** 2 for l in lengths) / len(lengths))
        cv = variance / avg_len
    else:
        cv = 0

    # Clause-nesting punctuation density
    clause_markers = len(re.findall(r'[;:\u2014\u2013]', text))
    clause_density = clause_markers / len(sentences) if sentences else 0

    # Parenthetical/subordinate clause indicators
    parens_and_dashes = len(re.findall(r'[()—\u2014]', text))
    subordination = parens_and_dashes / len(sentences) if sentences else 0

    components = {
        'avg_sentence_length': avg_len,
        'max_sentence_length': max_len,
        'sentence_cv': cv,
        'clause_density': clause_density,
        'subordination': subordination,
        'n_sentences': len(sentences),
    }

    return components


def normalize_scores(texts_data):
    """
    Normalize familiarity and complexity scores to 0-1 range across the corpus.
    """
    # Collect raw values
    familiarities = [d['raw_familiarity'] for d in texts_data]
    complexities = [d['raw_complexity'] for d in texts_data]

    fam_min, fam_max = min(familiarities), max(familiarities)
    cpx_min, cpx_max = min(complexities), max(complexities)

    fam_range = fam_max - fam_min if fam_max > fam_min else 1
    cpx_range = cpx_max - cpx_min if cpx_max > cpx_min else 1

    for d in texts_data:
        d['familiarity_norm'] = (d['raw_familiarity'] - fam_min) / fam_range
        d['complexity_norm'] = (d['raw_complexity'] - cpx_min) / cpx_range
        # Combined difficulty: diagonal distance from origin
        d['difficulty'] = math.sqrt(d['familiarity_norm'] ** 2 +
                                     d['complexity_norm'] ** 2) / math.sqrt(2)

    return texts_data


def assign_tiers(texts_data, n_tiers=4):
    """
    Assign curriculum tiers based on combined difficulty score.
    Tier 1 = easiest, Tier N = hardest.
    """
    sorted_texts = sorted(texts_data, key=lambda d: d['difficulty'])
    per_tier = len(sorted_texts) / n_tiers

    for i, d in enumerate(sorted_texts):
        d['tier'] = min(int(i / per_tier) + 1, n_tiers)
        d['curriculum_order'] = i

    return texts_data


def identify_anchors(texts_data, n=10):
    """
    Identify anchor text candidates: short, low complexity, high familiarity.
    Good for repeated exposure throughout training.
    """
    candidates = []
    for d in texts_data:
        # Anchor score: short + simple + familiar
        length_score = 1.0 / (1 + d['n_tokens'] / 10000)  # shorter = higher
        simplicity = 1.0 - d['complexity_norm']
        familiarity = 1.0 - d['familiarity_norm']
        anchor_score = length_score * 0.4 + simplicity * 0.3 + familiarity * 0.3
        candidates.append((d['name'], anchor_score, d['n_tokens'],
                           d['tier'], d['difficulty']))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:n]


def ascii_scatter(texts_data, width=72, height=28):
    """
    Print an ASCII scatter plot of the difficulty map.
    X axis: token rarity (unfamiliarity), Y axis: structural complexity.
    """
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    tier_chars = {1: '.', 2: 'o', 3: '*', 4: '#'}

    for d in texts_data:
        x = int(d['familiarity_norm'] * (width - 1))
        y = int((1 - d['complexity_norm']) * (height - 1))  # invert Y
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        grid[y][x] = tier_chars.get(d['tier'], '?')

    # Draw
    print()
    print("  CORPUS DIFFICULTY MAP")
    print("  X: Token rarity (unfamiliarity) →")
    print("  Y: Structural complexity ↑")
    print("  Tiers: . = 1 (easiest)  o = 2  * = 3  # = 4 (hardest)")
    print()
    print("  complex ┤" + ''.join(grid[0]))
    for row in grid[1:-1]:
        print("          │" + ''.join(row))
    print("   simple ┤" + ''.join(grid[-1]))
    print("          └" + "─" * width)
    print("           familiar" + " " * (width - 20) + "rare")
    print()


def main():
    parser = argparse.ArgumentParser(description="Corpus difficulty map")
    parser.add_argument("--csv", action="store_true",
                        help="Output CSV only")
    parser.add_argument("--anchors", action="store_true",
                        help="Show anchor text suggestions")
    parser.add_argument("--tiers", type=int, default=4,
                        help="Number of curriculum tiers (default: 4)")
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    files = get_corpus_files()

    print(f"Analyzing {len(files)} texts...")
    print("Building corpus-wide token frequency distribution...")
    rank_map, global_counts = build_corpus_frequency(tokenizer, files)

    texts_data = []
    for f in files:
        text = f.read_text(encoding='utf-8')
        name = f.stem

        # Token familiarity
        avg_rarity, fragmentation, rare_ratio = score_token_familiarity(
            tokenizer, text, rank_map)

        # Structural complexity
        components = score_structural_complexity(text)

        # Token count
        encoding = tokenizer.encode(text)
        n_tokens = len(encoding.ids)

        # Raw composite scores (pre-normalization)
        # Familiarity axis: weighted combination of rarity and fragmentation
        raw_familiarity = avg_rarity * 0.5 + rare_ratio * 0.3 + (fragmentation - 1) * 0.2

        # Complexity axis: weighted combination of structural features
        raw_complexity = (
            components.get('avg_sentence_length', 0) / 100 * 0.35 +  # normalize ~20-60 words
            components.get('clause_density', 0) / 5 * 0.25 +          # normalize ~0-3
            components.get('sentence_cv', 0) * 0.20 +                  # already 0-2ish
            components.get('subordination', 0) / 3 * 0.20             # normalize ~0-2
        )

        texts_data.append({
            'name': name,
            'n_tokens': n_tokens,
            'raw_familiarity': raw_familiarity,
            'raw_complexity': raw_complexity,
            'avg_rarity': avg_rarity,
            'fragmentation': fragmentation,
            'rare_ratio': rare_ratio,
            **{f'struct_{k}': v for k, v in components.items()},
        })

    # Normalize and assign tiers
    texts_data = normalize_scores(texts_data)
    texts_data = assign_tiers(texts_data, n_tiers=args.tiers)

    if args.csv:
        # CSV output
        csv_path = PROJECT_DIR / "difficulty_map.csv"
        fieldnames = ['curriculum_order', 'tier', 'name', 'n_tokens', 'difficulty',
                       'familiarity_norm', 'complexity_norm',
                       'avg_rarity', 'fragmentation', 'rare_ratio',
                       'struct_avg_sentence_length', 'struct_max_sentence_length',
                       'struct_sentence_cv', 'struct_clause_density',
                       'struct_subordination', 'struct_n_sentences']
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames,
                                     extrasaction='ignore')
            writer.writeheader()
            for d in sorted(texts_data, key=lambda x: x['curriculum_order']):
                # Round floats for readability
                row = {k: (f"{v:.4f}" if isinstance(v, float) else v)
                       for k, v in d.items()}
                writer.writerow(row)
        print(f"\nCSV written to {csv_path}")
        return

    # Terminal output
    ascii_scatter(texts_data)

    # Tier summary
    print("  CURRICULUM TIERS")
    print("  " + "=" * 70)
    for tier in range(1, args.tiers + 1):
        tier_texts = [d for d in texts_data if d['tier'] == tier]
        tier_texts.sort(key=lambda d: d['difficulty'])
        total_tokens = sum(d['n_tokens'] for d in tier_texts)
        avg_fam = sum(d['familiarity_norm'] for d in tier_texts) / len(tier_texts)
        avg_cpx = sum(d['complexity_norm'] for d in tier_texts) / len(tier_texts)
        print(f"\n  Tier {tier}: {len(tier_texts)} texts, "
              f"{total_tokens:,} tokens, "
              f"avg unfamiliarity {avg_fam:.2f}, "
              f"avg complexity {avg_cpx:.2f}")
        for d in tier_texts:
            print(f"    {d['name'][:55]:<55s}  "
                  f"{d['n_tokens']:>8,} tok  "
                  f"fam:{d['familiarity_norm']:.2f}  "
                  f"cpx:{d['complexity_norm']:.2f}")

    # Anchor texts
    if args.anchors or True:  # always show
        print(f"\n  SUGGESTED ANCHOR TEXTS")
        print("  " + "=" * 70)
        print("  Short, familiar, structurally simple — good for repeated exposure")
        print()
        anchors = identify_anchors(texts_data)
        for name, score, n_tok, tier, diff in anchors:
            print(f"    {name[:50]:<50s}  {n_tok:>7,} tok  "
                  f"tier {tier}  diff {diff:.2f}")

    # Summary stats
    print(f"\n  SUMMARY")
    print("  " + "=" * 70)
    total_tokens = sum(d['n_tokens'] for d in texts_data)
    tier_tokens = {}
    for d in texts_data:
        tier_tokens.setdefault(d['tier'], 0)
        tier_tokens[d['tier']] += d['n_tokens']
    print(f"  Total: {len(texts_data)} texts, {total_tokens:,} tokens")
    for tier in sorted(tier_tokens):
        pct = tier_tokens[tier] / total_tokens * 100
        print(f"  Tier {tier}: {tier_tokens[tier]:,} tokens ({pct:.1f}%)")


if __name__ == "__main__":
    main()
