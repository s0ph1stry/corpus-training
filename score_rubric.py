#!/usr/bin/env python3
"""
Corpus quality rubric scoring.

Four dimensions, scored 1-5:
  1. Structural Integrity — Is the form load-bearing?
  2. Cohesiveness — Does it hold together?
  3. Artistic and Emotional Merit — Does it do something only it can do?
  4. Teachability — Can the model learn from this?

Workflow:
  1. Score 20 representative texts together (calibration round)
  2. Score remaining texts
  3. Output rubric_scores.csv

Scores inform:
  - Curriculum sampling weight (higher quality = more exposure)
  - Anchor text selection (score >= 4 on all dimensions, short)
  - Cross-text corruption source selection (high cohesiveness only)
  - Model-generated output evaluation

Usage:
    python score_rubric.py                    # Interactive scoring
    python score_rubric.py --calibration      # Show calibration examples
    python score_rubric.py --auto             # Auto-score based on heuristics
    python score_rubric.py --summary          # Show existing scores summary
"""

import argparse
import csv
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
CLEANED_DIR = PROJECT_DIR / "cleaned"
SCORES_PATH = PROJECT_DIR / "rubric_scores.csv"

# Calibration texts — representative across genres and quality levels
CALIBRATION_TEXTS = [
    # Fiction
    "Franz Kafka - The Metamorphosis",
    "Leo Tolstoy - Anna Karenina",
    "Fyodor Dostoevsky - Crime and Punishment",
    "F. Scott Fitzgerald - The Great Gatsby",
    # Poetry
    "Emily Dickinson - Complete Poems",
    "William Shakespeare - Sonnets",
    "Homer - Iliad",
    # Philosophy
    "Ludwig Wittgenstein - Tractatus Logico-Philosophicus",
    "Plato - Apology Phaedo Republic Symposium Meno Gorgias Theaetetus",
    "Michel de Montaigne - Essays",
    # Essays
    "Ralph Waldo Emerson - Essays First and Second Series",
    "George Orwell - Collected Essays",
    # Science
    "Charles Darwin - On the Origin of Species",
    "Richard Feynman - The Character of Physical Law",
    # Sacred/Wisdom
    "Bhagavad Gita",
    "Dhammapada",
    # Drama
    "Sophocles - Oedipus Rex Antigone Electra",
    "Samuel Beckett - Waiting for Godot",
    # Procedural
    "Mrs. Beeton - Book of Household Management",
    # Children's
    "Brothers Grimm - Grimm's Fairy Tales",
]

RUBRIC = {
    'structural_integrity': {
        5: "Structure IS the argument. Remove any element and the whole changes.",
        4: "Structure strongly serves content. Most elements necessary.",
        3: "Competent structure. Purposeful but some elements decorative.",
        2: "Structure inherited, not discovered. Follows conventions without necessity.",
        1: "Structure arbitrary or hindering.",
    },
    'cohesiveness': {
        5: "Perfect internal coherence. A foreign sentence immediately detectable.",
        4: "Strong coherence with purposeful variation. Register shifts earned.",
        3: "Generally cohesive with some unevenness.",
        2: "Multiple registers without clear purpose. Assembled rather than composed.",
        1: "Incoherent voice, logic, or tone.",
    },
    'artistic_merit': {
        5: "Irreducible. Creates experiences no other text produces.",
        4: "Strong achievement. Moments that exceed genre expectations.",
        3: "Competent and worthwhile. Historically significant.",
        2: "Value is documentary rather than experiential.",
        1: "Poor quality regardless of significance.",
    },
    'teachability': {
        5: "Coherence principles self-contained and generalizable. No external context needed.",
        4: "Mostly self-contained. Background helps but isn't essential.",
        3: "Coherence partly depends on genre conventions or historical context.",
        2: "Coherence depends heavily on external knowledge.",
        1: "Coherence opaque without extensive context.",
    },
}


def load_existing_scores() -> dict:
    """Load existing rubric scores if any."""
    scores = {}
    if SCORES_PATH.exists():
        with open(SCORES_PATH) as f:
            reader = csv.DictReader(f)
            for row in reader:
                scores[row['name']] = row
    return scores


def save_scores(scores: list):
    """Save scores to CSV."""
    fieldnames = ['name', 'structural_integrity', 'cohesiveness',
                  'artistic_merit', 'teachability', 'mean_score', 'notes']

    with open(SCORES_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in sorted(scores, key=lambda x: x.get('name', '')):
            writer.writerow(row)

    print(f"Saved {len(scores)} scores to {SCORES_PATH}")


def auto_score_heuristics(text: str, name: str) -> dict:
    """
    Rough heuristic scoring for automated pass.
    These are starting points — calibration round adjusts.
    """
    import re

    n_chars = len(text)
    n_words = len(text.split())
    sentences = re.split(r'[.!?]+', text)
    n_sentences = len([s for s in sentences if len(s.strip()) > 0])

    # Vocabulary richness
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    unique_words = len(set(words))
    ttr = unique_words / len(words) if words else 0

    # Sentence length variation (structural complexity signal)
    lengths = [len(s.split()) for s in sentences if len(s.strip()) > 0]
    if lengths:
        avg_len = sum(lengths) / len(lengths)
        len_var = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
    else:
        avg_len = 0
        len_var = 0

    # Category-based priors (from filename patterns)
    name_lower = name.lower()

    # These are very rough — meant to be overridden by manual scoring
    base_structural = 3
    base_cohesive = 3
    base_artistic = 3
    base_teach = 3

    # Known high-quality authors get a bump
    high_quality = ['kafka', 'dostoevsky', 'tolstoy', 'dickinson', 'shakespeare',
                    'homer', 'dante', 'wittgenstein', 'woolf', 'joyce',
                    'chekhov', 'austen', 'melville', 'borges', 'beckett',
                    'sophocles', 'plato', 'montaigne', 'emerson']
    if any(a in name_lower for a in high_quality):
        base_structural = min(5, base_structural + 1)
        base_artistic = min(5, base_artistic + 1)

    # Shorter texts tend to be more tightly constructed
    if n_words < 20000:
        base_structural = min(5, base_structural + 1)
        base_cohesive = min(5, base_cohesive + 1)

    return {
        'name': name,
        'structural_integrity': base_structural,
        'cohesiveness': base_cohesive,
        'artistic_merit': base_artistic,
        'teachability': base_teach,
        'mean_score': (base_structural + base_cohesive + base_artistic + base_teach) / 4,
        'notes': 'auto-scored (heuristic)',
    }


def print_calibration():
    """Print calibration examples with rubric reference."""
    print("CORPUS QUALITY RUBRIC — CALIBRATION")
    print("=" * 60)

    for dim, levels in RUBRIC.items():
        print(f"\n{dim.upper().replace('_', ' ')}:")
        for score, desc in sorted(levels.items(), reverse=True):
            print(f"  {score}: {desc}")

    print(f"\n{'='*60}")
    print("CALIBRATION TEXTS:")
    for i, name in enumerate(CALIBRATION_TEXTS, 1):
        exists = (CLEANED_DIR / f"{name}.txt").exists()
        marker = "  " if exists else "! "
        print(f"  {marker}{i:>2}. {name}")

    if any(not (CLEANED_DIR / f"{name}.txt").exists() for name in CALIBRATION_TEXTS):
        print("\n  ! = not found in cleaned/")


def print_summary():
    """Print summary of existing scores."""
    scores = load_existing_scores()
    if not scores:
        print("No scores found. Run scoring first.")
        return

    print(f"\nRUBRIC SCORES SUMMARY ({len(scores)} texts)")
    print("=" * 80)

    # Sort by mean score
    sorted_scores = sorted(scores.values(),
                           key=lambda x: float(x.get('mean_score', 0)),
                           reverse=True)

    print(f"{'Name':<50s} {'Struct':>6s} {'Cohe':>6s} {'Art':>6s} {'Teach':>6s} {'Mean':>6s}")
    print("-" * 80)
    for s in sorted_scores:
        print(f"{s['name'][:50]:<50s} "
              f"{s.get('structural_integrity', '?'):>6s} "
              f"{s.get('cohesiveness', '?'):>6s} "
              f"{s.get('artistic_merit', '?'):>6s} "
              f"{s.get('teachability', '?'):>6s} "
              f"{s.get('mean_score', '?'):>6s}")

    # Anchor candidates (all dimensions >= 4)
    anchors = [s for s in sorted_scores if all(
        float(s.get(d, 0)) >= 4
        for d in ['structural_integrity', 'cohesiveness', 'artistic_merit', 'teachability']
    )]
    if anchors:
        print(f"\nANCHOR CANDIDATES (all dimensions >= 4): {len(anchors)}")
        for s in anchors:
            print(f"  {s['name']}")


def main():
    parser = argparse.ArgumentParser(description="Corpus quality rubric scoring")
    parser.add_argument('--calibration', action='store_true',
                        help="Show calibration examples and rubric")
    parser.add_argument('--auto', action='store_true',
                        help="Auto-score all texts with heuristics")
    parser.add_argument('--summary', action='store_true',
                        help="Show existing scores summary")
    args = parser.parse_args()

    if args.calibration:
        print_calibration()
        return

    if args.summary:
        print_summary()
        return

    if args.auto:
        files = sorted(CLEANED_DIR.glob("*.txt"))
        existing = load_existing_scores()
        scores = []

        for f in files:
            name = f.stem
            if name in existing and existing[name].get('notes', '') != 'auto-scored (heuristic)':
                # Keep manually scored entries
                scores.append(existing[name])
                continue

            text = f.read_text(encoding='utf-8')
            score = auto_score_heuristics(text, name)
            scores.append(score)

        save_scores(scores)
        print_summary()
        return

    # Interactive mode would go here
    print("Interactive scoring not yet implemented.")
    print("Use --calibration to see rubric, --auto for heuristic scores, --summary for results.")


if __name__ == '__main__':
    main()
