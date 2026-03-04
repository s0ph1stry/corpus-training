"""
Corruption strategies for the denoising pretraining objective (Phase 1).

Pure functions, independently testable. Each takes token IDs and returns
corrupted token IDs + targets. No model dependencies.

Strategies:
  - span_mask: T5-style span masking, replace spans with <mask>
  - sentence_shuffle: detect sentence boundaries, shuffle order
  - span_deletion: like span_mask but mask token omitted (harder)
  - cross_text_insert: insert spans from a different text, wrapped in <corrupt>
  - semantic_corrupt: antonym/synonym substitution via lookup table

Progressive corruption schedule: linear from 0.15 to 0.80 over training.
"""

import random
import re
from typing import List, Tuple, Optional, Dict


# Token IDs for special tokens (set by init_special_tokens)
MASK_ID = None
CORRUPT_START_ID = None
CORRUPT_END_ID = None
BOS_ID = None
EOS_ID = None
PAD_ID = None


def init_special_tokens(tokenizer):
    """Initialize special token IDs from a loaded tokenizer."""
    global MASK_ID, CORRUPT_START_ID, CORRUPT_END_ID, BOS_ID, EOS_ID, PAD_ID
    MASK_ID = tokenizer.token_to_id("<mask>")
    CORRUPT_START_ID = tokenizer.token_to_id("<corrupt>")
    # <corrupt> is used for both start and end markers
    CORRUPT_END_ID = CORRUPT_START_ID
    BOS_ID = tokenizer.token_to_id("<bos>")
    EOS_ID = tokenizer.token_to_id("<eos>")
    PAD_ID = tokenizer.token_to_id("<pad>")


def corruption_rate(step: int, total_steps: int,
                    min_rate: float = 0.15, max_rate: float = 0.80) -> float:
    """Linear corruption schedule from min_rate to max_rate over training."""
    progress = min(step / max(total_steps, 1), 1.0)
    return min_rate + (max_rate - min_rate) * progress


def span_mask(token_ids: List[int], rate: float = 0.15,
              avg_span_length: int = 3) -> Tuple[List[int], List[int]]:
    """
    T5-style span masking. Replace random spans with <mask> tokens.

    Returns (corrupted_ids, target_ids) where target_ids contains the
    original tokens at masked positions and PAD elsewhere.
    """
    if not token_ids or rate <= 0:
        return list(token_ids), [PAD_ID] * len(token_ids)

    n = len(token_ids)
    n_to_mask = max(1, int(n * rate))

    # Generate span starts using geometric distribution for span lengths
    masked = [False] * n
    total_masked = 0
    attempts = 0

    while total_masked < n_to_mask and attempts < n * 3:
        start = random.randint(0, n - 1)
        # Geometric distribution for span length, mean = avg_span_length
        span_len = min(
            random.choices(
                range(1, avg_span_length * 3 + 1),
                weights=[((1 - 1/avg_span_length) ** (i-1)) * (1/avg_span_length)
                         for i in range(1, avg_span_length * 3 + 1)],
                k=1
            )[0],
            n - start
        )

        for i in range(start, min(start + span_len, n)):
            if not masked[i]:
                masked[i] = True
                total_masked += 1
        attempts += 1

    # Build corrupted sequence: replace masked spans with single <mask>
    corrupted = []
    targets = []
    in_span = False

    for i, tid in enumerate(token_ids):
        if masked[i]:
            if not in_span:
                corrupted.append(MASK_ID)
                targets.append(PAD_ID)  # alignment padding for the mask token
                in_span = True
            # Target gets the original token
            # (stored separately, not aligned 1:1 with corrupted)
        else:
            corrupted.append(tid)
            targets.append(PAD_ID)
            in_span = False

    # Build target sequence: just the masked tokens in order
    masked_tokens = [tid for i, tid in enumerate(token_ids) if masked[i]]

    return corrupted, masked_tokens


def sentence_shuffle(token_ids: List[int], tokenizer,
                     rate: float = 1.0) -> Tuple[List[int], List[int]]:
    """
    Detect sentence boundaries, shuffle sentence order.

    Decodes to text, splits on sentence boundaries, shuffles, re-encodes.
    Rate controls the fraction of sentences that participate in the shuffle.
    Returns (shuffled_ids, original_ids) for reconstruction target.
    """
    text = tokenizer.decode(token_ids)

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s for s in sentences if s.strip()]

    if len(sentences) < 2:
        return list(token_ids), list(token_ids)

    # Select sentences to shuffle
    n_shuffle = max(2, int(len(sentences) * rate))
    indices = list(range(len(sentences)))
    shuffle_indices = random.sample(indices, min(n_shuffle, len(indices)))

    # Shuffle selected sentences
    shuffle_values = [sentences[i] for i in shuffle_indices]
    random.shuffle(shuffle_values)
    shuffled_sentences = list(sentences)
    for i, idx in enumerate(shuffle_indices):
        shuffled_sentences[idx] = shuffle_values[i]

    shuffled_text = ' '.join(shuffled_sentences)
    shuffled_encoding = tokenizer.encode(shuffled_text)
    # Strip BOS/EOS added by post-processor
    shuffled_ids = [t for t in shuffled_encoding.ids
                    if t not in (BOS_ID, EOS_ID)]

    return shuffled_ids, list(token_ids)


def span_deletion(token_ids: List[int], rate: float = 0.15,
                  avg_span_length: int = 3) -> Tuple[List[int], List[int]]:
    """
    Like span_mask but the mask token is omitted entirely.
    The model must figure out that something is missing and what it was.
    Harder than span_mask — used later in training.

    Returns (corrupted_ids, deleted_tokens).
    """
    if not token_ids or rate <= 0:
        return list(token_ids), []

    n = len(token_ids)
    n_to_delete = max(1, int(n * rate))

    # Generate spans to delete
    deleted = [False] * n
    total_deleted = 0
    attempts = 0

    while total_deleted < n_to_delete and attempts < n * 3:
        start = random.randint(0, n - 1)
        span_len = min(
            random.choices(
                range(1, avg_span_length * 3 + 1),
                weights=[((1 - 1/avg_span_length) ** (i-1)) * (1/avg_span_length)
                         for i in range(1, avg_span_length * 3 + 1)],
                k=1
            )[0],
            n - start
        )

        for i in range(start, min(start + span_len, n)):
            if not deleted[i]:
                deleted[i] = True
                total_deleted += 1
        attempts += 1

    corrupted = [tid for i, tid in enumerate(token_ids) if not deleted[i]]
    deleted_tokens = [tid for i, tid in enumerate(token_ids) if deleted[i]]

    return corrupted, deleted_tokens


def cross_text_insert(token_ids: List[int],
                      donor_ids: List[int],
                      rate: float = 0.10,
                      avg_span_length: int = 5) -> Tuple[List[int], List[bool]]:
    """
    Insert spans from a structurally different text, wrapped in <corrupt> tokens.

    The donor text should be from a different category AND difficulty tier
    to maximize structural mismatch (not just vocabulary mismatch).

    Returns (corrupted_ids, is_foreign_mask) where is_foreign_mask marks
    which tokens in the corrupted sequence are insertions.
    """
    if not token_ids or not donor_ids or rate <= 0:
        return list(token_ids), [False] * len(token_ids)

    n = len(token_ids)
    n_to_insert = max(1, int(n * rate))

    # Extract spans from donor
    donor_spans = []
    remaining = n_to_insert
    while remaining > 0 and len(donor_ids) > avg_span_length:
        span_len = min(
            random.randint(max(1, avg_span_length - 2), avg_span_length + 2),
            remaining,
            len(donor_ids) - 1
        )
        start = random.randint(0, len(donor_ids) - span_len)
        donor_spans.append(donor_ids[start:start + span_len])
        remaining -= span_len

    if not donor_spans:
        return list(token_ids), [False] * len(token_ids)

    # Choose insertion points in the host text (at sentence-ish boundaries)
    insertion_points = sorted(random.sample(
        range(1, n), min(len(donor_spans), n - 1)
    ), reverse=True)

    corrupted = list(token_ids)
    is_foreign = [False] * len(corrupted)

    for point, span in zip(insertion_points, donor_spans):
        # Insert <corrupt> span <corrupt> at the insertion point
        insert = [CORRUPT_START_ID] + span + [CORRUPT_END_ID]
        foreign_mask = [True] * len(insert)

        corrupted = corrupted[:point] + insert + corrupted[point:]
        is_foreign = is_foreign[:point] + foreign_mask + is_foreign[point:]

    return corrupted, is_foreign


# Lookup table for semantic corruption (antonyms/near-misses)
# These are common words with clear opposites or confusable alternatives.
# No secondary model needed — just a lookup.
ANTONYM_TABLE = {
    'good': ['bad', 'evil', 'poor'],
    'bad': ['good', 'fine', 'excellent'],
    'light': ['dark', 'heavy', 'shadow'],
    'dark': ['light', 'bright', 'clear'],
    'love': ['hate', 'fear', 'loathing'],
    'hate': ['love', 'adore', 'cherish'],
    'life': ['death', 'void', 'nothing'],
    'death': ['life', 'birth', 'beginning'],
    'truth': ['lie', 'falsehood', 'fiction'],
    'true': ['false', 'wrong', 'untrue'],
    'false': ['true', 'right', 'correct'],
    'war': ['peace', 'calm', 'harmony'],
    'peace': ['war', 'conflict', 'strife'],
    'beautiful': ['ugly', 'hideous', 'plain'],
    'old': ['new', 'young', 'fresh'],
    'new': ['old', 'ancient', 'worn'],
    'great': ['small', 'terrible', 'petty'],
    'small': ['great', 'large', 'vast'],
    'begin': ['end', 'finish', 'cease'],
    'end': ['begin', 'start', 'commence'],
    'above': ['below', 'beneath', 'under'],
    'below': ['above', 'over', 'beyond'],
    'always': ['never', 'rarely', 'seldom'],
    'never': ['always', 'often', 'forever'],
    'all': ['none', 'nothing', 'few'],
    'none': ['all', 'every', 'many'],
    'heaven': ['hell', 'earth', 'abyss'],
    'sacred': ['profane', 'common', 'base'],
    'strong': ['weak', 'frail', 'feeble'],
    'weak': ['strong', 'mighty', 'powerful'],
    'wise': ['foolish', 'ignorant', 'naive'],
    'right': ['wrong', 'left', 'false'],
    'wrong': ['right', 'correct', 'proper'],
    'free': ['bound', 'captive', 'enslaved'],
    'open': ['closed', 'shut', 'sealed'],
    'first': ['last', 'final', 'ultimate'],
    'last': ['first', 'initial', 'beginning'],
    'joy': ['sorrow', 'grief', 'misery'],
    'sorrow': ['joy', 'delight', 'happiness'],
    'virtue': ['vice', 'sin', 'corruption'],
    'knowledge': ['ignorance', 'folly', 'blindness'],
    'silence': ['noise', 'clamor', 'sound'],
    'rise': ['fall', 'decline', 'sink'],
    'fall': ['rise', 'ascend', 'climb'],
}


def semantic_corrupt(token_ids: List[int], tokenizer,
                     rate: float = 0.05) -> Tuple[List[int], List[int]]:
    """
    Replace individual words with antonyms or near-misses using lookup table.
    No secondary model needed.

    Returns (corrupted_ids, original_ids_at_corrupted_positions).
    """
    text = tokenizer.decode(token_ids)
    words = text.split()

    n_to_corrupt = max(1, int(len(words) * rate))
    candidates = [(i, w.lower().strip('.,;:!?"\'()-'))
                  for i, w in enumerate(words)]
    candidates = [(i, w) for i, w in candidates if w in ANTONYM_TABLE]

    if not candidates:
        return list(token_ids), []

    selected = random.sample(candidates, min(n_to_corrupt, len(candidates)))
    corruption_map = {}
    for i, clean_word in selected:
        replacement = random.choice(ANTONYM_TABLE[clean_word])
        # Preserve original casing
        original = words[i]
        if original[0].isupper():
            replacement = replacement.capitalize()
        if original.isupper():
            replacement = replacement.upper()
        # Preserve trailing punctuation
        trail = ''
        for c in reversed(original):
            if c in '.,;:!?"\'()-':
                trail = c + trail
            else:
                break
        corruption_map[i] = replacement + trail

    corrupted_words = list(words)
    originals_at_positions = []
    for i, replacement in corruption_map.items():
        originals_at_positions.append((i, words[i]))
        corrupted_words[i] = replacement

    corrupted_text = ' '.join(corrupted_words)
    corrupted_encoding = tokenizer.encode(corrupted_text)
    corrupted_ids = [t for t in corrupted_encoding.ids
                     if t not in (BOS_ID, EOS_ID)]

    return corrupted_ids, originals_at_positions


def apply_corruption(token_ids: List[int],
                     strategy: str,
                     rate: float,
                     tokenizer=None,
                     donor_ids: Optional[List[int]] = None) -> dict:
    """
    Apply a corruption strategy. Dispatches to the appropriate function.

    Returns a dict with:
      - corrupted_ids: the corrupted token sequence
      - targets: reconstruction targets (format depends on strategy)
      - strategy: which strategy was used
    """
    if strategy == 'span_mask':
        corrupted, targets = span_mask(token_ids, rate)
        return {'corrupted_ids': corrupted, 'targets': targets, 'strategy': strategy}

    elif strategy == 'sentence_shuffle':
        assert tokenizer is not None, "sentence_shuffle requires tokenizer"
        corrupted, targets = sentence_shuffle(token_ids, tokenizer, rate)
        return {'corrupted_ids': corrupted, 'targets': targets, 'strategy': strategy}

    elif strategy == 'span_deletion':
        corrupted, targets = span_deletion(token_ids, rate)
        return {'corrupted_ids': corrupted, 'targets': targets, 'strategy': strategy}

    elif strategy == 'cross_text_insert':
        assert donor_ids is not None, "cross_text_insert requires donor_ids"
        corrupted, is_foreign = cross_text_insert(token_ids, donor_ids, rate)
        return {'corrupted_ids': corrupted, 'targets': is_foreign, 'strategy': strategy}

    elif strategy == 'semantic_corrupt':
        assert tokenizer is not None, "semantic_corrupt requires tokenizer"
        corrupted, originals = semantic_corrupt(token_ids, tokenizer, rate)
        return {'corrupted_ids': corrupted, 'targets': originals, 'strategy': strategy}

    else:
        raise ValueError(f"Unknown corruption strategy: {strategy}")


def sample_strategy(step: int, total_steps: int) -> str:
    """
    Sample a corruption strategy based on training progress.

    Early training: mostly span_mask (easiest)
    Mid training: mix of all strategies
    Late training: more span_deletion and cross_text_insert (harder)
    """
    progress = min(step / max(total_steps, 1), 1.0)

    if progress < 0.3:
        weights = {
            'span_mask': 0.6,
            'sentence_shuffle': 0.2,
            'semantic_corrupt': 0.15,
            'cross_text_insert': 0.05,
            'span_deletion': 0.0,
        }
    elif progress < 0.7:
        weights = {
            'span_mask': 0.3,
            'sentence_shuffle': 0.2,
            'semantic_corrupt': 0.15,
            'cross_text_insert': 0.15,
            'span_deletion': 0.2,
        }
    else:
        weights = {
            'span_mask': 0.15,
            'sentence_shuffle': 0.15,
            'semantic_corrupt': 0.1,
            'cross_text_insert': 0.25,
            'span_deletion': 0.35,
        }

    strategies = list(weights.keys())
    probs = list(weights.values())
    return random.choices(strategies, weights=probs, k=1)[0]
