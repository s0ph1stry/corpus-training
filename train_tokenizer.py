#!/usr/bin/env python3
"""
Train a custom BPE tokenizer on the cleaned corpus.

Uses HuggingFace's tokenizers library (Rust-backed, fast).
Trains on the full cleaned corpus so the vocabulary reflects
this specific collection, not web text.

BPE Dropout notes:
    The tokenizer is saved without dropout (dropout=None) for deterministic
    inference and downstream use. The bpe_dropout value stored in config.json
    is intended for the DATA PIPELINE only — pass it when constructing a
    BPE model used to tokenize training batches, not during evaluation or
    generation. Example for data pipeline:
        from tokenizers import Tokenizer, models
        tok = Tokenizer.from_file("tokenizer/tokenizer.json")
        # Re-wrap the vocab for stochastic tokenization during training:
        dropout_model = models.BPE.from_pretrained("tokenizer/", dropout=0.1)

Factored embeddings:
    config.json includes embedding_factored, embedding_inner_dim, and
    embedding_dim fields. Instead of a (vocab_size x embedding_dim) matrix,
    factored embeddings use two smaller matrices:
        (vocab_size x inner_dim)  +  (inner_dim x embedding_dim)
    This significantly reduces parameter count when vocab_size is large.

Usage:
    python train_tokenizer.py                          # Train with defaults (16K vocab)
    python train_tokenizer.py --vocab-size 32000        # Train with 32K vocab
    python train_tokenizer.py --bpe-dropout 0.1         # Set BPE dropout rate
    python train_tokenizer.py --test                    # Test tokenizer on sample passages
    python train_tokenizer.py --compare-vocab-sizes     # Compare 8K/16K/24K/32K vocab sizes
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from tokenizers.normalizers import Sequence, NFD, StripAccents, Lowercase

PROJECT_DIR = Path(__file__).parent
CLEANED_DIR = PROJECT_DIR / "cleaned"
TOKENIZER_DIR = PROJECT_DIR / "tokenizer"

# Special tokens for the architecture
SPECIAL_TOKENS = [
    "<pad>",       # Padding
    "<bos>",       # Beginning of sequence
    "<eos>",       # End of sequence
    "<mask>",      # Masked span (denoising)
    "<sep>",       # Encoder-decoder boundary
    "<corrupt>",   # Corruption marker
    "<unk>",       # Unknown token
]


def get_corpus_files():
    """Get all cleaned text files."""
    if not CLEANED_DIR.exists():
        print("No cleaned texts found. Run clean_corpus.py first.")
        sys.exit(1)

    files = sorted(CLEANED_DIR.glob("*.txt"))
    if not files:
        print("No .txt files in cleaned directory.")
        sys.exit(1)

    return files


def train_tokenizer(vocab_size=16000, min_frequency=2, bpe_dropout=0.1):
    """
    Train a BPE tokenizer on the cleaned corpus.

    The tokenizer is saved without dropout for deterministic inference.
    The bpe_dropout value is recorded in config.json for use in the data
    pipeline when constructing stochastic tokenizations during training.
    See module docstring for data-pipeline usage pattern.
    """

    print(f"Training BPE tokenizer with vocab size {vocab_size}")

    # Initialize BPE model — no dropout here; saved tokenizer must be deterministic.
    # bpe_dropout is for the data pipeline only (see module docstring).
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Pre-tokenization: split on whitespace and punctuation
    # This preserves newlines as meaningful tokens
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Punctuation(),
    ])

    # No normalization — we already normalized during cleaning
    # and we want to preserve the exact text (case-sensitive, etc.)

    # Decoder to reconstruct text from tokens
    tokenizer.decoder = decoders.BPEDecoder()

    # Trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Get corpus files
    files = get_corpus_files()
    file_paths = [str(f) for f in files]
    total_size = sum(f.stat().st_size for f in files)

    print(f"Training on {len(files)} files ({total_size:,} bytes / {total_size/1024/1024:.1f} MB)")

    # Train
    tokenizer.train(file_paths, trainer)

    # Add post-processor for BOS/EOS tokens
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <sep> $B <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
            ("<sep>", tokenizer.token_to_id("<sep>")),
        ],
    )

    # Enable padding
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("<pad>"),
        pad_token="<pad>",
    )

    # Save
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer_path = TOKENIZER_DIR / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    # Embedding parameter counts for reference
    actual_vocab = tokenizer.get_vocab_size()
    embedding_dim = 256
    embedding_inner_dim = 64
    full_params = actual_vocab * embedding_dim
    factored_params = actual_vocab * embedding_inner_dim + embedding_inner_dim * embedding_dim

    # Save config separately for reference
    config = {
        "vocab_size": actual_vocab,
        "special_tokens": SPECIAL_TOKENS,
        "min_frequency": min_frequency,
        "num_training_files": len(files),
        "training_size_bytes": total_size,
        "model_type": "BPE",
        # Factored embedding config — used by downstream model code
        "embedding_factored": True,
        "embedding_inner_dim": embedding_inner_dim,
        "embedding_dim": embedding_dim,  # reference only; model config may override
        # BPE dropout — use this rate in the DATA PIPELINE only, not for inference
        "bpe_dropout": bpe_dropout,
    }
    config_path = TOKENIZER_DIR / "config.json"
    config_path.write_text(json.dumps(config, indent=2))

    print(f"\nTokenizer saved to {tokenizer_path}")
    print(f"Vocabulary size: {actual_vocab}")
    print(f"BPE dropout rate (for data pipeline): {bpe_dropout}")
    print(f"Embedding params  — full: {full_params:,}  factored: {factored_params:,}  "
          f"({factored_params/full_params*100:.1f}% of full)")

    return tokenizer


def _vocab_stats(tokenizer):
    """
    Return a dict of vocabulary composition statistics.

    Counts single-character tokens, whole-word tokens (no leading Ġ/space-prefix
    byte or subword marker), and a Counter of token lengths in characters.
    """
    vocab = tokenizer.get_vocab()
    special_set = set(SPECIAL_TOKENS)

    single_char = 0
    whole_word = 0
    length_counter = Counter()

    for token in vocab:
        if token in special_set:
            continue
        # Strip any leading Ġ (ByteLevel space marker) or ## (WordPiece) for length
        display = token.lstrip("Ġ").lstrip("##")
        char_len = len(display)
        length_counter[char_len] += 1

        if char_len == 1:
            single_char += 1

        # Whole-word heuristic: token has no subword marker and is not a
        # leading-space-stripped fragment (i.e., starts with Ġ in ByteLevel BPE,
        # meaning it follows a space — a word boundary).
        # We count it as whole-word if it started with Ġ (the word-initial marker).
        if token.startswith("Ġ"):
            whole_word += 1

    return {
        "single_char": single_char,
        "whole_word": whole_word,
        "length_counter": length_counter,
    }


def _print_length_histogram(length_counter, max_len=12):
    """Print a simple text histogram of token lengths."""
    total = sum(length_counter.values())
    print(f"  Token length distribution (chars):")
    for length in range(1, max_len + 1):
        count = length_counter.get(length, 0)
        pct = count / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"    {length:>2} chars: {count:>6,}  ({pct:4.1f}%)  {bar}")
    longer = sum(v for k, v in length_counter.items() if k > max_len)
    if longer:
        pct = longer / total * 100
        print(f"    >{max_len} chars: {longer:>6,}  ({pct:4.1f}%)")


def test_tokenizer(tokenizer=None):
    """Test the tokenizer on sample passages from the corpus."""
    if tokenizer is None:
        tokenizer_path = TOKENIZER_DIR / "tokenizer.json"
        if not tokenizer_path.exists():
            print("No trained tokenizer found. Run train_tokenizer.py first.")
            return
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    files = get_corpus_files()
    vocab_size = tokenizer.get_vocab_size()

    print(f"\nTokenizer Test")
    print(f"{'='*60}")
    print(f"Vocabulary size: {vocab_size}")
    print()

    # Test on random passages from the corpus
    test_passages = []
    for _ in range(5):
        f = random.choice(files)
        text = f.read_text(encoding='utf-8')
        # Pick a random paragraph
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
        if paragraphs:
            p = random.choice(paragraphs)
            # Truncate to ~200 chars for display
            if len(p) > 200:
                p = p[:200] + "..."
            test_passages.append((f.stem, p))

    for source, passage in test_passages:
        encoding = tokenizer.encode(passage)
        tokens = encoding.tokens
        n_tokens = len(tokens)
        chars_per_token = len(passage) / n_tokens if n_tokens > 0 else 0

        print(f"Source: {source}")
        print(f"Text ({len(passage)} chars): {passage[:80]}...")
        print(f"Tokens ({n_tokens}): {' '.join(tokens[:20])}...")
        print(f"Chars/token: {chars_per_token:.1f}")
        print()

    # Corpus-wide statistics
    print(f"\nCorpus-wide Statistics")
    print(f"{'='*60}")

    total_tokens = 0
    total_chars = 0
    file_stats = []

    for f in files:
        text = f.read_text(encoding='utf-8')
        encoding = tokenizer.encode(text)
        n_tokens = len(encoding.tokens)
        total_tokens += n_tokens
        total_chars += len(text)
        file_stats.append((f.stem, n_tokens, len(text)))

    print(f"Total files: {len(files)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average chars/token: {total_chars/total_tokens:.1f}")
    print(f"Estimated training tokens: {total_tokens:,}")
    print()

    # Top 10 largest files by token count
    file_stats.sort(key=lambda x: x[1], reverse=True)
    print("Largest files by token count:")
    for name, n_tok, n_char in file_stats[:10]:
        print(f"  {name}: {n_tok:,} tokens ({n_char:,} chars)")

    # Check for unknown tokens
    unk_id = tokenizer.token_to_id("<unk>")
    unk_count = 0
    for f in files:
        text = f.read_text(encoding='utf-8')
        encoding = tokenizer.encode(text)
        unk_count += encoding.ids.count(unk_id)

    print(f"\nUnknown tokens in corpus: {unk_count} ({unk_count/total_tokens*100:.4f}%)")

    # Vocabulary composition
    print(f"\nVocabulary Composition")
    print(f"{'='*60}")
    vstats = _vocab_stats(tokenizer)
    n_special = len(SPECIAL_TOKENS)
    print(f"  Special tokens:          {n_special:>6,}")
    print(f"  Single-character tokens: {vstats['single_char']:>6,}")
    print(f"  Whole-word tokens (Ġ-prefixed): {vstats['whole_word']:>6,}")
    print()
    _print_length_histogram(vstats["length_counter"])

    # Factored embedding parameter savings
    embedding_dim = 256
    embedding_inner_dim = 64
    full_params = vocab_size * embedding_dim
    factored_params = vocab_size * embedding_inner_dim + embedding_inner_dim * embedding_dim
    savings_pct = (1 - factored_params / full_params) * 100
    print(f"\nEmbedding parameter estimates (dim={embedding_dim}, inner_dim={embedding_inner_dim}):")
    print(f"  Full embeddings:    {full_params:>10,}  ({vocab_size} x {embedding_dim})")
    print(f"  Factored:           {factored_params:>10,}  "
          f"({vocab_size} x {embedding_inner_dim} + {embedding_inner_dim} x {embedding_dim})")
    print(f"  Parameter savings:  {full_params - factored_params:>10,}  ({savings_pct:.1f}% reduction)")

    # Vocabulary composition sample
    vocab = tokenizer.get_vocab()
    # Sort by ID to see special tokens first, then learned tokens
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    print(f"\nFirst 20 vocabulary entries:")
    for token, id in sorted_vocab[:20]:
        print(f"  {id:>5}: {repr(token)}")

    print(f"\nLast 20 vocabulary entries (most specific):")
    for token, id in sorted_vocab[-20:]:
        print(f"  {id:>5}: {repr(token)}")


def compare_vocab_sizes(min_frequency=2):
    """
    Train temporary tokenizers at 8K, 16K, 24K, and 32K vocab sizes and
    report comparative statistics. Nothing is saved to disk.
    """
    sizes = [8000, 16000, 24000, 32000]
    embedding_dim = 256
    embedding_inner_dim = 64

    files = get_corpus_files()
    file_paths = [str(f) for f in files]
    total_size = sum(f.stat().st_size for f in files)

    # Read all corpus text once
    print(f"Loading corpus ({len(files)} files, {total_size/1024/1024:.1f} MB)...")
    corpus_texts = [f.read_text(encoding='utf-8') for f in files]
    total_chars = sum(len(t) for t in corpus_texts)

    print(f"\nVocab Size Comparison  (embedding_dim={embedding_dim}, inner_dim={embedding_inner_dim})")
    print(f"{'='*90}")
    header = (f"{'Vocab':>8}  {'Actual':>8}  {'Chars/Tok':>9}  {'Total Toks':>12}  "
              f"{'Full Emb Params':>16}  {'Factored Params':>16}  {'Savings':>8}")
    print(header)
    print(f"{'-'*90}")

    for vocab_size in sizes:
        print(f"  Training {vocab_size//1000}K tokenizer...", end="", flush=True)

        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Punctuation(),
        ])
        tok.decoder = decoders.BPEDecoder()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train(file_paths, trainer)

        actual_vocab = tok.get_vocab_size()

        # Corpus token count
        total_tokens = sum(len(tok.encode(text).tokens) for text in corpus_texts)
        chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0

        full_params = actual_vocab * embedding_dim
        factored_params = actual_vocab * embedding_inner_dim + embedding_inner_dim * embedding_dim
        savings_pct = (1 - factored_params / full_params) * 100

        print(f"\r  {vocab_size:>8,}  {actual_vocab:>8,}  {chars_per_token:>9.2f}  "
              f"{total_tokens:>12,}  {full_params:>16,}  {factored_params:>16,}  "
              f"{savings_pct:>7.1f}%")

    print(f"{'='*90}")
    print("Nothing saved — these are comparison runs only.")


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on corpus")
    parser.add_argument("--vocab-size", type=int, default=16000,
                        help="Vocabulary size (default: 16000)")
    parser.add_argument("--min-frequency", type=int, default=2,
                        help="Minimum token frequency (default: 2)")
    parser.add_argument("--bpe-dropout", type=float, default=0.1,
                        help="BPE dropout rate stored in config for data pipeline use "
                             "(default: 0.1). The saved tokenizer itself uses no dropout "
                             "for deterministic inference.")
    parser.add_argument("--test", action="store_true",
                        help="Test existing tokenizer on sample passages")
    parser.add_argument("--train-and-test", action="store_true",
                        help="Train tokenizer and then test it")
    parser.add_argument("--compare-vocab-sizes", action="store_true",
                        help="Train tokenizers at 8K/16K/24K/32K and report comparative "
                             "stats (chars/token, total tokens, embedding params). "
                             "Nothing is saved.")

    args = parser.parse_args()

    if args.compare_vocab_sizes:
        compare_vocab_sizes(min_frequency=args.min_frequency)
    elif args.test:
        test_tokenizer()
    elif args.train_and_test:
        tok = train_tokenizer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            bpe_dropout=args.bpe_dropout,
        )
        test_tokenizer(tok)
    else:
        train_tokenizer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            bpe_dropout=args.bpe_dropout,
        )


if __name__ == "__main__":
    main()
