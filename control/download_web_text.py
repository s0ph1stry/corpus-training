#!/usr/bin/env python3
"""
Download ~117MB of C4 (Colossal Clean Crawled Corpus) to match corpus volume.

Uses HuggingFace datasets to stream C4 and save a matching amount of text.
The control uses the same tokenizer as the corpus model (trained on literary
text, not web text) — this is intentional. The comparison isolates the
effect of text selection, not tokenizer fit.

Usage:
    python -m control.download_web_text
    python -m control.download_web_text --target-mb 117
"""

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
CONTROL_DIR = PROJECT_DIR / 'control'


def download_c4(target_bytes: int, output_dir: Path, seed: int = 42):
    """Stream C4 and save until we reach the target byte count."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'web_text.txt'

    print(f"Downloading C4 web text ({target_bytes / 1024 / 1024:.0f} MB target)...")

    # Stream C4 (English, validation split for reproducibility)
    dataset = load_dataset('allenai/c4', 'en', split='validation', streaming=True)

    total_bytes = 0
    n_docs = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in dataset:
            text = example['text']
            text_bytes = len(text.encode('utf-8'))

            f.write(text)
            f.write('\n\n')

            total_bytes += text_bytes
            n_docs += 1

            if n_docs % 1000 == 0:
                print(f"  {n_docs:,} docs, {total_bytes / 1024 / 1024:.1f} MB")

            if total_bytes >= target_bytes:
                break

    print(f"\nDone: {n_docs:,} documents, {total_bytes / 1024 / 1024:.1f} MB")
    print(f"Saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Download C4 web text for control")
    parser.add_argument('--target-mb', type=int, default=117,
                        help="Target size in MB (default: 117, matching corpus)")
    args = parser.parse_args()

    target_bytes = args.target_mb * 1024 * 1024
    data_dir = CONTROL_DIR / 'data'
    download_c4(target_bytes, data_dir)


if __name__ == '__main__':
    main()
