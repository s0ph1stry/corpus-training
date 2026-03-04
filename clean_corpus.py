#!/usr/bin/env python3
"""
Corpus cleaning script for the coherence training project.

Cleans raw downloaded texts:
- Strips Project Gutenberg headers/footers
- Normalizes unicode (dashes, quotes, whitespace)
- Preserves structural elements (line breaks, stanza breaks, speaker labels)
- Removes editorial apparatus, introductions, footnotes where possible

Usage:
    python clean_corpus.py                    # Clean all raw texts
    python clean_corpus.py --file "path"      # Clean a specific file
    python clean_corpus.py --status           # Show cleaning status
    python clean_corpus.py --preview "path"   # Preview cleaning of a file
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
RAW_DIR = PROJECT_DIR / "raw"
CLEANED_DIR = PROJECT_DIR / "cleaned"
SOURCES_CSV = PROJECT_DIR / "sources.csv"
FRONT_MATTER_CUTS = PROJECT_DIR / "front_matter_cuts.json"

# Load LLM-identified front matter cuts if available
_front_matter_cuts = {}
if FRONT_MATTER_CUTS.exists():
    with open(FRONT_MATTER_CUTS, 'r') as f:
        _raw_cuts = json.load(f)
    # Skip _metadata key
    _front_matter_cuts = {k: v for k, v in _raw_cuts.items() if not k.startswith('_')}


# --- Gutenberg Header/Footer Removal ---

# Patterns that mark the START of actual content (after Gutenberg boilerplate)
GUTENBERG_START_MARKERS = [
    r"\*\*\*\s*START OF TH(IS|E) PROJECT GUTENBERG EBOOK",
    r"\*\*\*\s*START OF THE PROJECT GUTENBERG",
    r"^\*\*\* START",
]

# Patterns that mark the END of actual content (before Gutenberg boilerplate)
GUTENBERG_END_MARKERS = [
    r"\*\*\*\s*END OF TH(IS|E) PROJECT GUTENBERG EBOOK",
    r"\*\*\*\s*END OF THE PROJECT GUTENBERG",
    r"^\*\*\* END",
    r"^End of the Project Gutenberg",
    r"^End of Project Gutenberg",
]


def strip_gutenberg_end_marker(lines):
    """Find and return the index of the Gutenberg end marker, or len(lines) if none."""
    end_idx = len(lines)
    search_limit = max(0, len(lines) - 100)
    for i in range(len(lines) - 1, search_limit - 1, -1):
        stripped_line = lines[i].strip()
        for pattern in GUTENBERG_END_MARKERS:
            if re.search(pattern, stripped_line, re.IGNORECASE):
                end_idx = i
                while end_idx > 0 and not lines[end_idx - 1].strip():
                    end_idx -= 1
                return end_idx
    return end_idx


def strip_gutenberg_boilerplate(text):
    """Remove Project Gutenberg header and footer."""
    lines = text.split('\n')

    # Find start of actual content
    start_idx = 0
    for i, line in enumerate(lines):
        for pattern in GUTENBERG_START_MARKERS:
            if re.search(pattern, line, re.IGNORECASE):
                start_idx = i + 1
                # Skip blank lines after the marker
                while start_idx < len(lines) and not lines[start_idx].strip():
                    start_idx += 1
                break
        if start_idx > 0:
            break

    end_idx = strip_gutenberg_end_marker(lines)

    return '\n'.join(lines[start_idx:end_idx])


def apply_front_matter_cut(text, filename):
    """Apply LLM-identified front matter cut for a specific file.

    Uses front_matter_cuts.json start_line (1-indexed) to skip all
    front matter, then strips the Gutenberg end marker from the bottom.
    Returns None if the file is marked as wrong_file.
    """
    cut_info = _front_matter_cuts.get(filename)
    if not cut_info:
        return None  # No cut info — caller should use fallback pipeline

    if cut_info.get('wrong_file'):
        return ''  # Empty string signals skip

    start_line = cut_info.get('start_line')
    if start_line is None:
        return None  # No start_line — use fallback

    lines = text.split('\n')
    # start_line is 1-indexed, convert to 0-indexed
    start_idx = max(0, start_line - 1)
    end_idx = strip_gutenberg_end_marker(lines)

    return '\n'.join(lines[start_idx:end_idx])


# --- Editorial Front Matter Removal ---

# Named editorial sections (not part of the primary text)
EDITORIAL_SECTION_PATTERNS = [
    r'^CONTENTS?\s*\.?\s*$',
    r'^TABLE OF CONTENTS?\s*\.?\s*$',
    r'^INTRODUCTION(\s+AND\s+ANALYSIS)?\s*\.?\s*$',
    r'^PREFACE\s*\.?\s*$',
    r'^FOREWORD\s*\.?\s*$',
    r"^TRANSLATOR'?S?'?\s*(NOTE|PREFACE)\s*\.?\s*$",
    r"^EDITOR'?S?'?\s*(NOTE|PREFACE|INTRODUCTION)\s*\.?\s*$",
    r"^TRANSCRIBER'?S?'?\s*NOTE\s*:?\s*$",
    r'^LIST OF ILLUSTRATIONS?\s*$',
    r'^ILLUSTRATIONS?\s*$',
    r'^INTRODUCTORY NOTE\s*\.?\s*$',
    r'^BIOGRAPHICAL\s+(NOTICE|NOTE|SKETCH)\s*\.?\s*$',
    r'^CHRONOLOG(Y|ICAL TABLE)\s*$',
    r'^ADVERTISEMENT\s*\.?\s*$',
    r'^BIBLIOGRAPH(Y|ICAL NOTE)\s*\.?\s*$',
    r'^A?\s?NOTE ON THE TEXT\s*\.?\s*$',
    r'^ABOUT THE AUTHOR\s*$',
    r'^ABOUT THIS EDITION\s*$',
    # Catch-all: all-caps heading ending with PREFACE or INTRODUCTION
    r"^[A-Z][A-Z'\s.,]{2,40}(PREFACE|INTRODUCTION)\s*\.?\s*$",
]

# Structural markers indicating the primary text has begun
PRIMARY_TEXT_PATTERNS = [
    r'^CHAPTER\s+[IVXLCDM\d]',
    r'^BOOK\s+[IVXLCDM\d]',
    r'^BOOK\s+(THE\s+)?(FIRST|SECOND|THIRD|FOURTH|FIFTH)',
    r'^VOLUME\s+[IVXLCDM\d]',
    r'^PART\s+[IVXLCDM\d]',
    r'^PART\s+(ONE|TWO|THREE|FIRST|SECOND|THE\s+FIRST)',
    r'^FIRST\s+(BOOK|PART)\b',
    r'^ACT\s+[IVXLCDM\d]',
    r'^SCENE\s+[IVXLCDM\d]',
    r'^CANTO\s+[IVXLCDM\d]',
    r'^STROPHE\s',
    r'^DRAMATIS\s+PERSONAE',
    r'^PROPOSITION\s+[IVXLCDM\d]',
    r'^LECTURE\s+[IVXLCDM\d]',
    r'^PSALM\s+[IVXLCDM\d]',
    r'^SUTRA\s',
    r'^MEDITATION\s+[IVXLCDM\d]',
    r'^PROLOGUE\b',
    r'^THE\s+(GENERAL\s+)?PROLOGUE\b',
    r'^APHORISM',
    r'^LETTER\s+[IVXLCDM\d]',
    r'^DIALOGUE\b',
    r'^(THE\s+)?\w+.{0,30}TALE\s*$',  # "The Knight's Tale", etc.
]


def _is_followed_by_content(lines, idx):
    """Check if a heading is followed by actual content (not a ToC entry).

    A ToC entry is followed by other short heading-like lines.
    Real content has prose paragraphs (long lines) or at least several
    non-heading lines.
    """
    non_blank = 0
    long_lines = 0
    structural = 0
    j = idx + 1
    while j < len(lines) and non_blank < 15:
        stripped = lines[j].strip()
        if stripped:
            non_blank += 1
            if len(stripped) > 60:
                long_lines += 1
            if any(re.match(p, stripped, re.IGNORECASE) for p in PRIMARY_TEXT_PATTERNS):
                structural += 1
        j += 1

    if non_blank == 0:
        return False
    # If many following lines are structural markers, this is a ToC
    if structural / non_blank > 0.3:
        return False
    # Real content: has long lines OR has several non-structural lines
    return long_lines >= 2 or (non_blank >= 4 and structural == 0)


def find_text_start(lines):
    """Find the first structural marker followed by real content (not a ToC entry)."""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        for pattern in PRIMARY_TEXT_PATTERNS:
            if re.match(pattern, stripped, re.IGNORECASE):
                if _is_followed_by_content(lines, i):
                    return i
                break  # matched pattern but not followed by content — skip
    return None


def strip_editorial_front_matter(text):
    """
    Remove editorial front matter: ToC, introductions, prefaces,
    transcriber's notes, illustration lists.

    Strategy: Find the first structural marker of the primary text that is
    followed by substantial content (distinguishing real headings from ToC
    entries). Then remove any named editorial sections before that point.

    Editorial sections are bounded: entering at an editorial heading, exiting
    at the next non-editorial heading (all-caps title, structural marker, etc.)
    or at text_start.
    """
    lines = text.split('\n')
    text_start = find_text_start(lines)

    if text_start is None or text_start < 5:
        return text  # No clear structure or trivial front matter

    # Scan front matter zone for editorial sections
    keep = [True] * len(lines)
    in_editorial = False

    for i in range(text_start):
        stripped = lines[i].strip()

        # Check if this is an editorial heading
        is_editorial_heading = any(
            re.match(p, stripped, re.IGNORECASE)
            for p in EDITORIAL_SECTION_PATTERNS
        ) if stripped else False

        # Check if this is a non-editorial heading that should end an editorial section
        is_other_heading = False
        if stripped and not is_editorial_heading:
            is_other_heading = (
                # All-caps heading (short, not a blank line)
                (stripped.isupper() and 3 <= len(stripped) <= 80)
                # Or a primary text pattern
                or any(re.match(p, stripped, re.IGNORECASE) for p in PRIMARY_TEXT_PATTERNS)
            )

        if is_editorial_heading:
            in_editorial = True
            keep[i] = False
            continue

        if in_editorial:
            if is_other_heading:
                # Non-editorial heading ends the editorial section
                in_editorial = False
                # keep[i] stays True — this heading is preserved
            else:
                keep[i] = False

    result = [lines[i] for i in range(len(lines)) if keep[i]]
    return '\n'.join(result)


# --- Unicode Normalization ---

def normalize_unicode(text):
    """Normalize unicode characters for consistency."""
    # Smart quotes -> straight quotes (debatable — but consistent for tokenization)
    text = text.replace('\u201c', '"')   # left double quote
    text = text.replace('\u201d', '"')   # right double quote
    text = text.replace('\u2018', "'")   # left single quote
    text = text.replace('\u2019', "'")   # right single quote

    # Em/en dashes
    text = text.replace('\u2014', '—')   # em dash (keep as is)
    text = text.replace('\u2013', '–')   # en dash (keep as is)

    # Ellipsis
    text = text.replace('\u2026', '...')

    # Non-breaking spaces
    text = text.replace('\u00a0', ' ')

    # Multiple spaces -> single space (but preserve indentation at line starts)
    lines = text.split('\n')
    normalized_lines = []
    for line in lines:
        # Preserve leading whitespace, normalize internal spaces
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]
        normalized = re.sub(r'  +', ' ', stripped)
        normalized_lines.append(indent + normalized)

    return '\n'.join(normalized_lines)


# --- Whitespace Cleaning ---

def clean_whitespace(text):
    """Clean excessive whitespace while preserving structure."""
    # Remove trailing whitespace on each line
    lines = [line.rstrip() for line in text.split('\n')]

    # Collapse runs of 3+ blank lines to 2 (preserve paragraph/stanza breaks)
    cleaned = []
    blank_count = 0
    for line in lines:
        if not line:
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)

    # Remove leading/trailing blank lines
    while cleaned and not cleaned[0]:
        cleaned.pop(0)
    while cleaned and not cleaned[-1]:
        cleaned.pop()

    return '\n'.join(cleaned)


# --- Line Break Normalization ---

def unwrap_prose_paragraphs(text):
    """
    Unwrap hard-wrapped prose paragraphs (common in Gutenberg texts).
    Preserve intentional line breaks in poetry, drama, and lists.

    Heuristic: if a line is 55-80 chars and the next line starts with a
    lowercase letter or continues a sentence, it's a hard wrap.
    """
    lines = text.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Blank line = paragraph break, always preserve
        if not line.strip():
            result.append(line)
            i += 1
            continue

        # Check if this looks like hard-wrapped prose
        # Don't unwrap if:
        #   - Line is very short (< 40 chars) — might be poetry/drama
        #   - Line starts with special chars (speaker labels, numbers, etc.)
        #   - Line is indented (might be verse or block quote)
        stripped = line.strip()

        is_prose_wrap = (
            40 <= len(stripped) <= 80
            and i + 1 < len(lines)
            and lines[i + 1].strip()  # next line is not blank
            and not line.startswith('  ')  # not indented
            and not re.match(r'^[A-Z]{2,}\.?\s', stripped)  # not SPEAKER label
            and not re.match(r'^\d+[\.\)]\s', stripped)  # not numbered list
            and not stripped.endswith(':')  # not a heading/label
        )

        if is_prose_wrap:
            # Check if next line continues the sentence
            next_line = lines[i + 1].strip()
            continues = (
                next_line
                and next_line[0].islower()
                or next_line[0] in ',;—'
                or stripped[-1] not in '.!?"\')'
            )

            if continues:
                # Join with next line
                combined = stripped + ' ' + next_line
                lines[i + 1] = combined  # replace next line with combined
                i += 1
                continue

        result.append(line)
        i += 1

    return '\n'.join(result)


# --- Page Number Removal ---

def remove_page_numbers(text):
    """Remove standalone page numbers."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Remove lines that are just a number (page numbers)
        if re.match(r'^\d{1,4}$', stripped):
            continue
        # Remove lines like "[p. 123]" or "(p. 123)"
        if re.match(r'^[\[\(]p\.?\s*\d+[\]\)]$', stripped, re.IGNORECASE):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)


# --- Footnote Removal ---

def remove_footnotes(text):
    """Remove footnote markers and footnote text where detectable."""
    # Remove footnote markers like [1], [2], etc. in running text
    text = re.sub(r'\[(\d{1,3})\]', '', text)

    # Remove lines that look like footnotes at the bottom
    # (numbered lines after a "NOTES" or "FOOTNOTES" section)
    lines = text.split('\n')
    in_footnotes = False
    cleaned = []

    for line in lines:
        stripped = line.strip()
        if re.match(r'^(NOTES|FOOTNOTES|END\s*NOTES)\s*$', stripped, re.IGNORECASE):
            in_footnotes = True
            continue
        if in_footnotes:
            # End footnotes section at a clear section break
            if re.match(r'^(CHAPTER|BOOK|PART|ACT|SCENE|CANTO)\s', stripped, re.IGNORECASE):
                in_footnotes = False
                cleaned.append(line)
            # Skip footnote lines
            continue
        cleaned.append(line)

    return '\n'.join(cleaned)


# --- Main Cleaning Pipeline ---

def clean_text(text, source_type='gutenberg', preserve_line_breaks=False, filename=None):
    """
    Full cleaning pipeline.

    Args:
        text: Raw text content
        source_type: 'gutenberg', 'manual', 'wikisource', etc.
        preserve_line_breaks: If True, don't unwrap paragraphs (for poetry/drama)
        filename: Basename of the raw file (for front_matter_cuts.json lookup)
    """
    # Step 1: Apply LLM-identified front matter cut if available
    if filename and filename in _front_matter_cuts:
        result = apply_front_matter_cut(text, filename)
        if result == '':
            return ''  # wrong_file — skip
        if result is not None:
            text = result  # LLM cut applied — skip Steps 1b and 2
        else:
            # No usable cut — fall back to heuristic pipeline
            if source_type == 'gutenberg':
                text = strip_gutenberg_boilerplate(text)
            text = strip_editorial_front_matter(text)
    else:
        # No entry in cuts file — use heuristic pipeline
        if source_type == 'gutenberg':
            text = strip_gutenberg_boilerplate(text)
        text = strip_editorial_front_matter(text)

    # Step 3: Remove page numbers
    text = remove_page_numbers(text)

    # Step 4: Remove footnotes
    text = remove_footnotes(text)

    # Step 5: Normalize unicode
    text = normalize_unicode(text)

    # Step 6: Unwrap hard-wrapped prose (unless preserving line breaks)
    if not preserve_line_breaks:
        text = unwrap_prose_paragraphs(text)

    # Step 7: Clean whitespace
    text = clean_whitespace(text)

    return text


# Categories where line breaks are structurally meaningful
PRESERVE_LINEBREAKS_CATEGORIES = {
    'Poetry', 'Poetry/Epic', 'Drama', 'Contemplative',
    'Oral/Nonstandard', 'Poetry/Japanese', 'Sacred/Wisdom',
}


def should_preserve_linebreaks(category, title):
    """Determine if line breaks should be preserved for this text."""
    if any(cat in category for cat in PRESERVE_LINEBREAKS_CATEGORIES):
        return True
    # Specific texts that need line break preservation
    preserve_titles = [
        'Tractatus', 'Pensées', 'Meditations',  # numbered/fragmentary philosophy
        'Sonnets', 'Leaves of Grass', 'Poems',
    ]
    return any(t.lower() in title.lower() for t in preserve_titles)


def clean_file(raw_path, cleaned_path, source_type='gutenberg', category='', title=''):
    """Clean a single file."""
    filename = raw_path.name
    text = raw_path.read_text(encoding='utf-8', errors='replace')
    preserve = should_preserve_linebreaks(category, title)
    cleaned = clean_text(text, source_type=source_type, preserve_line_breaks=preserve, filename=filename)

    if not cleaned:
        return len(text), 0  # wrong_file or empty result

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_path.write_text(cleaned, encoding='utf-8')
    return len(text), len(cleaned)


def cmd_clean_all(args):
    """Clean all downloaded raw texts."""
    if not SOURCES_CSV.exists():
        print("No sources.csv found. Run source_corpus.py download first.")
        return

    sources = {}
    wrong_count = 0
    with open(SOURCES_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            downloaded = row.get('Downloaded', '').strip().lower()
            if downloaded == 'wrong_file':
                wrong_count += 1
                continue
            if downloaded == 'yes' and row.get('Filename'):
                sources[(row['Title'], row['Author'])] = row

    print(f"Found {len(sources)} downloaded texts to clean ({wrong_count} wrong files skipped)")
    cleaned_count = 0
    skipped = 0
    wrong_skipped = 0

    for key, source in sorted(sources.items()):
        raw_path = RAW_DIR / source['Filename']
        cleaned_path = CLEANED_DIR / source['Filename']
        filename = source['Filename']

        if not raw_path.exists():
            print(f"  MISSING: {filename}")
            continue

        # Skip files marked wrong in front_matter_cuts.json
        cut_info = _front_matter_cuts.get(filename, {})
        if cut_info.get('wrong_file'):
            wrong_skipped += 1
            print(f"  WRONG FILE (skipped): {filename}")
            # Remove stale cleaned version if it exists
            if cleaned_path.exists():
                cleaned_path.unlink()
            continue

        if cleaned_path.exists() and not getattr(args, 'force', False):
            skipped += 1
            continue

        raw_size, clean_size = clean_file(
            raw_path, cleaned_path,
            source_type=source.get('Source_Type', 'gutenberg'),
            category=source.get('Category', ''),
            title=source.get('Title', ''),
        )

        if clean_size == 0:
            print(f"  EMPTY (skipped): {filename}")
            continue

        reduction = (1 - clean_size / raw_size) * 100 if raw_size > 0 else 0
        print(f"  Cleaned: {filename} ({raw_size:,} -> {clean_size:,} bytes, -{reduction:.0f}%)")
        cleaned_count += 1

    print(f"\nCleaned {cleaned_count} texts, skipped {skipped} already cleaned, {wrong_skipped} wrong files")


def cmd_preview(args):
    """Preview cleaning of a specific file."""
    path = Path(args.file)
    if not path.exists():
        # Try as filename in raw dir
        path = RAW_DIR / args.file
    if not path.exists():
        print(f"File not found: {args.file}")
        return

    text = path.read_text(encoding='utf-8', errors='replace')
    cleaned = clean_text(text)

    print(f"=== First 100 lines of cleaned text ===")
    for i, line in enumerate(cleaned.split('\n')[:100]):
        print(line)
    print(f"\n=== Raw: {len(text):,} bytes | Cleaned: {len(cleaned):,} bytes ===")


def cmd_status(args):
    """Show cleaning status."""
    raw_files = list(RAW_DIR.glob("*.txt")) if RAW_DIR.exists() else []
    cleaned_files = list(CLEANED_DIR.glob("*.txt")) if CLEANED_DIR.exists() else []

    raw_size = sum(f.stat().st_size for f in raw_files)
    cleaned_size = sum(f.stat().st_size for f in cleaned_files)

    print(f"Cleaning Status")
    print(f"{'='*50}")
    print(f"Raw texts:     {len(raw_files)} files ({raw_size:,} bytes / {raw_size/1024/1024:.1f} MB)")
    print(f"Cleaned texts: {len(cleaned_files)} files ({cleaned_size:,} bytes / {cleaned_size/1024/1024:.1f} MB)")

    if raw_files and cleaned_files:
        reduction = (1 - cleaned_size / raw_size) * 100
        print(f"Size reduction: {reduction:.1f}%")

    uncleaned = set(f.name for f in raw_files) - set(f.name for f in cleaned_files)
    if uncleaned:
        print(f"\nUncleaned files ({len(uncleaned)}):")
        for name in sorted(uncleaned)[:20]:
            print(f"  - {name}")
        if len(uncleaned) > 20:
            print(f"  ... and {len(uncleaned) - 20} more")


def main():
    parser = argparse.ArgumentParser(description="Clean corpus texts")
    sub = parser.add_subparsers()

    cl = sub.add_parser("clean")
    cl.add_argument("--force", action="store_true", help="Re-clean already cleaned files")
    cl.set_defaults(func=cmd_clean_all)

    pr = sub.add_parser("preview")
    pr.add_argument("file", help="File to preview")
    pr.set_defaults(func=cmd_preview)

    st = sub.add_parser("status")
    st.set_defaults(func=cmd_status)

    # Handle bare invocation as "clean"
    if len(sys.argv) == 1:
        sys.argv.append('clean')

    # Handle --flag style
    if len(sys.argv) > 1:
        arg = sys.argv[1].lstrip('-')
        if arg in ('status', 'clean', 'preview'):
            sys.argv[1] = arg

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
