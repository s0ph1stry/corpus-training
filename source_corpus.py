#!/usr/bin/env python3
"""
Corpus sourcing script for the coherence training project.

Downloads public domain texts from Project Gutenberg and other sources.
Tracks download status in sources.csv.

Usage:
    python source_corpus.py --download          # Download all PD texts
    python source_corpus.py --status            # Show download status
    python source_corpus.py --search "title"    # Search Gutenberg for a text
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

# --- Configuration ---

PROJECT_DIR = Path(__file__).parent
RAW_DIR = PROJECT_DIR / "raw"
SOURCES_CSV = PROJECT_DIR / "sources.csv"

GUTENBERG_API = "https://gutendex.com/books/"
GUTENBERG_TEXT_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
GUTENBERG_MIRROR = "https://www.gutenberg.org/files/{id}/{id}-0.txt"

# Rate limiting for API requests
REQUEST_DELAY = 1.0  # seconds between requests


# --- Known Gutenberg IDs ---
# Manually verified IDs for faster downloading without API search.
# Format: (title_fragment, author_fragment) -> gutenberg_id

KNOWN_GUTENBERG_IDS = {
    # Fiction
    ("Great Gatsby", "Fitzgerald"): 64317,
    ("Dubliners", "Joyce"): 2814,
    ("Heart of Darkness", "Conrad"): 219,
    ("Notes from Underground", "Dostoevsky"): 600,
    ("Brothers Karamazov", "Dostoevsky"): 28054,
    ("Crime and Punishment", "Dostoevsky"): 2554,
    ("Idiot", "Dostoevsky"): 2638,
    ("Metamorphosis", "Kafka"): 5200,
    ("Turn of the Screw", "James"): 209,
    ("Siddhartha", "Hesse"): 2500,
    ("Death of Ivan", "Tolstoy"): 927,
    ("Billy Budd", "Melville"): 21422,
    ("Christmas Carol", "Dickens"): 46,
    ("Anna Karenina", "Tolstoy"): 1399,
    ("War and Peace", "Tolstoy"): 2600,
    ("Pride and Prejudice", "Austen"): 1342,
    ("Persuasion", "Austen"): 105,
    ("Middlemarch", "Eliot"): 145,
    ("Wuthering Heights", "Brontë"): 768,
    ("Jane Eyre", "Brontë"): 1260,
    ("Frankenstein", "Shelley"): 84,
    ("Don Quixote", "Cervantes"): 996,
    ("Candide", "Voltaire"): 19942,
    ("Gulliver", "Swift"): 829,
    ("Trial", "Kafka"): 7849,
    ("Castle", "Kafka"): 14858,
    ("Demons", "Dostoevsky"): 8117,
    ("Jekyll and Hyde", "Stevenson"): 43,
    ("Moby-Dick", "Melville"): 2701,
    ("Flatland", "Abbott"): 97,
    ("Alice", "Carroll"): 11,
    ("Looking-Glass", "Carroll"): 12,
    ("Winnie-the-Pooh", "Milne"): 67098,
    ("Velveteen Rabbit", "Williams"): 11757,

    # Poetry
    ("Poems", "Dickinson"): 12242,
    ("Sonnets", "Shakespeare"): 1041,
    ("Paradise Lost", "Milton"): 26,
    ("Leaves of Grass", "Whitman"): 1322,
    ("Iliad", "Homer"): 6130,
    ("Odyssey", "Homer"): 1727,
    ("Divine Comedy", "Dante"): 8800,
    ("Aeneid", "Virgil"): 228,
    ("Metamorphoses", "Ovid"): 21765,
    ("Songs of Innocence", "Blake"): 1934,

    # Philosophy
    ("Meditations", "Aurelius"): 2680,
    ("Essays", "Montaigne"): 3600,
    ("Republic", "Plato"): 1497,
    ("Symposium", "Plato"): 1600,
    ("Apology", "Plato"): 1656,
    ("Phaedo", "Plato"): 1658,
    ("Meno", "Plato"): 1643,
    ("Gorgias", "Plato"): 1672,
    ("Nicomachean Ethics", "Aristotle"): 8438,
    ("Poetics", "Aristotle"): 1974,
    ("Tao Te Ching", "Laozi"): 216,
    ("Pensées", "Pascal"): 18269,
    ("Enquiry", "Hume"): 9662,
    ("Discourses", "Epictetus"): 10661,
    ("Confessions", "Augustine"): 3296,
    ("Beyond Good and Evil", "Nietzsche"): 4363,
    ("Genealogy of Moral", "Nietzsche"): 52319,
    ("Thus Spoke Zarathustra", "Nietzsche"): 1998,
    ("Ecce Homo", "Nietzsche"): 52190,
    ("Gay Science", "Nietzsche"): 52881,
    ("Meditations on First", "Descartes"): 59,
    ("Social Contract", "Rousseau"): 46333,
    ("Two Treatises", "Locke"): 7370,
    ("Critique of Pure Reason", "Kant"): 4280,
    ("On Liberty", "Mill"): 34901,
    ("Leviathan", "Hobbes"): 3207,
    ("Fundamental Principles", "Kant"): 5682,
    ("Sickness Unto Death", "Kierkegaard"): 44305,

    # Essays
    ("Self-Reliance", "Emerson"): 16643,
    ("Walden", "Thoreau"): 205,
    ("Civil Disobedience", "Thoreau"): 71357,
    ("Narrative", "Douglass"): 23,
    ("Areopagitica", "Milton"): 608,
    ("Common Sense", "Paine"): 147,
    ("Autobiography", "Franklin"): 148,
    ("Democracy in America", "Tocqueville"): 815,
    ("Souls of Black Folk", "Du Bois"): 408,
    ("Vindication", "Wollstonecraft"): 3420,
    ("Varieties of Religious", "James"): 621,
    ("Interpretation of Dreams", "Freud"): 38219,
    ("Protestant Ethic", "Weber"): 46773,
    ("Communist Manifesto", "Marx"): 61,
    ("Wealth of Nations", "Smith"): 3300,

    # Drama
    ("Hamlet", "Shakespeare"): 1524,
    ("King Lear", "Shakespeare"): 1532,
    ("Macbeth", "Shakespeare"): 1533,
    ("Tempest", "Shakespeare"): 23042,
    ("Othello", "Shakespeare"): 1531,

    # History
    ("Peloponnesian", "Thucydides"): 7142,
    ("Histories", "Herodotus"): 2131,
    ("Decline and Fall", "Gibbon"): 25717,
    ("Prince", "Machiavelli"): 1232,

    # Science
    ("Origin of Species", "Darwin"): 1228,
    ("Voyage of the Beagle", "Darwin"): 944,
    ("Chemical History of a Candle", "Faraday"): 14474,

    # Sacred/Contemplative
    ("Bhagavad Gita", ""): 2388,
    ("Analects", "Confucius"): 4094,
    ("Dhammapada", ""): 2017,

    # Oral/Epic
    ("Canterbury Tales", "Chaucer"): 2383,
    ("Beowulf", ""): 16328,
    ("Epic of Gilgamesh", ""): 74882,
    ("Kalevala", ""): 5186,
    ("Poetic Edda", ""): 14726,

    # Craft/Design
    ("Book of Tea", "Okakura"): 769,
    ("Book of Five Rings", "Musashi"): 55802,
    ("Art of War", "Sun Tzu"): 132,

    # Fables
    ("Grimm", "Grimm"): 2591,
    ("Aesop", "Aesop"): 11339,
    ("1001 Nights", ""): 34206,

    # Legal
    ("Common Law", "Holmes"): 2449,
}


def search_gutenberg(query, author=None):
    """Search the Gutenberg catalog via gutendex API."""
    params = {"search": query}
    if author:
        params["search"] = f"{query} {author}"

    try:
        resp = requests.get(GUTENBERG_API, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except Exception as e:
        print(f"  Search error: {e}")
        return []


def download_gutenberg_text(gutenberg_id, save_path):
    """Download plain text from Project Gutenberg."""
    urls = [
        GUTENBERG_TEXT_URL.format(id=gutenberg_id),
        GUTENBERG_MIRROR.format(id=gutenberg_id),
    ]

    for url in urls:
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                text = resp.text
                # Basic validation: should be substantial text
                if len(text) > 1000:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    save_path.write_text(text, encoding="utf-8")
                    return True, len(text)
        except Exception as e:
            continue

    return False, 0


def download_url(url, save_path):
    """Download from an arbitrary URL."""
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(resp.text, encoding="utf-8")
        return True, len(resp.text)
    except Exception as e:
        print(f"  Download error: {e}")
        return False, 0


def find_gutenberg_id(title, author):
    """Try to find a Gutenberg ID for a text, first from known IDs, then via search."""
    # Check known IDs
    for (t_frag, a_frag), gid in KNOWN_GUTENBERG_IDS.items():
        if t_frag.lower() in title.lower():
            if not a_frag or a_frag.lower() in author.lower():
                return gid

    # Search API
    results = search_gutenberg(title, author)
    if results:
        # Return the first result's ID
        return results[0]["id"]

    return None


def make_filename(title, author):
    """Create a clean filename from title and author."""
    # Clean up for filesystem
    clean = f"{author} - {title}" if author and author != "—" else title
    clean = re.sub(r'[<>:"/\\|?*]', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    # Truncate if too long
    if len(clean) > 120:
        clean = clean[:120]
    return clean + ".txt"


def load_reading_list():
    """Load the reading list CSV."""
    reading_list_path = PROJECT_DIR / "reading-list.csv"
    entries = []
    with open(reading_list_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    return entries


def load_sources():
    """Load existing sources tracking CSV."""
    if not SOURCES_CSV.exists():
        return {}
    sources = {}
    with open(SOURCES_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['Title'], row['Author'])
            sources[key] = row
    return sources


def save_sources(sources):
    """Save sources tracking CSV."""
    fieldnames = ['Title', 'Author', 'Category', 'Status', 'Source_Type',
                  'Source_ID', 'Source_URL', 'Filename', 'File_Size', 'Downloaded']
    with open(SOURCES_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(sources.keys()):
            writer.writerow(sources[key])


def cmd_download(args):
    """Download all public domain texts."""
    entries = load_reading_list()
    sources = load_sources()

    pd_entries = [e for e in entries
                  if 'Public Domain' in e.get('Status', '')
                  or 'Free' in e.get('Status', '')
                  or 'Archive' in e.get('Status', '')]

    print(f"Found {len(pd_entries)} public domain / freely available entries")
    print(f"Already tracked: {len(sources)} entries")
    print()

    downloaded = 0
    skipped = 0
    failed = 0

    for entry in pd_entries:
        title = entry['Title']
        author = entry['Author']
        key = (title, author)

        # Check if already downloaded
        if key in sources and sources[key].get('Downloaded') == 'yes':
            filename = sources[key].get('Filename', '')
            if filename and (RAW_DIR / filename).exists():
                skipped += 1
                continue

        print(f"Processing: {title} by {author}")

        filename = make_filename(title, author)
        save_path = RAW_DIR / filename

        # Try Gutenberg first
        gid = find_gutenberg_id(title, author)
        if gid:
            print(f"  Gutenberg ID: {gid}")
            success, size = download_gutenberg_text(gid, save_path)
            if success:
                sources[key] = {
                    'Title': title,
                    'Author': author,
                    'Category': entry.get('Category', ''),
                    'Status': entry.get('Status', ''),
                    'Source_Type': 'gutenberg',
                    'Source_ID': str(gid),
                    'Source_URL': GUTENBERG_TEXT_URL.format(id=gid),
                    'Filename': filename,
                    'File_Size': str(size),
                    'Downloaded': 'yes',
                }
                downloaded += 1
                print(f"  Downloaded: {size:,} bytes -> {filename}")
                save_sources(sources)
                time.sleep(REQUEST_DELAY)
                continue

        # If Gutenberg failed, record as needing manual sourcing
        sources[key] = {
            'Title': title,
            'Author': author,
            'Category': entry.get('Category', ''),
            'Status': entry.get('Status', ''),
            'Source_Type': 'manual',
            'Source_ID': '',
            'Source_URL': '',
            'Filename': '',
            'File_Size': '',
            'Downloaded': 'no',
        }
        failed += 1
        print(f"  Not found on Gutenberg — needs manual sourcing")
        save_sources(sources)
        time.sleep(REQUEST_DELAY)

    print()
    print(f"Results: {downloaded} downloaded, {skipped} already had, {failed} need manual sourcing")
    save_sources(sources)


def cmd_status(args):
    """Show download status."""
    sources = load_sources()
    entries = load_reading_list()

    total = len(entries)
    pd_entries = [e for e in entries
                  if 'Public Domain' in e.get('Status', '')
                  or 'Free' in e.get('Status', '')
                  or 'Archive' in e.get('Status', '')]
    need_source = [e for e in entries if 'Need to Source' in e.get('Status', '')]

    downloaded = sum(1 for s in sources.values() if s.get('Downloaded') == 'yes')
    manual_needed = sum(1 for s in sources.values() if s.get('Downloaded') == 'no')

    total_size = sum(int(s.get('File_Size', 0)) for s in sources.values()
                     if s.get('Downloaded') == 'yes')

    print(f"Corpus Status")
    print(f"{'='*50}")
    print(f"Total entries in reading list:    {total}")
    print(f"Public domain / freely available: {len(pd_entries)}")
    print(f"Need to source (copyrighted):     {len(need_source)}")
    print()
    print(f"Downloaded:                       {downloaded}")
    print(f"Need manual sourcing:             {manual_needed}")
    print(f"Not yet attempted:                {len(pd_entries) - downloaded - manual_needed}")
    print()
    print(f"Total downloaded size:            {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")

    if manual_needed > 0:
        print()
        print("Texts needing manual sourcing:")
        for key, s in sorted(sources.items()):
            if s.get('Downloaded') == 'no':
                print(f"  - {s['Title']} ({s['Author']})")


def cmd_search(args):
    """Search Gutenberg for a specific text."""
    query = args.query
    print(f"Searching Gutenberg for: {query}")
    results = search_gutenberg(query)

    if not results:
        print("No results found.")
        return

    for r in results[:10]:
        authors = ", ".join(a["name"] for a in r.get("authors", []))
        print(f"  ID {r['id']:>6}: {r['title']} — {authors}")


def main():
    parser = argparse.ArgumentParser(description="Corpus sourcing for coherence training")
    sub = parser.add_subparsers()

    dl = sub.add_parser("download", aliases=["--download"])
    dl.set_defaults(func=cmd_download)

    st = sub.add_parser("status", aliases=["--status"])
    st.set_defaults(func=cmd_status)

    sr = sub.add_parser("search", aliases=["--search"])
    sr.add_argument("query", help="Search query")
    sr.set_defaults(func=cmd_search)

    # Handle --flag style args for convenience
    if len(sys.argv) > 1:
        arg = sys.argv[1].lstrip('-')
        if arg == 'download':
            sys.argv[1] = 'download'
        elif arg == 'status':
            sys.argv[1] = 'status'
        elif arg == 'search':
            sys.argv[1] = 'search'

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
