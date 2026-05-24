"""Build label entries from human-readable gesture symbols and update labels.json."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Tuple

from label_registry import DATA_ROOT, DEFAULT_FRAMES, DEFAULT_SEQUENCES, LabelEntry, load_labels

LABELS_PATH = Path("dataset/labels.json")


def parse_symbols(raw: str) -> List[str]:
    """
    Split user input into gesture symbols.

    Supports comma-separated and/or one symbol per line. Empty parts are skipped.
    """
    if not raw or not raw.strip():
        return []
    parts: List[str] = []
    for line in raw.replace("\r\n", "\n").split("\n"):
        for chunk in line.split(","):
            symbol = chunk.strip()
            if symbol:
                parts.append(symbol)
    return parts


def gloss_slug(display_text: str) -> str:
    """Stable gloss / class_id suffix for words; single chars returned as-is."""
    text = display_text.strip()
    if len(text) == 1:
        return text
    slug = text.lower()
    slug = slug.replace("'", "").replace("'", "").replace("'", "")
    slug = re.sub(r"[^\w\s\-]", "", slug, flags=re.UNICODE)
    slug = slug.replace("-", "_")
    slug = re.sub(r"\s+", "_", slug.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or text.lower()


def build_label_entry(symbol: str, language: str) -> LabelEntry:
    """
    Create metadata for one gesture symbol following project conventions:

    - letter (len 1): en_letter_a / uk_letter_а, class_id en.a / uk.а
    - word  (len >1): en_word_hello, class_id en.hello, gloss im_good for "I'm good"
    """
    language = language.strip().lower()
    if language not in ("en", "uk"):
        raise ValueError(f"Unsupported language: {language}")

    display_text = symbol.strip()
    if not display_text:
        raise ValueError("Empty gesture symbol")

    gloss = gloss_slug(display_text)
    class_id = f"{language}.{gloss}"

    if len(gloss) == 1:
        data_dir = f"{language}_letter_{gloss}"
    else:
        data_dir = f"{language}_word_{gloss}"

    return LabelEntry(
        class_id=class_id,
        language=language,
        gloss=gloss,
        display_text=display_text,
        data_dir=data_dir,
    )


def build_labels_from_symbols(symbols: Iterable[str], language: str) -> List[LabelEntry]:
    return [build_label_entry(s, language) for s in symbols]


def merge_labels(
    new_entries: List[LabelEntry],
    *,
    labels_path: Path = LABELS_PATH,
) -> Tuple[List[LabelEntry], List[LabelEntry]]:
    """
    Append new entries to labels.json (skip duplicates by class_id or data_dir).

    Returns (all_labels_after_merge, actually_added).
    """
    existing = load_labels(str(labels_path))
    by_class = {e.class_id: e for e in existing}
    by_dir = {e.data_dir: e for e in existing}

    added: List[LabelEntry] = []
    for entry in new_entries:
        if entry.class_id in by_class:
            continue
        if entry.data_dir in by_dir:
            continue
        added.append(entry)
        by_class[entry.class_id] = entry
        by_dir[entry.data_dir] = entry

    merged = sorted(by_class.values(), key=lambda e: e.class_id)
    payload = {"version": 2, "labels": [asdict(e) for e in merged]}
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return merged, added


def prepare_data_directories(
    entries: List[LabelEntry],
    *,
    data_root: Path = DATA_ROOT,
    sequences: int = DEFAULT_SEQUENCES,
) -> None:
    """Create empty sequence folders for recording."""
    for entry in entries:
        for seq in range(sequences):
            (data_root / entry.data_dir / str(seq)).mkdir(parents=True, exist_ok=True)


def register_label_entries(
    entries: List[LabelEntry],
    *,
    labels_path: Path = LABELS_PATH,
) -> Tuple[List[LabelEntry], List[LabelEntry]]:
    """Write entries to labels.json and create data/ sequence folders."""
    _, added = merge_labels(entries, labels_path=labels_path)
    prepare_data_directories(entries)
    return entries, added


def register_gestures(
    symbols: Iterable[str],
    language: str,
    *,
    labels_path: Path = LABELS_PATH,
) -> Tuple[List[LabelEntry], List[LabelEntry]]:
    """
    Parse symbols, update labels.json, create data/ folders.

    Returns (entries_to_record, newly_added_to_json).
    """
    entries = build_labels_from_symbols(symbols, language)
    return register_label_entries(entries, labels_path=labels_path)
