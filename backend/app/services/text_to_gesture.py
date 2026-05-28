"""Text-to-gesture translation service.

Given an input string and a target sign language, walks the text token by
token and returns a sequence of gesture frames using a "word-first, letters
fallback" strategy:

1.  Tokenize the text by whitespace.
2.  For each token (lowercased):
    * If the lexicon has a *whole-word* gesture for that token in the target
      language, emit a single ``word`` frame.
    * Otherwise, spell the token out letter by letter using ``letter`` frames.
      Letters that have no gesture entry are emitted as ``missing`` frames so
      the frontend can show a clear placeholder.
3.  Insert a ``space`` separator between tokens so the UI can visually break
    words apart.

The lexicon is the JSON file produced by ``dataset/build_gesture_lexicon.py``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("uvicorn.error")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEXICON_PATH = PROJECT_ROOT / "dataset" / "gesture_lexicon.json"
TRANSLATION_DATA_ROOT = PROJECT_ROOT / "translation_data"

HAND_VECTOR_LEN = 63
FRAMES_PER_SEQUENCE = 20


def _is_letter_gloss(gloss: str) -> bool:
    """A gloss is treated as a "letter" entry if it is exactly one character.

    This deliberately matches the dataset convention where letter classes use
    a single-character gloss (e.g. ``"a"``, ``"б"``) and word classes use
    multi-character glosses (e.g. ``"hello"``, ``"name"``).
    """
    return isinstance(gloss, str) and len(gloss) == 1


class TextToGestureService:
    """Loads the gesture lexicon lazily and translates text into frames."""

    def __init__(
        self,
        lexicon_path: Path = LEXICON_PATH,
        translation_data_root: Path = TRANSLATION_DATA_ROOT,
    ) -> None:
        self._lexicon_path = lexicon_path
        self._translation_data_root = translation_data_root
        self._lock = Lock()
        self._loaded = False
        self._words_by_lang: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._letters_by_lang: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # data_dir -> [{"lh": [...63 floats], "rh": [...63 floats]}, x20]
        self._sequence_cache: Dict[str, List[Dict[str, List[float]]]] = {}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._load_unlocked()
            self._loaded = True

    def _load_unlocked(self) -> None:
        if not self._lexicon_path.exists():
            logger.warning(
                "Gesture lexicon missing at %s; text-to-gesture will return only "
                "missing frames until you run dataset/build_gesture_lexicon.py",
                self._lexicon_path,
            )
            return

        payload = json.loads(self._lexicon_path.read_text(encoding="utf-8"))
        entries = payload.get("entries", [])

        words: Dict[str, Dict[str, Dict[str, Any]]] = {}
        letters: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for entry in entries:
            language = entry.get("language", "en")
            gloss = (entry.get("gloss") or "").lower()
            if not gloss:
                continue
            target = letters if _is_letter_gloss(gloss) else words
            target.setdefault(language, {})[gloss] = entry

        self._words_by_lang = words
        self._letters_by_lang = letters
        logger.info(
            "Gesture lexicon loaded: %s",
            ", ".join(
                f"{lang}={len(words.get(lang, {}))}w/{len(letters.get(lang, {}))}l"
                for lang in sorted(set(list(words.keys()) + list(letters.keys())))
            )
            or "<empty>",
        )

    def reload(self) -> None:
        """Force re-reading the lexicon file (useful after data collection)."""
        with self._lock:
            self._loaded = False
            self._words_by_lang = {}
            self._letters_by_lang = {}
            self._sequence_cache = {}

    def _load_sequence(self, data_dir: Optional[str]) -> Optional[List[Dict[str, List[float]]]]:
        """Return a 20-frame sequence for ``data_dir`` from ``translation_data/``.

        Each entry is ``{"lh": [...63], "rh": [...63]}``. Cached after first
        load. Returns ``None`` when the folder is missing or incomplete; the
        caller falls back to the single representative pose from the lexicon.
        """
        if not data_dir:
            return None
        cached = self._sequence_cache.get(data_dir)
        if cached is not None:
            return cached

        dir_path = self._translation_data_root / data_dir
        if not dir_path.is_dir():
            return None

        frames: List[Dict[str, List[float]]] = []
        for frame_idx in range(FRAMES_PER_SEQUENCE):
            frame_path = dir_path / f"{frame_idx}.npy"
            if not frame_path.is_file():
                return None
            try:
                vec = np.load(frame_path)
            except Exception:
                logger.exception("Failed to load %s", frame_path)
                return None
            if vec.shape != (2 * HAND_VECTOR_LEN,):
                logger.warning(
                    "Skipping %s with unexpected shape %s", frame_path, vec.shape
                )
                return None
            frames.append(
                {
                    "lh": vec[:HAND_VECTOR_LEN].astype(float).tolist(),
                    "rh": vec[HAND_VECTOR_LEN:].astype(float).tolist(),
                }
            )

        self._sequence_cache[data_dir] = frames
        return frames

    # Characters that we treat as "part of a word" when grouping tokens so
    # things like "what's" or "self-care" arrive intact for lexicon lookup.
    _WORD_INNER_CHARS = "'\u2019-"

    @classmethod
    def _tokenize(cls, text: str) -> List[Dict[str, str]]:
        """Split ``text`` into ordered (kind, value) tokens.

        ``kind`` is one of:
        * ``"word"``  - a run of letters (optionally containing apostrophes
                         or hyphens) that we will look up in the lexicon.
        * ``"ws"``    - a run of whitespace -> emitted as a silent " ".
        * ``"punct"`` - any single non-letter, non-whitespace character.
                         Emitted as a silent frame so it shows up in the
                         built sentence without playing on the stage.
        """
        tokens: List[Dict[str, str]] = []
        i = 0
        n = len(text)
        inner = cls._WORD_INNER_CHARS
        while i < n:
            ch = text[i]
            if ch.isspace():
                j = i
                while j < n and text[j].isspace():
                    j += 1
                tokens.append({"kind": "ws", "value": text[i:j]})
                i = j
            elif ch.isalpha():
                j = i
                while j < n and (text[j].isalpha() or text[j] in inner):
                    j += 1
                tokens.append({"kind": "word", "value": text[i:j]})
                i = j
            else:
                tokens.append({"kind": "punct", "value": ch})
                i += 1
        return tokens

    def _frame_from_entry(self, entry: Dict[str, Any], frame_type: str) -> Dict[str, Any]:
        return {
            "type": frame_type,
            "label": entry.get("display_text") or entry.get("gloss", ""),
            "gloss": entry.get("gloss"),
            "class_id": entry.get("class_id"),
            "language": entry.get("language"),
            "lh": entry.get("lh"),
            "rh": entry.get("rh"),
            "sequence": self._load_sequence(entry.get("data_dir")),
        }

    @staticmethod
    def _missing_frame(label: str, reason: str) -> Dict[str, Any]:
        return {
            "type": "missing",
            "label": label,
            "gloss": None,
            "class_id": None,
            "language": None,
            "lh": None,
            "rh": None,
            "sequence": None,
            "reason": reason,
        }

    @staticmethod
    def _silent_frame(label: str) -> Dict[str, Any]:
        """A frame that is appended to the sentence but never shown on stage.

        Used for whitespace and punctuation so the built sentence reads as
        the user typed it, without breaking gesture playback rhythm.
        """
        return {
            "type": "silent",
            "label": label,
            "gloss": None,
            "class_id": None,
            "language": None,
            "lh": None,
            "rh": None,
            "sequence": None,
        }

    def translate(self, text: str, language: str) -> Dict[str, Any]:
        """Return gesture frames + a small summary block."""
        self._ensure_loaded()
        language = (language or "en").lower()
        words = self._words_by_lang.get(language, {})
        letters = self._letters_by_lang.get(language, {})

        tokens_in = self._tokenize(text or "")
        frames: List[Dict[str, Any]] = []

        words_matched = 0
        letters_matched = 0
        letters_missing: List[str] = []
        word_tokens = 0

        for tok in tokens_in:
            kind = tok["kind"]
            value = tok["value"]

            if kind == "ws":
                # Collapse any run of whitespace into one silent space.
                frames.append(self._silent_frame(" "))
                continue

            if kind == "punct":
                frames.append(self._silent_frame(value))
                continue

            # kind == "word"
            word_tokens += 1
            lowered = value.lower()

            if lowered in words:
                frames.append(self._frame_from_entry(words[lowered], "word"))
                words_matched += 1
                continue

            # Letter-by-letter fallback. Apostrophes/hyphens inside a word
            # become silent frames so the sentence still shows them.
            for ch in lowered:
                if ch in letters:
                    frames.append(self._frame_from_entry(letters[ch], "letter"))
                    letters_matched += 1
                elif ch in self._WORD_INNER_CHARS:
                    frames.append(self._silent_frame(ch))
                else:
                    frames.append(self._missing_frame(ch, "no_letter_gesture"))
                    letters_missing.append(ch)

        return {
            "language": language,
            "frames": frames,
            "summary": {
                "input_tokens": word_tokens,
                "words_matched": words_matched,
                "letters_matched": letters_matched,
                "letters_missing": letters_missing,
            },
        }


text_to_gesture_service = TextToGestureService()
