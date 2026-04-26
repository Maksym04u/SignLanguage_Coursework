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

logger = logging.getLogger("uvicorn.error")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LEXICON_PATH = PROJECT_ROOT / "dataset" / "gesture_lexicon.json"


def _is_letter_gloss(gloss: str) -> bool:
    """A gloss is treated as a "letter" entry if it is exactly one character.

    This deliberately matches the dataset convention where letter classes use
    a single-character gloss (e.g. ``"a"``, ``"б"``) and word classes use
    multi-character glosses (e.g. ``"hello"``, ``"name"``).
    """
    return isinstance(gloss, str) and len(gloss) == 1


class TextToGestureService:
    """Loads the gesture lexicon lazily and translates text into frames."""

    def __init__(self, lexicon_path: Path = LEXICON_PATH) -> None:
        self._lexicon_path = lexicon_path
        self._lock = Lock()
        self._loaded = False
        self._words_by_lang: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._letters_by_lang: Dict[str, Dict[str, Dict[str, Any]]] = {}

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

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Split on any whitespace; punctuation is stripped per-token below.
        return [t for t in re.split(r"\s+", text.strip()) if t]

    @staticmethod
    def _strip_token_punctuation(token: str) -> str:
        # Keep apostrophes/hyphens inside words (e.g. "what's", "so-so") since
        # those appear as glosses in our dataset, but trim outer punctuation.
        return re.sub(r"^[^\w'\-]+|[^\w'\-]+$", "", token, flags=re.UNICODE)

    @staticmethod
    def _frame_from_entry(entry: Dict[str, Any], frame_type: str) -> Dict[str, Any]:
        return {
            "type": frame_type,
            "label": entry.get("display_text") or entry.get("gloss", ""),
            "gloss": entry.get("gloss"),
            "class_id": entry.get("class_id"),
            "language": entry.get("language"),
            "lh": entry.get("lh"),
            "rh": entry.get("rh"),
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
            "reason": reason,
        }

    @staticmethod
    def _space_frame() -> Dict[str, Any]:
        return {
            "type": "space",
            "label": " ",
            "gloss": None,
            "class_id": None,
            "language": None,
            "lh": None,
            "rh": None,
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

        for index, raw_token in enumerate(tokens_in):
            token = self._strip_token_punctuation(raw_token).lower()
            if not token:
                continue

            if index > 0:
                frames.append(self._space_frame())

            if token in words:
                frames.append(self._frame_from_entry(words[token], "word"))
                words_matched += 1
                continue

            # Letter-by-letter fallback.
            for ch in token:
                if ch in letters:
                    frames.append(self._frame_from_entry(letters[ch], "letter"))
                    letters_matched += 1
                else:
                    frames.append(self._missing_frame(ch, "no_letter_gesture"))
                    letters_missing.append(ch)

        return {
            "language": language,
            "frames": frames,
            "summary": {
                "input_tokens": len(tokens_in),
                "words_matched": words_matched,
                "letters_matched": letters_matched,
                "letters_missing": letters_missing,
            },
        }


text_to_gesture_service = TextToGestureService()
