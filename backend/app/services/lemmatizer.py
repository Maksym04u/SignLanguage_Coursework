"""Word lemmatization via OpenAI for the text-to-gesture lookup.

Reduces inflected words to their dictionary base form so a whole-word gesture
can still be matched when the user typed a variant (``loves`` -> ``love``;
``любить`` -> ``любити``). Mirrors :class:`GrammarService`'s lazy-client and
graceful-degradation pattern: when ``OPENAI_API_KEY`` is missing or a request
fails, lemmatization is simply skipped and the caller falls back to spelling
the word out letter by letter.
"""

import logging
import os
from typing import Dict, List

from .gpt_lemmatize import (
    LEMMATIZE_TOOL_CHOICE,
    LEMMATIZE_TOOLS,
    extract_lemmas,
)
from .gpt_prompts import LemmatizePrompts

logger = logging.getLogger("uvicorn.error")


class LemmatizerService:
    MODEL = "gpt-4o-mini"
    TIMEOUT_SECONDS = 15

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self._client = None
        self._client_init_error = None
        # (language, word) -> lemma. Lemmas are intrinsic to the word, so this
        # cache is safe regardless of how the gesture vocabulary changes.
        self._cache: Dict[tuple, str] = {}

    def _client_or_none(self):
        if self._client is not None:
            return self._client
        if self._client_init_error is not None:
            return None
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not self.api_key:
            self._client_init_error = "OPENAI_API_KEY is not set"
            logger.error("Lemmatizer disabled: %s", self._client_init_error)
            return None
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, timeout=self.TIMEOUT_SECONDS)
            logger.info("Lemmatizer backend: OpenAI model=%s", self.MODEL)
            return self._client
        except Exception as exc:  # pragma: no cover - defensive guard
            self._client_init_error = str(exc)
            logger.exception("Failed to initialize OpenAI lemmatizer client")
            return None

    def lemmatize(self, words: List[str], source_language: str) -> Dict[str, str]:
        """Map each input word (lowercased) to its base form.

        Returns ``{word: lemma}``. Words that cannot be resolved are absent.
        Cached results are reused; only uncached words hit the API, in a single
        batched request.
        """
        lang = (source_language or "en").lower()

        cleaned: List[str] = []
        seen = set()
        for w in words:
            wl = (w or "").strip().lower()
            if wl and wl not in seen:
                seen.add(wl)
                cleaned.append(wl)
        if not cleaned:
            return {}

        result: Dict[str, str] = {}
        pending: List[str] = []
        for w in cleaned:
            key = (lang, w)
            if key in self._cache:
                result[w] = self._cache[key]
            else:
                pending.append(w)

        if not pending:
            return result

        client = self._client_or_none()
        if client is None:
            return result

        try:
            completion = client.chat.completions.create(
                model=self.MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": LemmatizePrompts.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": LemmatizePrompts.user_prompt(pending, lang),
                    },
                ],
                tools=LEMMATIZE_TOOLS,
                tool_choice=LEMMATIZE_TOOL_CHOICE,
            )
            message = completion.choices[0].message
            mapping = extract_lemmas(message)
            for w in pending:
                lemma = mapping.get(w, w)
                self._cache[(lang, w)] = lemma
                result[w] = lemma
            logger.info(
                "Lemmatize | language=%s input=%r output=%r",
                lang,
                pending,
                {w: result[w] for w in pending},
            )
        except Exception:
            logger.exception("OpenAI lemmatization failed for source_language=%s", lang)

        return result


lemmatizer_service = LemmatizerService()
