import logging
import os

from .gpt_function_call import GRAMMAR_TOOL_CHOICE, GRAMMAR_TOOLS, extract_corrected_text
from .gpt_prompts import GrammarPrompts

# Route grammar logs to the same stream uvicorn prints to.
logger = logging.getLogger("uvicorn.error")


class GrammarService:
    MODEL = "gpt-4o-mini"
    TIMEOUT_SECONDS = 15

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.log_each_request = True
        self._client = None
        self._client_init_error = None

    def _client_or_none(self):
        if self._client is not None:
            return self._client
        if self._client_init_error is not None:
            return None
        if not self.api_key:
            self._client_init_error = "OPENAI_API_KEY is not set"
            logger.error(self._client_init_error)
            return None
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, timeout=self.TIMEOUT_SECONDS)
            logger.info("Grammar backend: OpenAI model=%s", self.MODEL)
            return self._client
        except Exception as exc:  # pragma: no cover - defensive guard for runtime env issues
            self._client_init_error = str(exc)
            logger.exception("Failed to initialize OpenAI grammar client")
            return None

    def correct(self, text: str, source_language: str) -> str:
        normalized = text.strip()
        if not normalized:
            return text

        client = self._client_or_none()
        if client is None:
            return text

        try:
            completion = client.chat.completions.create(
                model=self.MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": GrammarPrompts.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": GrammarPrompts.user_prompt(normalized, source_language),
                    },
                ],
                tools=GRAMMAR_TOOLS,
                tool_choice=GRAMMAR_TOOL_CHOICE,
            )
            message = completion.choices[0].message
            corrected = extract_corrected_text(message)
            if not corrected:
                corrected = normalized
            if self.log_each_request:
                logger.info(
                    "Grammar request | backend=openai model=%s language=%s changed=%s input=%r output=%r",
                    self.MODEL,
                    source_language,
                    corrected != normalized,
                    normalized,
                    corrected,
                )
            return corrected
        except Exception:
            logger.exception("OpenAI grammar correction failed for source_language=%s", source_language)
            return text
