"""OpenAI function-call spec for word lemmatization.

Mirrors ``gpt_function_call.py`` (the grammar tool) but returns the dictionary
base form (lemma) of each input word so the text-to-gesture lookup can match a
whole-word gesture even when the user typed an inflected form
(``loves``/``loved`` -> ``love``; ``любить``/``люблю`` -> ``любити``).
"""

import json
from typing import Any, Dict

LEMMATIZE_TOOL_NAME = "set_word_lemmas"

LEMMATIZE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": LEMMATIZE_TOOL_NAME,
            "description": (
                "Return the dictionary base form (lemma) of each given word, "
                "in the same language. English: loves/loved/loving -> love. "
                "Ukrainian verbs: reduce to the infinitive ending in -ти "
                "(любить/люблю -> любити)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lemmas": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "word": {"type": "string"},
                                "lemma": {"type": "string"},
                            },
                            "required": ["word", "lemma"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["lemmas"],
                "additionalProperties": False,
            },
        },
    }
]

LEMMATIZE_TOOL_CHOICE = {"type": "function", "function": {"name": LEMMATIZE_TOOL_NAME}}


def extract_lemmas(message: Any) -> Dict[str, str]:
    """Parse the tool call into a ``{word: lemma}`` map (both lowercased)."""
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return {}
    function = getattr(tool_calls[0], "function", None)
    if function is None:
        return {}
    raw_args = getattr(function, "arguments", None)
    if not raw_args:
        return {}
    try:
        payload = json.loads(raw_args)
    except Exception:
        return {}
    items = payload.get("lemmas")
    if not isinstance(items, list):
        return {}

    result: Dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        word = item.get("word")
        lemma = item.get("lemma")
        if not isinstance(word, str) or not isinstance(lemma, str):
            continue
        key = word.strip().lower()
        value = lemma.strip().lower()
        if key and value:
            result[key] = value
    return result
