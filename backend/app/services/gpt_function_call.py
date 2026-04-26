import json
from typing import Any

TOOL_NAME = "set_corrected_text"

GRAMMAR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Returns corrected text preserving original meaning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "corrected_text": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["corrected_text"],
                "additionalProperties": False,
            },
        },
    }
]

GRAMMAR_TOOL_CHOICE = {"type": "function", "function": {"name": TOOL_NAME}}


def extract_corrected_text(message: Any) -> str | None:
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return None
    function = getattr(tool_calls[0], "function", None)
    if function is None:
        return None
    raw_args = getattr(function, "arguments", None)
    if not raw_args:
        return None
    try:
        payload = json.loads(raw_args)
    except Exception:
        return None
    value = payload.get("corrected_text")
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None
