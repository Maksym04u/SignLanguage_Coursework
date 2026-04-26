import re
from typing import List


class TranslatorService:
    """Compose human-readable text from recognized gesture tokens."""

    @staticmethod
    def compose_text(tokens: List[str]) -> str:
        cleaned = [t for t in tokens if isinstance(t, str) and t != ""]
        if not cleaned:
            return ""

        # Support both token styles:
        # 1) legacy stream tokens that already include spacing ("I ", "go ")
        # 2) API/docs tokens without spacing ("I", "go", "school")
        if any(t.startswith(" ") or t.endswith(" ") for t in cleaned):
            text = "".join(cleaned)
        else:
            text = " ".join(cleaned)

        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return text
