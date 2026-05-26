import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

DATA_ROOT = Path("data")
DEFAULT_SEQUENCES = 30
DEFAULT_FRAMES = 20

# Padded one-hot size for language conditioning (slots 2–3 reserved for future langs).
LANGUAGE_DIM = 4
LANGUAGE_INDEX: Dict[str, int] = {"en": 0, "uk": 1}


def language_from_data_dir(data_dir: str) -> str:
    """Parse language code from folder names like ``en_letter_a`` / ``uk_letter_а``."""
    code = data_dir.split("_", 1)[0].lower()
    if code not in LANGUAGE_INDEX:
        raise ValueError(
            f"Cannot infer language from data_dir={data_dir!r}; "
            f"expected prefix in {list(LANGUAGE_INDEX)}"
        )
    return code


def language_to_vector(language: str) -> List[float]:
    """One-hot language vector of length LANGUAGE_DIM."""
    code = language.lower()
    if code not in LANGUAGE_INDEX:
        raise ValueError(f"Unsupported language: {language}")
    vec = [0.0] * LANGUAGE_DIM
    vec[LANGUAGE_INDEX[code]] = 1.0
    return vec


@dataclass(frozen=True)
class LabelEntry:
    class_id: str
    language: str
    gloss: str
    display_text: str
    data_dir: str


def load_labels(labels_path: str = "dataset/labels.json") -> List[LabelEntry]:
    payload = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    labels = [LabelEntry(**entry) for entry in payload["labels"]]
    labels.sort(key=lambda item: item.class_id)
    return labels


def class_ids(labels: List[LabelEntry]) -> List[str]:
    return [entry.class_id for entry in labels]


def filter_labels(
    labels: List[LabelEntry],
    *,
    language: str | None = None,
    class_ids_filter: List[str] | None = None,
) -> List[LabelEntry]:
    """Return labels matching optional language and/or class_id allow-list."""
    out = labels
    if language is not None:
        out = [e for e in out if e.language == language]
    if class_ids_filter is not None:
        allowed = set(class_ids_filter)
        out = [e for e in out if e.class_id in allowed]
    return out


def class_to_display(labels: List[LabelEntry]) -> Dict[str, str]:
    return {entry.class_id: entry.display_text for entry in labels}


def class_to_data_dir(labels: List[LabelEntry]) -> Dict[str, str]:
    return {entry.class_id: entry.data_dir for entry in labels}


def has_complete_data(
    data_dir: str,
    *,
    data_root: Path = DATA_ROOT,
    sequences: int = DEFAULT_SEQUENCES,
    frames: int = DEFAULT_FRAMES,
) -> bool:
    """True when every sequence/frame .npy file exists for this class."""
    for sequence in range(sequences):
        for frame in range(frames):
            path = data_root / data_dir / str(sequence) / f"{frame}.npy"
            if not path.is_file():
                return False
    return True


def labels_with_complete_data(labels: List[LabelEntry]) -> List[LabelEntry]:
    return [entry for entry in labels if has_complete_data(entry.data_dir)]


def labels_missing_data(labels: List[LabelEntry]) -> List[LabelEntry]:
    return [entry for entry in labels if not has_complete_data(entry.data_dir)]

