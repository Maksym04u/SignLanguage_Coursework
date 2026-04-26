import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


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


def class_to_display(labels: List[LabelEntry]) -> Dict[str, str]:
    return {entry.class_id: entry.display_text for entry in labels}


def class_to_data_dir(labels: List[LabelEntry]) -> Dict[str, str]:
    return {entry.class_id: entry.data_dir for entry in labels}

