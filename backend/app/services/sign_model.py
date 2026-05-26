import json
import os
import sys
import threading
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from label_registry import LANGUAGE_DIM, language_to_vector
DEFAULT_MODEL_PATH = PROJECT_ROOT / "my_model.h5"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "dataset" / "model_labels.json"
FALLBACK_LABELS_PATH = PROJECT_ROOT / "dataset" / "labels.json"


class SignModelService:
    """
    Lazy-loads the trained Keras gesture classifier and the label manifest.
    Expects a two-input model: [sequence (20, 126), language one-hot (LANGUAGE_DIM,)].
    """

    BUFFER_FRAMES = 20
    KEYPOINTS_PER_FRAME = 126

    def __init__(self) -> None:
        self._model = None
        self._labels: List[dict] | None = None
        self._language_dim: int = LANGUAGE_DIM
        self._lock = threading.Lock()
        self._model_path = Path(os.getenv("SIGN_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        labels_env = os.getenv("SIGN_LABELS_PATH")
        if labels_env:
            self._labels_path = Path(labels_env)
        elif DEFAULT_LABELS_PATH.is_file():
            self._labels_path = DEFAULT_LABELS_PATH
        else:
            self._labels_path = FALLBACK_LABELS_PATH

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._labels is not None:
            return
        with self._lock:
            if self._labels is None:
                payload = json.loads(self._labels_path.read_text(encoding="utf-8"))
                entries = list(payload["labels"])
                entries.sort(key=lambda item: item["class_id"])
                self._labels = entries
                self._language_dim = int(payload.get("language_dim", LANGUAGE_DIM))
            if self._model is None:
                from keras._tf_keras.keras.models import load_model

                self._model = load_model(str(self._model_path))

    def _indices_for_language(self, source_language: str) -> List[int]:
        code = source_language.lower()
        return [i for i, entry in enumerate(self._labels) if entry.get("language") == code]

    def predict(self, keypoints: List[List[float]], source_language: str = "en") -> dict:
        self._ensure_loaded()
        arr = np.asarray(keypoints, dtype=np.float32)
        if arr.ndim != 2 or arr.shape != (self.BUFFER_FRAMES, self.KEYPOINTS_PER_FRAME):
            raise ValueError(
                f"Expected ({self.BUFFER_FRAMES}, {self.KEYPOINTS_PER_FRAME}) keypoints, got {arr.shape}"
            )

        lang_vec = np.asarray(language_to_vector(source_language), dtype=np.float32)
        if lang_vec.shape != (self._language_dim,):
            raise ValueError(
                f"Language vector length {lang_vec.shape[0]} != model language_dim {self._language_dim}"
            )

        prediction = self._model.predict(
            [arr[np.newaxis, :, :], lang_vec[np.newaxis, :]],
            verbose=0,
        )[0]

        candidate_indices = self._indices_for_language(source_language)
        if not candidate_indices:
            raise ValueError(f"No trained classes for language={source_language!r}")

        index = max(candidate_indices, key=lambda i: float(prediction[i]))
        confidence = float(prediction[index])

        if not (0 <= index < len(self._labels)):
            return {
                "class_id": "unknown",
                "language": "unknown",
                "display_text": "?",
                "confidence": confidence,
            }

        entry = self._labels[index]
        return {
            "class_id": entry["class_id"],
            "language": entry.get("language", source_language),
            "display_text": entry.get("display_text", entry["class_id"]),
            "confidence": confidence,
        }


sign_model_service = SignModelService()
