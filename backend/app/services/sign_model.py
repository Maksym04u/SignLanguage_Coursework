import json
import os
import threading
from pathlib import Path
from typing import List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "my_model.h5"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "dataset" / "labels.json"


class SignModelService:
    """
    Lazy-loads the trained Keras gesture classifier and the label registry once.
    Inference is thread-safe; the FastAPI worker calls predict() on the request
    thread, and Keras releases the GIL during model.predict.
    """

    BUFFER_FRAMES = 20
    KEYPOINTS_PER_FRAME = 126

    def __init__(self) -> None:
        self._model = None
        self._labels: List[dict] | None = None
        self._lock = threading.Lock()
        self._model_path = Path(os.getenv("SIGN_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        self._labels_path = Path(os.getenv("SIGN_LABELS_PATH", str(DEFAULT_LABELS_PATH)))

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._labels is not None:
            return
        with self._lock:
            if self._labels is None:
                payload = json.loads(self._labels_path.read_text(encoding="utf-8"))
                entries = list(payload["labels"])
                entries.sort(key=lambda item: item["class_id"])
                self._labels = entries
            if self._model is None:
                from keras._tf_keras.keras.models import load_model

                self._model = load_model(str(self._model_path))

    def predict(self, keypoints: List[List[float]]) -> dict:
        self._ensure_loaded()
        arr = np.asarray(keypoints, dtype=np.float32)
        if arr.ndim != 2 or arr.shape != (self.BUFFER_FRAMES, self.KEYPOINTS_PER_FRAME):
            raise ValueError(
                f"Expected ({self.BUFFER_FRAMES}, {self.KEYPOINTS_PER_FRAME}) keypoints, got {arr.shape}"
            )

        prediction = self._model.predict(arr[np.newaxis, :, :], verbose=0)[0]
        index = int(np.argmax(prediction))
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
            "language": entry.get("language", "en"),
            "display_text": entry.get("display_text", entry["class_id"]),
            "confidence": confidence,
        }


sign_model_service = SignModelService()
