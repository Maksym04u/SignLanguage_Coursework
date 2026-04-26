"""Build a "perfect-example" lexicon of canonical hand poses per class.

The gesture recognition pipeline stores each recorded sequence as a series of
.npy frames in ``data/<data_dir>/<sequence>/<frame>.npy``. Each frame is a
(126,) float vector with two halves:
- first 63 floats: left-hand vector  (21 landmarks * 3 coords, normalized)
- last 63 floats:  right-hand vector (21 landmarks * 3 coords, normalized)

For the reverse-translation feature (text -> sequence of gestures) we only
need ONE representative frame per class. We pick a stable "middle-ish" frame,
averaged across the first few sequences, to wash out small recording jitter.

The output is ``dataset/gesture_lexicon.json``, with the schema:

    {
      "version": 1,
      "entries": [
        {
          "class_id": "en.hello",
          "language": "en",
          "gloss": "hello",
          "display_text": "Hello",
          "data_dir": "Hello",
          "lh": [63 floats],            # may be all zeros if no left hand
          "rh": [63 floats],            # may be all zeros if no right hand
          "source": {"sequences": [0,1,2], "frames": [8,9,10,11,12]}
        },
        ...
      ]
    }
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from label_registry import LabelEntry, load_labels  # noqa: E402

logger = logging.getLogger(__name__)

DATA_ROOT = PROJECT_ROOT / "data"
LEXICON_PATH = PROJECT_ROOT / "dataset" / "gesture_lexicon.json"
HAND_VECTOR_LEN = 63
FRAME_VECTOR_LEN = 2 * HAND_VECTOR_LEN  # 126

# How we pick a representative frame: average across these (sequence, frame)
# combinations when available. They are intentionally near the middle of a
# 30-sequence x 20-frame recording to avoid the moments around the SPACE press.
DEFAULT_SEQUENCES = (0, 1, 2)
DEFAULT_FRAMES = (8, 9, 10, 11, 12)


def _load_frame_vectors(
    data_dir: str,
    sequences: Iterable[int] = DEFAULT_SEQUENCES,
    frames: Iterable[int] = DEFAULT_FRAMES,
) -> List[np.ndarray]:
    """Load valid frame vectors for a class. Silently skips missing files."""
    class_dir = DATA_ROOT / data_dir
    if not class_dir.exists():
        return []

    vectors: List[np.ndarray] = []
    for seq in sequences:
        seq_dir = class_dir / str(seq)
        if not seq_dir.exists():
            continue
        for frame in frames:
            frame_path = seq_dir / f"{frame}.npy"
            if not frame_path.exists():
                continue
            try:
                vec = np.load(frame_path)
            except Exception:
                logger.warning("Failed to load %s", frame_path)
                continue
            if vec.shape != (FRAME_VECTOR_LEN,):
                logger.warning(
                    "Skipping %s with unexpected shape %s", frame_path, vec.shape
                )
                continue
            vectors.append(vec.astype(np.float64))
    return vectors


def _representative_vector(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
    """Average the available frame vectors. Returns None if nothing usable."""
    if not vectors:
        return None

    stack = np.stack(vectors, axis=0)  # (N, 126)

    lh_slice = stack[:, :HAND_VECTOR_LEN]
    rh_slice = stack[:, HAND_VECTOR_LEN:]

    lh_present = np.any(np.abs(lh_slice) > 1e-6, axis=1)
    rh_present = np.any(np.abs(rh_slice) > 1e-6, axis=1)

    # Average ONLY across frames where the hand is actually present so that
    # missing detections do not pull the canonical pose toward the origin.
    if lh_present.any():
        lh = lh_slice[lh_present].mean(axis=0)
    else:
        lh = np.zeros(HAND_VECTOR_LEN)

    if rh_present.any():
        rh = rh_slice[rh_present].mean(axis=0)
    else:
        rh = np.zeros(HAND_VECTOR_LEN)

    return np.concatenate([lh, rh])


def _entry_from_label(
    label: LabelEntry,
    sequences: Iterable[int] = DEFAULT_SEQUENCES,
    frames: Iterable[int] = DEFAULT_FRAMES,
) -> Optional[Dict]:
    vectors = _load_frame_vectors(label.data_dir, sequences=sequences, frames=frames)
    rep = _representative_vector(vectors)
    if rep is None:
        return None
    lh = rep[:HAND_VECTOR_LEN].tolist()
    rh = rep[HAND_VECTOR_LEN:].tolist()
    return {
        "class_id": label.class_id,
        "language": label.language,
        "gloss": label.gloss,
        "display_text": label.display_text,
        "data_dir": label.data_dir,
        "lh": lh,
        "rh": rh,
        "source": {
            "sequences": list(sequences),
            "frames": list(frames),
        },
    }


def _read_lexicon() -> Dict:
    if not LEXICON_PATH.exists():
        return {"version": 1, "entries": []}
    try:
        return json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to parse %s, starting fresh", LEXICON_PATH)
        return {"version": 1, "entries": []}


def _write_lexicon(payload: Dict) -> None:
    LEXICON_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEXICON_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_full_lexicon(
    sequences: Iterable[int] = DEFAULT_SEQUENCES,
    frames: Iterable[int] = DEFAULT_FRAMES,
) -> Dict:
    """Rebuild every lexicon entry from scratch using the labels registry."""
    labels = load_labels(str(PROJECT_ROOT / "dataset" / "labels.json"))
    entries: List[Dict] = []
    skipped: List[str] = []
    for label in labels:
        entry = _entry_from_label(label, sequences=sequences, frames=frames)
        if entry is None:
            skipped.append(label.class_id)
            continue
        entries.append(entry)

    payload = {"version": 1, "entries": sorted(entries, key=lambda e: e["class_id"])}
    _write_lexicon(payload)

    if skipped:
        logger.warning(
            "Lexicon built with %d entries; skipped (no data found): %s",
            len(entries),
            ", ".join(skipped),
        )
    else:
        logger.info("Lexicon built with %d entries", len(entries))
    return payload


def update_lexicon_for_action(
    class_id: str,
    sequences: Iterable[int] = DEFAULT_SEQUENCES,
    frames: Iterable[int] = DEFAULT_FRAMES,
) -> bool:
    """Refresh a single entry. Designed to be called from data_collection.py
    right after a new sequence is recorded so the lexicon never drifts from
    what the model is being trained on.

    Returns True if the lexicon was updated, False otherwise.
    """
    labels = load_labels(str(PROJECT_ROOT / "dataset" / "labels.json"))
    label_by_id = {entry.class_id: entry for entry in labels}
    if class_id not in label_by_id:
        logger.warning("Unknown class_id %s; cannot update lexicon", class_id)
        return False

    entry = _entry_from_label(label_by_id[class_id], sequences=sequences, frames=frames)
    if entry is None:
        logger.info("No data yet for %s; lexicon unchanged", class_id)
        return False

    payload = _read_lexicon()
    existing = payload.get("entries", [])
    new_entries = [e for e in existing if e.get("class_id") != class_id]
    new_entries.append(entry)
    payload["entries"] = sorted(new_entries, key=lambda e: e["class_id"])
    payload["version"] = payload.get("version", 1)

    _write_lexicon(payload)
    logger.info("Lexicon updated for %s", class_id)
    return True


def update_lexicon_for_data_dir(
    data_dir: str,
    sequences: Iterable[int] = DEFAULT_SEQUENCES,
    frames: Iterable[int] = DEFAULT_FRAMES,
) -> bool:
    """Convenience wrapper for callers that only know the on-disk folder name."""
    labels = load_labels(str(PROJECT_ROOT / "dataset" / "labels.json"))
    by_data_dir = {entry.data_dir: entry for entry in labels}
    if data_dir not in by_data_dir:
        logger.warning("Unknown data_dir %s; cannot update lexicon", data_dir)
        return False
    return update_lexicon_for_action(
        by_data_dir[data_dir].class_id,
        sequences=sequences,
        frames=frames,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    payload = build_full_lexicon()
    print(f"Wrote {LEXICON_PATH} with {len(payload['entries'])} entries")


if __name__ == "__main__":
    main()
