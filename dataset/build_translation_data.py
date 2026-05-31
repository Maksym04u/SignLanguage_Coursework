"""Write raw landmark playback clips into ``translation_data/``.

Training data under ``data/<data_dir>/<sequence>/<frame>.npy`` stays
wrist-centered and spread-normalized for the GRU model.

Playback data under ``translation_data/<data_dir>/`` stores **raw**
MediaPipe image coordinates (126 floats per frame) so whole-hand motion
(e.g. Ukrainian ц travelling top-to-bottom) is visible on the frontend.

Raw playback files are written automatically at the end of each recorded
sequence by ``data_collection.py``. This module exposes the shared writer
and a bulk status/rebuild helper.

Layout::

    translation_data/<data_dir>/0.npy   # raw (126,) float vector
    ...
    translation_data/<data_dir>/19.npy
    translation_data/<data_dir>/meta.json
        {"format": "raw_v1", "source_sequence": 15, "frames": 20}

Usage::

    python -m dataset.build_translation_data --status
    python -m dataset.build_translation_data --only uk_letter_ц
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from label_registry import LabelEntry, load_labels  # noqa: E402

logger = logging.getLogger(__name__)

TRANSLATION_DATA_ROOT = PROJECT_ROOT / "translation_data"
FRAMES_PER_SEQUENCE = 20
PLAYBACK_FORMAT = "raw_v1"
FRAME_VECTOR_LEN = 126


def _write_meta(target_dir: Path, source_sequence: int) -> None:
    meta = {
        "format": PLAYBACK_FORMAT,
        "source_sequence": source_sequence,
        "frames": FRAMES_PER_SEQUENCE,
    }
    (target_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_raw_playback(
    data_dir: str,
    raw_frames: Iterable[np.ndarray],
    source_sequence: int,
) -> bool:
    """Persist one raw playback clip (20 frames) for ``data_dir``.

    Called from ``data_collection.py`` after each sequence finishes.
    Returns ``True`` on success.
    """
    frames = list(raw_frames)
    if len(frames) != FRAMES_PER_SEQUENCE:
        logger.warning(
            "Expected %d raw frames for %s, got %d; skipping playback write",
            FRAMES_PER_SEQUENCE,
            data_dir,
            len(frames),
        )
        return False

    target_dir = TRANSLATION_DATA_ROOT / data_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    for existing in target_dir.glob("*.npy"):
        try:
            existing.unlink()
        except OSError:
            pass

    for idx, vec in enumerate(frames):
        arr = np.asarray(vec, dtype=np.float64)
        if arr.shape != (FRAME_VECTOR_LEN,):
            logger.warning(
                "Frame %d for %s has shape %s; expected (%d,)",
                idx,
                data_dir,
                arr.shape,
                FRAME_VECTOR_LEN,
            )
            return False
        np.save(target_dir / f"{idx}.npy", arr)

    _write_meta(target_dir, source_sequence)
    logger.info(
        "translation_data/%s updated (raw_v1, sequence %s, 20 frames)",
        data_dir,
        source_sequence,
    )
    return True


def has_raw_playback(data_dir: str) -> bool:
    """Return True when ``translation_data/<data_dir>`` has a raw playback clip."""
    target_dir = TRANSLATION_DATA_ROOT / data_dir
    meta_path = target_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("format") == PLAYBACK_FORMAT:
                return _sequence_is_complete(target_dir)
        except Exception:
            pass
    return False


def _sequence_is_complete(seq_dir: Path) -> bool:
    return all((seq_dir / f"{i}.npy").is_file() for i in range(FRAMES_PER_SEQUENCE))


def build_for_data_dir(data_dir: str, sequence: int = 0) -> Optional[int]:
    """Legacy hook kept for ``data_collection`` import compatibility.

    Raw playback is written during collection via ``write_raw_playback``.
    This function only reports whether raw playback already exists.
    """
    if has_raw_playback(data_dir):
        meta_path = TRANSLATION_DATA_ROOT / data_dir / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return int(meta.get("source_sequence", sequence))
        except Exception:
            return sequence
    logger.info(
        "translation_data/%s has no raw playback yet; re-record with "
        "data_collection.py to populate it",
        data_dir,
    )
    return None


def build_full(
    sequence: int = 0,
    only: Optional[Iterable[str]] = None,
) -> Tuple[List[str], List[str]]:
    labels = load_labels(str(PROJECT_ROOT / "dataset" / "labels.json"))
    if only:
        wanted = set(only)
        labels = [lbl for lbl in labels if lbl.class_id in wanted or lbl.data_dir in wanted]

    ok: List[str] = []
    skipped: List[str] = []
    for lbl in labels:
        if has_raw_playback(lbl.data_dir):
            ok.append(lbl.class_id)
        else:
            skipped.append(lbl.class_id)
    return ok, skipped


def print_status() -> None:
    labels = load_labels(str(PROJECT_ROOT / "dataset" / "labels.json"))
    raw_ok: List[LabelEntry] = []
    legacy: List[LabelEntry] = []
    missing: List[LabelEntry] = []

    for lbl in labels:
        target = TRANSLATION_DATA_ROOT / lbl.data_dir
        if has_raw_playback(lbl.data_dir):
            raw_ok.append(lbl)
        elif _sequence_is_complete(target):
            legacy.append(lbl)
        else:
            missing.append(lbl)

    print(f"\nRaw playback (raw_v1): {len(raw_ok)}")
    print(f"Legacy normalized (re-record recommended): {len(legacy)}")
    print(f"Missing playback data: {len(missing)}")
    if legacy:
        print("\nLegacy (need re-record for motion gestures):")
        for lbl in legacy[:20]:
            print(f"  - {lbl.class_id} ({lbl.data_dir})")
        if len(legacy) > 20:
            print(f"  ... and {len(legacy) - 20} more")
    if missing:
        print("\nMissing:")
        for lbl in missing[:20]:
            print(f"  - {lbl.class_id} ({lbl.data_dir})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or verify raw playback data in translation_data/. "
            "New recordings write raw clips automatically during data_collection."
        )
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print how many classes have raw vs legacy vs missing playback data.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="ID",
        help="Limit --status to specific class_id or data_dir values.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.status:
        if args.only:
            labels = load_labels(str(PROJECT_ROOT / "dataset" / "labels.json"))
            wanted = set(args.only)
            subset = [
                lbl for lbl in labels if lbl.class_id in wanted or lbl.data_dir in wanted
            ]
            print()
            for lbl in subset:
                state = (
                    "raw_v1"
                    if has_raw_playback(lbl.data_dir)
                    else "legacy/missing"
                )
                print(f"  {lbl.class_id:<20} {lbl.data_dir:<24} {state}")
        else:
            print_status()
        return

    ok, skipped = build_full(only=args.only)
    print(f"Raw playback ready: {len(ok)} class(es).")
    if skipped:
        print(f"Need re-record: {len(skipped)} class(es).")


if __name__ == "__main__":
    main()
