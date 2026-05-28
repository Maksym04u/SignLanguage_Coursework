"""Copy one full 20-frame sequence per gesture class into ``translation_data/``.

The training set under ``data/<data_dir>/<sequence>/<frame>.npy`` contains many
recordings per class. For the text-to-gesture playback feature we only need
*one* representative sequence per class so the frontend can animate through
20 frames and show the gesture as a tiny movie clip instead of a static pose.

This script flattens the chosen sequence into:

    translation_data/<data_dir>/0.npy
    translation_data/<data_dir>/1.npy
    ...
    translation_data/<data_dir>/19.npy

Each ``.npy`` is the same (126,) wrist-centered, scale-normalized vector that
the model is trained on, so no additional normalization is required at serve
time. The frontend just unpacks each frame into left/right hand 63-floats and
draws them on a canvas.

Usage::

    python -m dataset.build_translation_data            # all classes, seq 15
    python -m dataset.build_translation_data --sequence 17
    python -m dataset.build_translation_data --only en.hello uk.а
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from label_registry import LabelEntry, load_labels  # noqa: E402

logger = logging.getLogger(__name__)

DATA_ROOT = PROJECT_ROOT / "data"
TRANSLATION_DATA_ROOT = PROJECT_ROOT / "translation_data"
FRAMES_PER_SEQUENCE = 20
DEFAULT_SEQUENCE = 15  # middle of 0..29; far from the SPACE-press boundaries


def _candidate_sequences(class_dir: Path, preferred: int) -> List[int]:
    """Return sequence indices to try, starting with the preferred one."""
    if not class_dir.is_dir():
        return []
    available: List[int] = []
    for child in class_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            available.append(int(child.name))
        except ValueError:
            continue
    available.sort()
    if not available:
        return []
    ordered = [preferred] if preferred in available else []
    ordered.extend(s for s in available if s != preferred)
    return ordered


def _sequence_is_complete(seq_dir: Path) -> bool:
    return all((seq_dir / f"{i}.npy").is_file() for i in range(FRAMES_PER_SEQUENCE))


def build_for_data_dir(
    data_dir: str,
    sequence: int = DEFAULT_SEQUENCE,
) -> Optional[int]:
    """Copy a single complete sequence for ``data_dir`` into ``translation_data/``.

    Returns the sequence index that was used, or ``None`` when no complete
    sequence is available on disk yet.
    """
    class_dir = DATA_ROOT / data_dir
    target_dir = TRANSLATION_DATA_ROOT / data_dir

    for candidate in _candidate_sequences(class_dir, sequence):
        src_seq_dir = class_dir / str(candidate)
        if not _sequence_is_complete(src_seq_dir):
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        for existing in target_dir.glob("*.npy"):
            try:
                existing.unlink()
            except OSError:
                pass
        for frame in range(FRAMES_PER_SEQUENCE):
            shutil.copy2(src_seq_dir / f"{frame}.npy", target_dir / f"{frame}.npy")
        logger.info(
            "translation_data/%s populated from data/%s/%s (20 frames)",
            data_dir,
            data_dir,
            candidate,
        )
        return candidate
    return None


def build_full(
    sequence: int = DEFAULT_SEQUENCE,
    only: Optional[Iterable[str]] = None,
) -> Tuple[List[str], List[str]]:
    labels = load_labels(str(PROJECT_ROOT / "dataset" / "labels.json"))
    if only:
        wanted = set(only)
        labels = [lbl for lbl in labels if lbl.class_id in wanted or lbl.data_dir in wanted]

    ok: List[str] = []
    skipped: List[str] = []
    for lbl in labels:
        used = build_for_data_dir(lbl.data_dir, sequence=sequence)
        if used is None:
            skipped.append(lbl.class_id)
        else:
            ok.append(lbl.class_id)
    return ok, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Snapshot one complete sequence per gesture class into "
            "translation_data/ for text-to-gesture playback."
        )
    )
    parser.add_argument(
        "--sequence",
        type=int,
        default=DEFAULT_SEQUENCE,
        help=f"Preferred sequence index to copy (default: {DEFAULT_SEQUENCE}). "
        "If the preferred sequence is missing or incomplete, the next "
        "available sequence is used.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="ID",
        help="Limit to specific class_id or data_dir values.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    ok, skipped = build_full(sequence=args.sequence, only=args.only)
    print(f"Populated translation_data/ for {len(ok)} class(es).")
    if skipped:
        print(f"Skipped {len(skipped)} class(es) without complete data:")
        for cid in skipped:
            print(f"  - {cid}")


if __name__ == "__main__":
    main()
