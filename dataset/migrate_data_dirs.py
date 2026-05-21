"""One-time migration: rename legacy data/ folders to the new naming scheme.

Run from project root:
    python -m dataset.migrate_data_dirs
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

# old folder name -> new folder name (only folders that exist on disk)
RENAME_MAP = {
    # English letters
    "a": "en_letter_a",
    "b": "en_letter_b",
    "c": "en_letter_c",
    "d": "en_letter_d",
    "e": "en_letter_e",
    "f": "en_letter_f",
    "g": "en_letter_g",
    "h": "en_letter_h",
    "i": "en_letter_i",
    "j": "en_letter_j",
    "k": "en_letter_k",
    "l": "en_letter_l",
    "m": "en_letter_m",
    "n": "en_letter_n",
    "o": "en_letter_o",
    "p": "en_letter_p",
    "q": "en_letter_q",
    "r": "en_letter_r",
    "s": "en_letter_s",
    "t": "en_letter_t",
    "u": "en_letter_u",
    "v": "en_letter_v",
    "w": "en_letter_w",
    "x": "en_letter_x",
    "y": "en_letter_y",
    "z": "en_letter_z",
    # English words
    "Age": "en_word_age",
    "Bad": "en_word_bad",
    "Busy": "en_word_busy",
    "Fine": "en_word_fine",
    "Great": "en_word_great",
    "Hello": "en_word_hello",
    "I'm good": "en_word_im_good",
    "Name": "en_word_name",
    "Nothing": "en_word_nothing",
    "So-so": "en_word_so_so",
    "What": "en_word_what",
    "What's up": "en_word_whats_up",
    "Where": "en_word_where",
    "Work": "en_word_work",
}


def main() -> None:
    if not DATA_ROOT.exists():
        print(f"No data folder at {DATA_ROOT}")
        sys.exit(1)

    renamed = 0
    skipped = 0
    for old_name, new_name in RENAME_MAP.items():
        src = DATA_ROOT / old_name
        dst = DATA_ROOT / new_name
        if not src.exists():
            skipped += 1
            continue
        if dst.exists():
            print(f"SKIP (target exists): {old_name} -> {new_name}")
            skipped += 1
            continue
        shutil.move(str(src), str(dst))
        print(f"OK: {old_name} -> {new_name}")
        renamed += 1

    print(f"Done. Renamed {renamed}, skipped {skipped}.")


if __name__ == "__main__":
    main()
