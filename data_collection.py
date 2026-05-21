# %%

# Import necessary libraries
import argparse
import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
from my_functions import *
import logging
from typing import List

from label_registry import (
    LabelEntry,
    filter_labels,
    has_complete_data,
    labels_missing_data,
    load_labels,
)
from dataset.build_gesture_lexicon import update_lexicon_for_data_dir

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SignLanguageDataCollector:
    """Records gesture sequences under data/<data_dir>/ (not class_id)."""

    def __init__(self, labels: List[LabelEntry], sequences: int = 30, frames: int = 20):
        self.labels = labels
        self.sequences = sequences
        self.frames = frames
        self.PATH = os.path.join("data")
        self.setup_directories()

    def setup_directories(self) -> None:
        """Create necessary directories for data storage."""
        try:
            for entry, sequence in product(self.labels, range(self.sequences)):
                os.makedirs(
                    os.path.join(self.PATH, entry.data_dir, str(sequence)),
                    exist_ok=True,
                )
            logging.info("Directories created successfully")
        except Exception as e:
            logging.error(f"Error creating directories: {e}")
            raise

    def initialize_camera(self) -> cv2.VideoCapture:
        """Initialize camera with basic error handling"""
        logging.info("Initializing camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot access camera")
        return cap

    def collect_data(self) -> None:
        """Data collection loop with improved interface"""
        cap = self.initialize_camera()
        data_dirs_with_new_data: set[str] = set()

        try:
            with mp.solutions.holistic.Holistic(
                min_detection_confidence=0.75, min_tracking_confidence=0.75
            ) as holistic:
                for entry, sequence in product(self.labels, range(self.sequences)):
                    title = f"{entry.display_text} ({entry.class_id})"
                    print(f"\nRecording {title}, sequence {sequence + 1}/{self.sequences}")
                    print("Press SPACE to start recording, 'q' to quit")
                    print(
                        "You can use either your left or right hand - "
                        "the system will handle it correctly"
                    )

                    while True:
                        ret, image = cap.read()
                        if not ret:
                            continue

                        image = cv2.flip(image, 1)
                        results = image_process(image, holistic)
                        draw_landmarks(image, results)

                        cv2.putText(
                            image,
                            f"Sign: {entry.display_text}",
                            (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            image,
                            f"Sequence: {sequence + 1}/{self.sequences}",
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            image,
                            "Press SPACE to start",
                            (20, 400),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        cv2.imshow("Camera", image)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord(" "):
                            break
                        if key == ord("q"):
                            raise KeyboardInterrupt

                    print("Recording...")
                    for frame in range(self.frames):
                        ret, image = cap.read()
                        if not ret:
                            continue

                        image = cv2.flip(image, 1)
                        results = image_process(image, holistic)
                        draw_landmarks(image, results)

                        keypoints = keypoint_extraction(results)
                        frame_path = os.path.join(
                            self.PATH, entry.data_dir, str(sequence), str(frame)
                        )
                        np.save(frame_path, keypoints)

                        cv2.putText(
                            image,
                            f"Frame: {frame + 1}/{self.frames}",
                            (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.imshow("Camera", image)
                        cv2.waitKey(1)

                    print("Sequence completed")
                    data_dirs_with_new_data.add(entry.data_dir)

        except KeyboardInterrupt:
            print("\nData collection interrupted by user")
        except Exception as e:
            logging.error(f"Error during data collection: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            for data_dir in sorted(data_dirs_with_new_data):
                try:
                    if update_lexicon_for_data_dir(data_dir):
                        logging.info("Refreshed gesture lexicon for %s", data_dir)
                except Exception:
                    logging.exception(
                        "Failed to refresh gesture lexicon for %s", data_dir
                    )


def _print_label_table(labels: List[LabelEntry]) -> None:
    print(f"\n{'#':>4}  {'class_id':<16}  {'display':<14}  {'data_dir':<22}  status")
    print("-" * 72)
    for idx, entry in enumerate(labels, start=1):
        status = "ready" if has_complete_data(entry.data_dir) else "missing"
        print(
            f"{idx:>4}  {entry.class_id:<16}  {entry.display_text:<14}  "
            f"{entry.data_dir:<22}  {status}"
        )
    print()


def _letters_only(labels: List[LabelEntry]) -> List[LabelEntry]:
    return [e for e in labels if len(e.gloss) == 1]


def _interactive_select(all_labels: List[LabelEntry]) -> List[LabelEntry]:
    """Ask which gestures to record when no CLI filters were passed."""
    missing = labels_missing_data(all_labels)
    uk_letters = _letters_only(filter_labels(all_labels, language="uk"))
    en_letters = _letters_only(filter_labels(all_labels, language="en"))
    uk_missing_letters = [e for e in uk_letters if not has_complete_data(e.data_dir)]

    print("\n=== Gesture data collection ===")
    print("Choose what to record (you will NOT be asked to do everything by default).\n")
    print("  1) Ukrainian letters still missing data")
    print(f"     ({len(uk_missing_letters)} classes — good starting point)")
    print(f"  2) All Ukrainian letters ({len(uk_letters)} classes)")
    print(f"  3) English letters only ({len(en_letters)} classes)")
    print(f"  4) Everything still missing ({len(missing)} classes, any language)")
    print("  5) Pick language: en or uk")
    print("  6) Enter class_ids manually (comma-separated, e.g. uk.а,uk.б)")
    print("  7) List all classes and pick by number (e.g. 1,3,5-8)")
    print("  0) Quit")
    choice = input("\nChoice [1]: ").strip() or "1"

    if choice == "0":
        raise SystemExit(0)
    if choice == "1":
        return uk_missing_letters
    if choice == "2":
        return uk_letters
    if choice == "3":
        return en_letters
    if choice == "4":
        return missing
    if choice == "5":
        lang = input("Language (en/uk): ").strip().lower()
        if lang not in ("en", "uk"):
            raise SystemExit("Unknown language.")
        only_letters = input("Letters only? [y/N]: ").strip().lower() in ("y", "yes")
        picked = filter_labels(all_labels, language=lang)
        return _letters_only(picked) if only_letters else picked
    if choice == "6":
        raw = input("class_ids: ").strip()
        ids = [part.strip() for part in raw.split(",") if part.strip()]
        return filter_labels(all_labels, class_ids_filter=ids)
    if choice == "7":
        _print_label_table(all_labels)
        raw = input(
            "Enter numbers (comma-separated, ranges allowed e.g. 1,3,5-8): "
        ).strip()
        selected: List[LabelEntry] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                for num in range(int(start_s), int(end_s) + 1):
                    selected.append(all_labels[num - 1])
            else:
                selected.append(all_labels[int(part) - 1])
        return selected

    raise SystemExit(f"Unknown choice: {choice}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect sign language gesture data.",
        epilog=(
            "With no filters, an interactive menu opens so you pick what to record. "
            "Example: python data_collection.py --language uk --only-letters --missing-only"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--language",
        choices=["en", "uk"],
        help="Only labels for this language.",
    )
    parser.add_argument(
        "--only-letters",
        action="store_true",
        help="Only single-character glosses (letters), skip words.",
    )
    parser.add_argument(
        "--classes",
        metavar="IDS",
        help="Comma-separated class_ids (e.g. uk.а,uk.б,en.hello).",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Skip classes that already have full data (30 sequences x 20 frames).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all classes with ready/missing status and exit.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Non-interactive: record every label in labels.json (old behaviour).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt before recording.",
    )
    return parser.parse_args()


def _resolve_labels(args: argparse.Namespace) -> List[LabelEntry]:
    all_labels = load_labels()

    if args.list:
        _print_label_table(all_labels)
        raise SystemExit(0)

    used_cli_filter = any(
        [
            args.language,
            args.only_letters,
            args.classes,
            args.missing_only,
            args.all,
        ]
    )

    if args.all:
        labels = all_labels
    elif used_cli_filter:
        class_ids_filter = None
        if args.classes:
            class_ids_filter = [p.strip() for p in args.classes.split(",") if p.strip()]
        labels = filter_labels(all_labels, language=args.language, class_ids_filter=class_ids_filter)
        if args.only_letters:
            labels = _letters_only(labels)
        if args.missing_only:
            labels = [e for e in labels if not has_complete_data(e.data_dir)]
    else:
        labels = _interactive_select(all_labels)

    if not labels:
        raise SystemExit("No labels matched; nothing to collect.")
    return labels


if __name__ == "__main__":
    args = _parse_args()
    labels = _resolve_labels(args)

    sequences_total = len(labels) * 30
    print(f"\nWill record {len(labels)} class(es) x 30 sequences = {sequences_total} sequences.")
    print("Press SPACE before each sequence; press 'q' during preview to stop early.\n")
    for entry in labels:
        status = "re-record" if has_complete_data(entry.data_dir) else "new"
        print(f"  - {entry.display_text} ({entry.class_id}) [{status}] -> data/{entry.data_dir}/")

    if not args.yes:
        confirm = input("\nStart recording? [Y/n]: ").strip().lower()
        if confirm in ("n", "no"):
            raise SystemExit("Cancelled.")

    collector = SignLanguageDataCollector(labels)
    collector.collect_data()
