# %%

"""Record sign-language gesture sequences via webcam + MediaPipe."""

import argparse
import logging
import os
import time
from typing import List

import cv2
import mediapipe as mp
import numpy as np
from itertools import product

from my_functions import *
from label_registry import (
    LabelEntry,
    filter_labels,
    has_complete_data,
    labels_missing_data,
    load_labels,
)
from dataset.build_gesture_lexicon import update_lexicon_for_data_dir
from dataset.build_translation_data import write_raw_playback
from dataset.collection_setup_ui import run_setup_dialog
from dataset.label_builder import (
    build_labels_from_symbols,
    parse_symbols,
    register_label_entries,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Automatic pauses before each recording (replaces pressing SPACE).
PAUSE_BETWEEN_SEQUENCES_SEC = 1.0
PAUSE_BETWEEN_WORDS_SEC = 12.0


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
        logging.info("Initializing camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot access camera")
        return cap

    def _draw_preview_overlay(
        self,
        image,
        entry: LabelEntry,
        sequence: int,
        footer: str,
        countdown_sec: float | None = None,
    ) -> None:
        put_ui_text(
            image,
            f"Sign: {entry.display_text}",
            (20, 20),
            font_scale=0.7,
            color_bgr=(0, 0, 255),
        )
        put_ui_text(
            image,
            f"Sequence: {sequence + 1}/{self.sequences}",
            (20, 55),
            font_scale=0.5,
            color_bgr=(0, 0, 255),
        )
        put_ui_text(
            image,
            footer,
            (20, 380),
            font_scale=0.85,
            color_bgr=(0, 0, 255),
        )
        if countdown_sec is not None and countdown_sec > 0:
            put_ui_text(
                image,
                f"{int(countdown_sec + 0.999)}",
                (20, 430),
                font_scale=1.4,
                color_bgr=(0, 0, 255),
            )
        put_ui_text(
            image,
            "Press 'q' to quit",
            (20, 480),
            font_scale=0.55,
            color_bgr=(0, 0, 255),
        )

    def _countdown(
        self,
        cap: cv2.VideoCapture,
        holistic,
        entry: LabelEntry,
        sequence: int,
        seconds: float,
        message: str,
    ) -> None:
        """Show live preview while counting down before a recording starts."""
        deadline = time.monotonic() + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            ret, image = cap.read()
            if not ret:
                continue

            image = cv2.flip(image, 1)
            results = image_process(image, holistic)
            draw_landmarks(image, results)
            self._draw_preview_overlay(
                image, entry, sequence, message, countdown_sec=remaining
            )

            cv2.imshow("Camera", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt

    def collect_data(self) -> None:
        cap = self.initialize_camera()
        data_dirs_with_new_data: set[str] = set()

        try:
            with mp.solutions.holistic.Holistic(
                min_detection_confidence=0.75, min_tracking_confidence=0.75
            ) as holistic:
                for entry in self.labels:
                    for sequence in range(self.sequences):
                        title = f"{entry.display_text} ({entry.class_id})"
                        print(
                            f"\nRecording {title}, "
                            f"sequence {sequence + 1}/{self.sequences}"
                        )

                        if sequence == 0:
                            pause = PAUSE_BETWEEN_WORDS_SEC
                            message = (
                                f"Get ready — recording starts in {int(pause)}s "
                                f"(new sign)"
                            )
                            print(
                                f"Waiting {int(pause)}s before first sequence "
                                f"of this sign..."
                            )
                        else:
                            pause = PAUSE_BETWEEN_SEQUENCES_SEC
                            message = (
                                f"Next sequence in {int(pause)}s — hold the sign"
                            )
                            print(f"Waiting {int(pause)}s before next sequence...")

                        self._countdown(cap, holistic, entry, sequence, pause, message)

                        print("Recording...")
                        raw_sequence_frames: List[np.ndarray] = []
                        for frame in range(self.frames):
                            ret, image = cap.read()
                            if not ret:
                                continue

                            image = cv2.flip(image, 1)
                            results = image_process(image, holistic)
                            draw_landmarks(image, results)

                            keypoints = keypoint_extraction(results)
                            raw_keypoints = raw_keypoint_extraction(results)
                            frame_path = os.path.join(
                                self.PATH, entry.data_dir, str(sequence), str(frame)
                            )
                            np.save(frame_path, keypoints)
                            raw_sequence_frames.append(raw_keypoints)

                            put_ui_text(
                                image,
                                f"Sign: {entry.display_text}",
                                (20, 20),
                                font_scale=0.7,
                                color_bgr=(0, 0, 255),
                            )
                            put_ui_text(
                                image,
                                f"Frame: {frame + 1}/{self.frames}",
                                (20, 55),
                                font_scale=0.5,
                                color_bgr=(0, 0, 255),
                            )
                            cv2.imshow("Camera", image)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord("q"):
                                raise KeyboardInterrupt

                        if write_raw_playback(
                            entry.data_dir, raw_sequence_frames, sequence
                        ):
                            logging.info(
                                "Updated translation_data/%s from sequence %s "
                                "(raw playback)",
                                entry.data_dir,
                                sequence,
                            )
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
    missing = labels_missing_data(all_labels)
    uk_letters = _letters_only(filter_labels(all_labels, language="uk"))
    en_letters = _letters_only(filter_labels(all_labels, language="en"))
    uk_missing_letters = [e for e in uk_letters if not has_complete_data(e.data_dir)]

    print("\n=== Pick existing labels from labels.json ===\n")
    print("  1) Ukrainian letters still missing data")
    print(f"     ({len(uk_missing_letters)} classes)")
    print(f"  2) All Ukrainian letters ({len(uk_letters)} classes)")
    print(f"  3) English letters only ({len(en_letters)} classes)")
    print(f"  4) Everything still missing ({len(missing)} classes)")
    print("  5) Pick language: en or uk")
    print("  6) Enter class_ids manually")
    print("  7) List all classes and pick by number")
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
        raw = input("Enter numbers (e.g. 1,3,5-8): ").strip()
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
            "Default: opens a setup window to enter new gesture symbols.\n"
            "Examples:\n"
            "  python data_collection.py\n"
            "  python data_collection.py --symbols \"а,б,в\" --language uk -y\n"
            "  python data_collection.py --pick-existing --language uk --missing-only\n"
            "  python data_collection.py --setup-only --symbols \"hello\" --language en"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--language", choices=["en", "uk"], help="Language for new symbols.")
    parser.add_argument(
        "--symbols",
        metavar="TEXT",
        help='Gesture symbols (comma/newline separated), e.g. "а,б,в" or "hello, I\'m good".',
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only update labels.json and create folders; do not open the camera.",
    )
    parser.add_argument(
        "--pick-existing",
        action="store_true",
        help="Pick gestures already listed in labels.json (terminal menu).",
    )
    parser.add_argument("--only-letters", action="store_true", help="With --pick-existing.")
    parser.add_argument("--classes", metavar="IDS", help="With --pick-existing: class_ids.")
    parser.add_argument("--missing-only", action="store_true", help="With --pick-existing.")
    parser.add_argument("--list", action="store_true", help="List labels and exit.")
    parser.add_argument("--all", action="store_true", help="With --pick-existing: all labels.")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts.")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip setup window (requires --symbols or --pick-existing).",
    )
    return parser.parse_args()


def _resolve_existing_labels(args: argparse.Namespace) -> List[LabelEntry]:
    all_labels = load_labels()
    if args.all:
        labels = all_labels
    else:
        class_ids_filter = None
        if args.classes:
            class_ids_filter = [p.strip() for p in args.classes.split(",") if p.strip()]
        labels = filter_labels(
            all_labels, language=args.language, class_ids_filter=class_ids_filter
        )
        if args.only_letters:
            labels = _letters_only(labels)
        if args.missing_only:
            labels = [e for e in labels if not has_complete_data(e.data_dir)]
    if not labels:
        raise SystemExit("No labels matched; nothing to collect.")
    return labels


def _print_plan(labels: List[LabelEntry]) -> None:
    sequences_total = len(labels) * 30
    print(f"\nWill record {len(labels)} class(es) x 30 sequences = {sequences_total} sequences.")
    print(
        f"Automatic pauses: {int(PAUSE_BETWEEN_SEQUENCES_SEC)}s between sequences, "
        f"{int(PAUSE_BETWEEN_WORDS_SEC)}s before each new sign. Press 'q' to stop.\n"
    )
    for entry in labels:
        status = "re-record" if has_complete_data(entry.data_dir) else "new"
        print(
            f"  - {entry.display_text} ({entry.class_id}) [{status}] "
            f"-> data/{entry.data_dir}/"
        )


def _confirm_start(args: argparse.Namespace) -> None:
    if not args.yes:
        confirm = input("\nStart recording? [Y/n]: ").strip().lower()
        if confirm in ("n", "no"):
            raise SystemExit("Cancelled.")


def _register_and_report(entries: List[LabelEntry]) -> List[LabelEntry]:
    _, added = register_label_entries(entries)
    if added:
        print(f"\nAdded {len(added)} new label(s) to dataset/labels.json:")
        for e in added:
            print(f"  + {e.display_text} → {e.class_id} → {e.data_dir}")
    else:
        print("\nAll gesture(s) already exist in labels.json; data folders ensured.")
    for entry in entries:
        print(f"  data/{entry.data_dir}/")
    return entries


def _start_recording(labels: List[LabelEntry], args: argparse.Namespace) -> None:
    _print_plan(labels)
    _confirm_start(args)
    SignLanguageDataCollector(labels).collect_data()


def _run_new_symbols_flow(args: argparse.Namespace) -> None:
    if not args.language:
        raise SystemExit("--language en|uk is required with --symbols or --no-gui")
    if not args.symbols:
        raise SystemExit("--symbols is required with --no-gui")

    symbols = parse_symbols(args.symbols)
    if not symbols:
        raise SystemExit("No symbols parsed from --symbols")

    entries = build_labels_from_symbols(symbols, args.language)
    _register_and_report(entries)

    if args.setup_only:
        print("\nSetup complete (--setup-only). Run again without --setup-only to record.")
        return

    _start_recording(entries, args)


def _run_gui_flow(args: argparse.Namespace) -> None:
    recorded: List[LabelEntry] = []

    def on_confirm(entries: List[LabelEntry], _language: str) -> None:
        nonlocal recorded
        _register_and_report(entries)
        recorded = entries
        if args.setup_only:
            print("\nSetup complete. Re-run without --setup-only to record.")
            return
        _start_recording(entries, args)

    run_setup_dialog(on_confirm)
    if not recorded and not args.setup_only:
        print("Setup closed without recording.")


def main() -> None:
    args = _parse_args()

    if args.list:
        _print_label_table(load_labels())
        return

    if args.pick_existing:
        labels = _interactive_select(load_labels()) if not (
            args.all or args.language or args.classes or args.missing_only
        ) else _resolve_existing_labels(args)
        _start_recording(labels, args)
        return

    if args.symbols or args.no_gui:
        _run_new_symbols_flow(args)
        return

    _run_gui_flow(args)


if __name__ == "__main__":
    main()
