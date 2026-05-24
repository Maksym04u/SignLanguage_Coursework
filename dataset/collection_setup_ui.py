"""Minimal Tkinter UI to choose language + gestures before data collection."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, List, Optional

from label_registry import LabelEntry

from dataset.label_builder import build_labels_from_symbols, parse_symbols


class CollectionSetupWindow:
    """Small form: language + comma/newline-separated symbols -> preview -> OK."""

    def __init__(self, on_confirm: Callable[[List[LabelEntry], str], None]) -> None:
        self._on_confirm = on_confirm
        self._result: Optional[List[LabelEntry]] = None

        self.root = tk.Tk()
        self._language = tk.StringVar(master=self.root, value="uk")
        self.root.title("Gesture collection setup")
        self.root.minsize(480, 420)
        self.root.resizable(True, True)

        pad = {"padx": 12, "pady": 6}

        header = ttk.Label(
            self.root,
            text="Enter the gesture(s) you want to record, then press OK.",
            wraplength=440,
        )
        header.pack(anchor="w", **pad)

        lang_row = ttk.Frame(self.root)
        lang_row.pack(fill="x", **pad)
        ttk.Label(lang_row, text="Language:").pack(side="left")
        ttk.Radiobutton(lang_row, text="English (en)", variable=self._language, value="en").pack(
            side="left", padx=(8, 4)
        )
        ttk.Radiobutton(lang_row, text="Ukrainian (uk)", variable=self._language, value="uk").pack(
            side="left", padx=4
        )

        ttk.Label(
            self.root,
            text="Gestures (comma-separated and/or one per line):",
        ).pack(anchor="w", padx=12)

        text_frame = ttk.Frame(self.root)
        text_frame.pack(fill="both", expand=True, padx=12, pady=(0, 6))

        self.symbols_text = tk.Text(text_frame, height=8, wrap="word", font=("Segoe UI", 11))
        scroll = ttk.Scrollbar(text_frame, command=self.symbols_text.yview)
        self.symbols_text.configure(yscrollcommand=scroll.set)
        self.symbols_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        hint = (
            "Examples:\n"
            "  Letters: а, б, в   or one per line\n"
            "  Words: hello, I'm good, So-so"
        )
        ttk.Label(self.root, text=hint, foreground="#555").pack(anchor="w", padx=12)

        ttk.Label(self.root, text="Preview (class_id → data_dir):").pack(anchor="w", padx=12, pady=(8, 0))

        preview_frame = ttk.Frame(self.root)
        preview_frame.pack(fill="both", expand=True, padx=12, pady=(0, 6))

        self.preview = tk.Listbox(preview_frame, height=6, font=("Consolas", 10))
        preview_scroll = ttk.Scrollbar(preview_frame, command=self.preview.yview)
        self.preview.configure(yscrollcommand=preview_scroll.set)
        self.preview.pack(side="left", fill="both", expand=True)
        preview_scroll.pack(side="right", fill="y")

        btn_row = ttk.Frame(self.root)
        btn_row.pack(fill="x", padx=12, pady=(4, 12))
        ttk.Button(btn_row, text="Update preview", command=self._refresh_preview).pack(side="left")
        ttk.Button(btn_row, text="Cancel", command=self.root.destroy).pack(side="right", padx=(4, 0))
        ttk.Button(btn_row, text="OK — create & record", command=self._on_ok).pack(side="right", padx=(4, 0))

        self.symbols_text.insert("1.0", "а, б, в")
        self._refresh_preview()
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    def _refresh_preview(self) -> None:
        self.preview.delete(0, tk.END)
        symbols = parse_symbols(self.symbols_text.get("1.0", tk.END))
        if not symbols:
            self.preview.insert(tk.END, "(no symbols parsed)")
            return
        lang = self._language.get()
        try:
            for entry in build_labels_from_symbols(symbols, lang):
                kind = "letter" if len(entry.gloss) == 1 else "word"
                line = f"{entry.display_text}  →  {entry.class_id}  →  {entry.data_dir} ({kind})"
                self.preview.insert(tk.END, line)
        except ValueError as exc:
            self.preview.insert(tk.END, f"Error: {exc}")

    def _on_ok(self) -> None:
        symbols = parse_symbols(self.symbols_text.get("1.0", tk.END))
        if not symbols:
            messagebox.showwarning("No gestures", "Enter at least one gesture symbol.")
            return
        lang = self._language.get()
        try:
            entries = build_labels_from_symbols(symbols, lang)
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self._result = entries
        self.root.destroy()
        self._on_confirm(entries, lang)

    def run(self) -> Optional[List[LabelEntry]]:
        self.root.mainloop()
        return self._result


def run_setup_dialog(
    on_confirm: Callable[[List[LabelEntry], str], None],
) -> Optional[List[LabelEntry]]:
    win = CollectionSetupWindow(on_confirm)
    return win.run()
