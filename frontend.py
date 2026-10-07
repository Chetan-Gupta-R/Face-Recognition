from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

BASE_DIR = Path(__file__).resolve().parent
APP_PATH = BASE_DIR / "face_recognition_app.py"
if not APP_PATH.exists():
    APP_PATH = BASE_DIR / "app.py"


class FaceRecognitionFrontend(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Face Recognition Frontend")
        self.geometry("520x420")
        self.minsize(420, 320)

        self.name_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")

        title = tk.Label(self, text="Face Recognition Demo", font=("Segoe UI", 18, "bold"))
        title.pack(pady=(18, 8))

        form = ttk.Frame(self, padding=12)
        form.pack(fill="x")

        ttk.Label(form, text="Person name:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=6)
        name_entry = ttk.Entry(form, textvariable=self.name_var, width=28)
        name_entry.grid(row=0, column=1, sticky="ew", pady=6)
        name_entry.focus_set()

        actions = [
            ("Enroll", "enroll"),
            ("Train", "train"),
            ("Recognize", "recognize"),
            ("List", "list"),
            ("Remove", "remove"),
        ]

        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10)

        for label, action in actions:
            ttk.Button(button_frame, text=label, command=lambda action=action: self.run_action(action)).pack(side="left", padx=6)

        status_label = ttk.Label(self, textvariable=self.status_var, foreground="#0b5d2d")
        status_label.pack(pady=(8, 6))

        self.output = tk.Text(self, height=12, wrap="word", state="disabled", bg="#f7f7f7")
        self.output.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=(0, 12))

    def run_action(self, action: str) -> None:
        name = self.name_var.get().strip()
        args = [sys.executable, str(APP_PATH)]
        if action in {"enroll", "remove"}:
            if not name:
                messagebox.showwarning("Missing name", "Please enter a person's name before continuing.")
                return
            args.extend([action, name])
        else:
            args.append(action)

        self.status_var.set(f"Running: {action}")
        try:
            result = subprocess.run(args, capture_output=True, text=True)
            output = (result.stdout or "") + (result.stderr or "")
            self.output.config(state="normal")
            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, output.strip() or f"{action.title()} completed.")
            self.output.config(state="disabled")

            if result.returncode == 0:
                self.status_var.set("Completed successfully")
            else:
                self.status_var.set(f"Failed: exit code {result.returncode}")
                messagebox.showerror("Command failed", output.strip() or "The action did not complete successfully.")
        except FileNotFoundError:
            self.status_var.set("Error: app not found")
            messagebox.showerror("Missing script", "The face recognition app script could not be found.")


if __name__ == "__main__":
    app = FaceRecognitionFrontend()
    app.mainloop()
