from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def resolve_app_path() -> Path:
    candidates = [BASE_DIR / "face_recognition_app.py", BASE_DIR / "app.py"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No supported app entry point was found.")


def build_command(action: str, value: str | None = None) -> list[str]:
    script_name = resolve_app_path().name
    command = [script_name, action]
    if value:
        command.append(value)
    return command


def run_command(action: str, value: str | None = None) -> subprocess.CompletedProcess[str]:
    app_path = resolve_app_path()
    args = [sys.executable, str(app_path), action]
    if value:
        args.append(value)
    return subprocess.run(args, capture_output=True, text=True)


def main() -> None:
    print("Face Recognition Frontend")
    print("1. Enroll a person")
    print("2. Train the model")
    print("3. Recognize faces")
    print("4. List enrolled people")
    print("5. Remove a person")
    print("6. Exit")

    while True:
        choice = input("\nChoose an option (1-6): ").strip()

        if choice == "1":
            name = input("Enter person name: ").strip()
            if not name:
                print("Name cannot be empty.")
                continue
            result = run_command("enroll", name)
        elif choice == "2":
            result = run_command("train")
        elif choice == "3":
            result = run_command("recognize")
        elif choice == "4":
            result = run_command("list")
        elif choice == "5":
            name = input("Enter person name to remove: ").strip()
            if not name:
                print("Name cannot be empty.")
                continue
            result = run_command("remove", name)
        elif choice == "6":
            print("Goodbye.")
            break
        else:
            print("Invalid choice.")
            continue

        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
        if result.returncode != 0:
            print(f"Command failed with exit code {result.returncode}.")


if __name__ == "__main__":
    main()
