from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
MODEL_PATH = DATA_DIR / "trainer.yml"
LABELS_PATH = DATA_DIR / "labels.json"
RECOGNIZED_LOG_PATH = DATA_DIR / "recognized_faces.txt"
CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
FACE_SIZE = (200, 200)
SAMPLES_NEEDED = 30

def fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    if not cleaned:
        fail("Please provide a name containing letters or numbers.")
    return cleaned


def get_enrolled_people(faces_dir: Path = FACES_DIR) -> list[tuple[str, int]]:
    if not faces_dir.exists():
        return []
    people: list[tuple[str, int]] = []
    for person_dir in sorted(path for path in faces_dir.iterdir() if path.is_dir()):
        samples = sorted(person_dir.glob("*.*"))
        if samples:
            people.append((person_dir.name, len(samples)))
    return people


def detector() -> cv2.CascadeClassifier:
    cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if cascade.empty():
        fail("OpenCV's face detector could not be loaded.")
    return cascade


def recognizer():
    if not hasattr(cv2, "face"):
        fail("opencv-contrib-python is required. Install dependencies from requirements.txt.")
    return cv2.face.LBPHFaceRecognizer_create()


def open_camera() -> cv2.VideoCapture:
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        fail("No webcam was found or it is in use by another application.")
    return camera


def largest_face(gray: np.ndarray, cascade: cv2.CascadeClassifier):
    faces = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(100, 100), maxSize=(700, 700))
    return max(faces, key=lambda face: face[2] * face[3]) if len(faces) else None


def list_people() -> None:
    people = get_enrolled_people(FACES_DIR)
    if not people:
        print("No enrolled people found.")
        return
    print("Enrolled people:")
    for name, count in people:
        print(f" - {name}: {count} sample(s)")


def remove_person(raw_name: str) -> None:
    name = safe_name(raw_name)
    target = FACES_DIR / name
    if not target.exists():
        fail(f"No enrolled face found for '{name}'.")
    for item in sorted(target.iterdir(), reverse=True):
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            for sub_item in sorted(item.rglob("*"), reverse=True):
                if sub_item.is_file() or sub_item.is_symlink():
                    sub_item.unlink()
                elif sub_item.is_dir():
                    sub_item.rmdir()
            item.rmdir()
    target.rmdir()
    print(f"Removed person '{name}'.")


def record_recognition(name: str, distance: float) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} | {name} | distance={distance:.1f}\n"
    with RECOGNIZED_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(line)


def enroll(raw_name: str) -> None:
    name = safe_name(raw_name)
    destination = FACES_DIR / name
    destination.mkdir(parents=True, exist_ok=True)
    cascade, camera = detector(), open_camera()
    saved, last_saved = 0, 0.0

    print("Look at the camera and slowly turn your head. Press Q to cancel.")
    try:
        while saved < SAMPLES_NEEDED:
            ok, frame = camera.read()
            if not ok:
                fail("Could not read from the webcam.")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = largest_face(gray, cascade)
            if face is not None:
                x, y, w, h = face
                crop = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
                now = time.monotonic()
                if now - last_saved >= 0.20:
                    saved += 1
                    last_saved = now
                    cv2.imwrite(str(destination / f"{saved:03d}.png"), crop)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(frame, f"Samples: {saved}/{SAMPLES_NEEDED}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
            cv2.putText(frame, "Q: cancel", (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Enroll face", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                print("Enrollment cancelled.")
                return
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"Saved {saved} samples for {name}. Run: python face_recognition_app.py train")


def train() -> None:
    images, labels, names = [], [], []
    if not FACES_DIR.exists():
        fail("No enrolled faces yet. Run enroll first.")

    enrolled_people = get_enrolled_people(FACES_DIR)
    if not enrolled_people:
        fail("No enrolled faces yet. Run enroll first.")

    for label, (person_name, _) in enumerate(enrolled_people):
        person_dir = FACES_DIR / person_name
        files = sorted(person_dir.glob("*.png")) + sorted(person_dir.glob("*.jpg")) + sorted(person_dir.glob("*.jpeg"))
        if not files:
            continue
        names.append(person_name)
        for file in sorted(files):
            image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(image)
                labels.append(label)

    if not images:
        fail("No usable face samples found. Enroll a face again.")

    DATA_DIR.mkdir(exist_ok=True)
    model = recognizer()
    model.train(images, np.array(labels, dtype=np.int32))
    model.write(str(MODEL_PATH))
    LABELS_PATH.write_text(json.dumps(names, indent=2), encoding="utf-8")
    print(f"Training complete: {len(images)} samples for {len(names)} person(s).")


def recognize() -> None:
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        fail("No trained model. Run train after enrolling faces.")
    names = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    if not names:
        fail("No trained model labels found. Run train after enrolling faces.")

    model, cascade, camera = recognizer(), detector(), open_camera()
    model.read(str(MODEL_PATH))
    print("Recognition started. Press Q to close.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                fail("Could not read from the webcam.")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for x, y, w, h in cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(100, 100)):
                crop = cv2.resize(gray[y:y + h, x:x + w], FACE_SIZE)
                label, distance = model.predict(crop)
                known = 0 <= label < len(names) and distance < 65
                if known:
                    recognized_name = names[label]
                    record_recognition(recognized_name, distance)
                    text = f"{recognized_name} ({distance:.0f})"
                    color = (0, 200, 0)
                else:
                    text = "Unknown"
                    color = (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, "Q: close", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Face recognition", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local webcam face-recognition demo")
    commands = parser.add_subparsers(dest="command", required=True)

    enroll_parser = commands.add_parser("enroll", help="Capture face samples")
    enroll_parser.add_argument("name", help="Name to associate with this face")

    commands.add_parser("train", help="Train the local recognition model")
    commands.add_parser("recognize", help="Start live recognition")
    commands.add_parser("list", help="Show the enrolled identities and sample counts")

    remove_parser = commands.add_parser("remove", help="Remove a person and their samples")
    remove_parser.add_argument("name", help="Name of the person to remove")

    args = parser.parse_args()
    actions = {
        "enroll": lambda: enroll(args.name),
        "train": train,
        "recognize": recognize,
        "list": list_people,
        "remove": lambda: remove_person(args.name),
    }
    actions[args.command]()


if __name__ == "__main__":
    main()
