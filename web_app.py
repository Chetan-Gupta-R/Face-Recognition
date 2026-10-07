from __future__ import annotations

import base64
import contextlib
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

import app

HOST = "127.0.0.1"
PORT = 8765
WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_REQUEST_BYTES = 5 * 1024 * 1024
SAMPLE_TARGET = app.SAMPLES_NEEDED
STATE_LOCK = threading.RLock()
MODEL_CACHE = {"stamp": None, "model": None, "names": [], "cascade": None}
LAST_LOGGED = {}


def _load_detector():
    if MODEL_CACHE["cascade"] is None:
        MODEL_CACHE["cascade"] = app.detector()
    return MODEL_CACHE["cascade"]


def _load_model():
    stamp = (app.MODEL_PATH.stat().st_mtime_ns, app.LABELS_PATH.stat().st_mtime_ns)
    if MODEL_CACHE["stamp"] != stamp:
        names = json.loads(app.LABELS_PATH.read_text(encoding="utf-8"))
        model = app.recognizer()
        model.read(str(app.MODEL_PATH))
        MODEL_CACHE.update(stamp=stamp, model=model, names=names)
    return MODEL_CACHE["model"], MODEL_CACHE["names"]


def _model_is_current() -> bool:
    if not app.MODEL_PATH.exists() or not app.LABELS_PATH.exists() or not app.FACES_DIR.exists():
        return False
    try:
        data_paths = [app.FACES_DIR, *app.FACES_DIR.rglob("*")]
        newest_data = max(path.stat().st_mtime_ns for path in data_paths)
        trained_at = min(app.MODEL_PATH.stat().st_mtime_ns, app.LABELS_PATH.stat().st_mtime_ns)
    except (OSError, ValueError):
        return False
    return trained_at >= newest_data


def _decode_image(payload: object) -> np.ndarray:
    if not isinstance(payload, str) or "," not in payload:
        raise ValueError("A camera frame is required.")
    encoded = payload.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("The camera frame could not be decoded.") from error
    if len(image_bytes) > MAX_REQUEST_BYTES:
        raise ValueError("The camera frame is too large.")
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The camera frame is not a supported image.")
    return image


def _status() -> dict[str, object]:
    people = app.get_enrolled_people(app.FACES_DIR)
    recent = []
    today_count = 0
    if app.RECOGNIZED_LOG_PATH.exists():
        lines = app.RECOGNIZED_LOG_PATH.read_text(encoding="utf-8").splitlines()
        today = time.strftime("%Y-%m-%d")
        today_count = sum(line.startswith(today) for line in lines)
        recent = lines[-8:][::-1]
    return {
        "people": [{"name": name, "samples": count} for name, count in people],
        "trained": _model_is_current(),
        "recent": recent,
        "todayCount": today_count,
        "sampleTarget": SAMPLE_TARGET,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "LocalFaceDashboard/1.0"

    def log_message(self, format_string: str, *args) -> None:
        return

    def _send(self, status: int, payload: object, content_type: str = "application/json; charset=utf-8") -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, object]:
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid request size.") from error
        if size <= 0 or size > MAX_REQUEST_BYTES:
            raise ValueError("The request is empty or too large.")
        try:
            payload = json.loads(self.rfile.read(size))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("The request body must be valid JSON.") from error
        if not isinstance(payload, dict):
            raise ValueError("The request body must be a JSON object.")
        return payload

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            with STATE_LOCK:
                self._send(200, _status())
            return

        files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        target = files.get(path)
        if target is None:
            self._send(404, {"error": "Not found."})
            return
        filename, content_type = target
        try:
            body = (WEB_DIR / filename).read_bytes()
        except FileNotFoundError:
            self._send(500, {"error": "Dashboard assets are missing."})
            return
        self._send(200, body, content_type)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{HOST}:{PORT}", f"http://localhost:{PORT}"}:
            self._send(403, {"error": "Requests must come from the local dashboard."})
            return
        try:
            payload = self._read_json()
            with STATE_LOCK:
                if path == "/api/train":
                    output = io.StringIO()
                    errors = io.StringIO()
                    try:
                        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                            app.train()
                    except SystemExit as error:
                        message = errors.getvalue().strip() or output.getvalue().strip() or "Training could not be completed."
                        self._send(400, {"error": message})
                        return
                    MODEL_CACHE["stamp"] = None
                    self._send(200, {"message": output.getvalue().strip() or "Model trained successfully."})
                    return

                if path == "/api/remove":
                    try:
                        name = app.safe_name(str(payload.get("name", "")))
                        app.remove_person(name)
                    except (SystemExit, OSError) as error:
                        self._send(400, {"error": str(error) or "Could not remove this person."})
                        return
                    MODEL_CACHE["stamp"] = None
                    self._send(200, {"message": f"Removed {name} and their saved samples."})
                    return

                if path == "/api/enroll":
                    try:
                        name = app.safe_name(str(payload.get("name", "")))
                        frame = _decode_image(payload.get("image"))
                    except (SystemExit, ValueError) as error:
                        self._send(400, {"error": str(error)})
                        return
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face = app.largest_face(gray, _load_detector())
                    if face is None:
                        self._send(200, {"saved": False, "count": 0, "target": SAMPLE_TARGET, "message": "Move into the camera frame."})
                        return
                    destination = app.FACES_DIR / name
                    destination.mkdir(parents=True, exist_ok=True)
                    existing = list(destination.glob("*.png"))
                    count = len(existing)
                    if count >= SAMPLE_TARGET:
                        self._send(200, {"saved": True, "count": count, "target": SAMPLE_TARGET, "complete": True, "message": "Enrollment samples are complete."})
                        return
                    x, y, width, height = face
                    crop = cv2.resize(gray[y:y + height, x:x + width], app.FACE_SIZE)
                    next_index = max((int(file.stem) for file in existing if file.stem.isdigit()), default=0) + 1
                    if not cv2.imwrite(str(destination / f"{next_index:03d}.png"), crop):
                        self._send(500, {"error": "Could not save this face sample."})
                        return
                    count += 1
                    self._send(200, {
                        "saved": True,
                        "count": count,
                        "target": SAMPLE_TARGET,
                        "complete": count >= SAMPLE_TARGET,
                        "message": f"Saved sample {count} of {SAMPLE_TARGET}.",
                    })
                    return

                if path == "/api/recognize":
                    if not _model_is_current():
                        self._send(400, {"error": "Train a model before starting recognition."})
                        return
                    try:
                        frame = _decode_image(payload.get("image"))
                        model, names = _load_model()
                        matches = app.recognize_frame(frame, model, names, _load_detector())
                    except (SystemExit, ValueError, OSError) as error:
                        self._send(400, {"error": str(error)})
                        return
                    now = time.monotonic()
                    for match in matches:
                        name = str(match["name"])
                        if name != "Unknown" and now - LAST_LOGGED.get(name, 0.0) >= 30:
                            app.record_recognition(name, float(match["distance"]))
                            LAST_LOGGED[name] = now
                    self._send(200, {"matches": matches})
                    return

            self._send(404, {"error": "Not found."})
        except ValueError as error:
            self._send(400, {"error": str(error)})
        except (OSError, cv2.error) as error:
            self._send(500, {"error": f"Local processing error: {error}"})


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)
    print(f"Face Recognition dashboard running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop the local server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
