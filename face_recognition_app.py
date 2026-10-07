import app as _app


def _sync_module_state() -> None:
    for name in [
        "BASE_DIR",
        "DATA_DIR",
        "FACES_DIR",
        "MODEL_PATH",
        "LABELS_PATH",
        "RECOGNIZED_LOG_PATH",
        "CASCADE_PATH",
        "FACE_SIZE",
        "SAMPLES_NEEDED",
    ]:
        if hasattr(_app, name):
            globals()[name] = getattr(_app, name)


_sync_module_state()


def safe_name(name: str) -> str:
    return _app.safe_name(name)


def get_enrolled_people(faces_dir):
    return _app.get_enrolled_people(faces_dir)


def detector():
    return _app.detector()


def recognizer():
    return _app.recognizer()


def open_camera():
    return _app.open_camera()


def largest_face(gray, cascade):
    return _app.largest_face(gray, cascade)


def recognize_frame(frame, model, names, cascade):
    return _app.recognize_frame(frame, model, names, cascade)


def list_people() -> None:
    return _app.list_people()


def remove_person(raw_name: str) -> None:
    return _app.remove_person(raw_name)


def record_recognition(name: str, distance: float) -> None:
    if "RECOGNIZED_LOG_PATH" in globals():
        _app.RECOGNIZED_LOG_PATH = globals()["RECOGNIZED_LOG_PATH"]
    return _app.record_recognition(name, distance)


def enroll(raw_name: str) -> None:
    return _app.enroll(raw_name)


def train() -> None:
    return _app.train()


def recognize() -> None:
    return _app.recognize()


def fail(message: str) -> None:
    return _app.fail(message)


def main() -> None:
    return _app.main()


if __name__ == "__main__":
    main()
