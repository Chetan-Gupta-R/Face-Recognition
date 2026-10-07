import tempfile
import unittest
from pathlib import Path

import face_recognition_app as app


class FaceRecognitionAppTests(unittest.TestCase):
    def test_safe_name_sanitizes_names(self):
        self.assertEqual(app.safe_name("  Alice Smith!  "), "Alice_Smith_")

    def test_get_enrolled_people_counts_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            faces_dir = Path(temp_dir) / "faces"
            person_a = faces_dir / "Alice"
            person_b = faces_dir / "Bob"
            person_a.mkdir(parents=True)
            person_b.mkdir(parents=True)
            (person_a / "001.png").write_bytes(b"x")
            (person_a / "002.png").write_bytes(b"x")
            (person_b / "001.png").write_bytes(b"x")

            people = app.get_enrolled_people(faces_dir)

            self.assertEqual(people, [("Alice", 2), ("Bob", 1)])

    def test_record_recognition_writes_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "recognized_faces.txt"
            original_path = app.RECOGNIZED_LOG_PATH
            app.RECOGNIZED_LOG_PATH = log_path
            try:
                app.record_recognition("Alice", 12.5)
                self.assertIn("Alice", log_path.read_text(encoding="utf-8"))
            finally:
                app.RECOGNIZED_LOG_PATH = original_path
