import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import face_recognition_app as app
import web_app


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

    def test_recognize_frame_returns_match_and_unknown_faces(self):
        import numpy as np

        cascade = Mock()
        cascade.detectMultiScale.return_value = [(1, 2, 120, 120), (130, 2, 120, 120)]
        model = Mock()
        model.predict.side_effect = [(0, 20.0), (4, 70.0)]
        frame = np.zeros((240, 260, 3), dtype=np.uint8)

        with patch.object(app, "FACE_SIZE", (20, 20)):
            matches = app.recognize_frame(frame, model, ["Alice"], cascade)

        self.assertEqual([item["name"] for item in matches], ["Alice", "Unknown"])
        self.assertEqual(matches[0]["box"], [1, 2, 120, 120])
        self.assertEqual(matches[1]["box"], [130, 2, 120, 120])

    def test_model_status_becomes_stale_after_face_data_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            faces_dir = root / "faces"
            person_dir = faces_dir / "Alice"
            person_dir.mkdir(parents=True)
            sample = person_dir / "001.png"
            model_path = root / "trainer.yml"
            labels_path = root / "labels.json"
            sample.write_bytes(b"sample")
            model_path.write_bytes(b"model")
            labels_path.write_text('["Alice"]', encoding="utf-8")

            with patch.object(web_app.app, "FACES_DIR", faces_dir), patch.object(web_app.app, "MODEL_PATH", model_path), patch.object(web_app.app, "LABELS_PATH", labels_path):
                self.assertTrue(web_app._model_is_current())
                future = labels_path.stat().st_mtime_ns + 1_000_000
                sample.touch()
                os.utime(sample, ns=(future, future))
                self.assertFalse(web_app._model_is_current())

    def test_build_gui_command_returns_expected_args(self):
        import importlib.util

        module_path = Path(__file__).resolve().parents[1] / "gui_app.py"
        self.assertTrue(module_path.exists())

        spec = importlib.util.spec_from_file_location("gui_app", module_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.assertEqual(module.build_command("enroll", "Alice"), ["face_recognition_app.py", "enroll", "Alice"])
        self.assertEqual(module.build_command("list", None), ["face_recognition_app.py", "list"])
