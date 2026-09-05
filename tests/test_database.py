"""
Unit tests for DatabaseManager and session persistence.
"""

import unittest
import tempfile
from pathlib import Path
from visionvoice.memory.database import DatabaseManager
from visionvoice.core.models import ReadingSession, OCRResult


class TestDatabaseManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_visionvoice.db"
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_session_creation_and_update(self):
        session = ReadingSession(
            session_id="test_sess_01",
            book_id="book_abc",
            current_page_number=1,
            preferred_language="ta",
            read_mode="translated",
        )
        self.db.create_or_update_session(session)

        # Update page number
        session.current_page_number = 2
        self.db.create_or_update_session(session)

        # Check duplicate hash (should be false before recording)
        self.assertFalse(self.db.is_duplicate_page("test_sess_01", "dummy_hash_123"))

    def test_record_page_and_duplicate_detection(self):
        ocr = OCRResult(
            text="Hello world from VisionVoice.",
            cleaned_text="Hello world from VisionVoice.",
            sentences=["Hello world from VisionVoice."],
            detected_language="eng",
            detected_script="English",
            composite_score=88.5,
            average_confidence=92.0,
            selected_variant="clahe",
            selected_psm=4,
            image_hash="abc123hash",
            original_image_path="/path/orig.jpg",
            processed_image_path="/path/proc.jpg",
        )

        page_id = self.db.record_page("test_sess_02", 1, ocr)
        self.assertGreater(page_id, 0)

        # Now check if hash is recognized as duplicate
        self.assertTrue(self.db.is_duplicate_page("test_sess_02", "abc123hash"))
        self.assertFalse(self.db.is_duplicate_page("test_sess_02", "other_hash"))


if __name__ == "__main__":
    unittest.main()
