"""
SQLite Memory Database for Assistive Reading Device.
Tracks reading sessions, page history, image hashes to prevent duplicate reads, and reading progress.
"""

from __future__ import annotations
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from visionvoice.core.models import OCRResult, ReadingSession
from visionvoice.utils.config import get_config
from visionvoice.utils.logging import get_logger

logger = get_logger("DatabaseManager")


class DatabaseManager:
    """Manages local SQLite database operations."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        cfg = get_config()
        self.db_path = db_path or cfg.DB_PATH
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes database schema if tables do not exist."""
        with self._lock:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = self._get_connection()
            try:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS sessions (
                            session_id TEXT PRIMARY KEY,
                            book_id TEXT,
                            start_time TEXT,
                            last_updated TEXT,
                            preferred_language TEXT,
                            read_mode TEXT,
                            current_page INTEGER,
                            current_sentence INTEGER
                        )
                    """)
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS pages (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT,
                            page_index INTEGER,
                            image_hash TEXT,
                            detected_language TEXT,
                            composite_score REAL,
                            average_confidence REAL,
                            selected_variant TEXT,
                            selected_psm INTEGER,
                            raw_text TEXT,
                            cleaned_text TEXT,
                            sentences_json TEXT,
                            original_image_path TEXT,
                            processed_image_path TEXT,
                            created_at TEXT,
                            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                        )
                    """)
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_hash ON pages (image_hash)")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_page ON pages (session_id, page_index)")
            finally:
                conn.close()

        logger.info(f"Database initialized at: {self.db_path}")

    def is_duplicate_page(self, session_id: str, image_hash: str) -> bool:
        """Checks if the page image hash was already read in the active session."""
        if not image_hash:
            return False

        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM pages WHERE session_id = ? AND image_hash = ?",
                    (session_id, image_hash)
                )
                count = cursor.fetchone()[0]
                return count > 0
            finally:
                conn.close()

    def create_or_update_session(self, session: ReadingSession) -> None:
        """Saves current session state to database."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO sessions (
                            session_id, book_id, start_time, last_updated,
                            preferred_language, read_mode, current_page, current_sentence
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(session_id) DO UPDATE SET
                            last_updated = excluded.last_updated,
                            preferred_language = excluded.preferred_language,
                            read_mode = excluded.read_mode,
                            current_page = excluded.current_page,
                            current_sentence = excluded.current_sentence
                    """, (
                        session.session_id,
                        session.book_id,
                        now,
                        now,
                        session.preferred_language,
                        session.read_mode,
                        session.current_page_number,
                        session.current_sentence_index,
                    ))
            finally:
                conn.close()

    def record_page(self, session_id: str, page_index: int, ocr: OCRResult) -> int:
        """Stores a scanned and OCR'd page into database."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO pages (
                            session_id, page_index, image_hash, detected_language,
                            composite_score, average_confidence, selected_variant,
                            selected_psm, raw_text, cleaned_text, sentences_json,
                            original_image_path, processed_image_path, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        session_id,
                        page_index,
                        ocr.image_hash,
                        ocr.detected_language,
                        ocr.composite_score,
                        ocr.average_confidence,
                        ocr.selected_variant,
                        ocr.selected_psm,
                        ocr.text,
                        ocr.cleaned_text,
                        json.dumps(ocr.sentences, ensure_ascii=False),
                        ocr.original_image_path,
                        ocr.processed_image_path,
                        now,
                    ))
                    page_id = cursor.lastrowid
                    logger.info(f"Recorded page {page_index} (id={page_id}) in database for session {session_id}")
                    return page_id or 0
            finally:
                conn.close()


_db_instance: Optional[DatabaseManager] = None


def get_database() -> DatabaseManager:
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
