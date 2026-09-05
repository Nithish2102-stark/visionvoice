"""
Data models and state definitions for VisionVoice assistive reader.
"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import numpy as np


class DeviceState(Enum):
    """Finite State Machine states."""
    INITIALIZING = "INITIALIZING"
    WAIT_WAKE_WORD = "WAIT_WAKE_WORD"
    ASK_PREFERENCES = "ASK_PREFERENCES"
    WAIT_START = "WAIT_START"
    ALIGNING_PAGE = "ALIGNING_PAGE"
    PROCESSING_PAGE = "PROCESSING_PAGE"
    READING = "READING"
    PAUSED = "PAUSED"
    PAGE_FINISHED = "PAGE_FINISHED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class PageDetectionStatus(Enum):
    """Page detection state in live camera stream."""
    PAGE_NOT_FOUND = "PAGE_NOT_FOUND"
    PAGE_DETECTED = "PAGE_DETECTED"
    PAGE_STABLE = "PAGE_STABLE"


class VoiceCommandIntent(Enum):
    """Recognized voice commands."""
    WAKE_WORD = "WAKE_WORD"
    START_READING = "START_READING"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    REPEAT = "REPEAT"
    NEXT_PAGE = "NEXT_PAGE"
    PREVIOUS_PAGE = "PREVIOUS_PAGE"
    STOP_READING = "STOP_READING"
    VOLUME_UP = "VOLUME_UP"
    VOLUME_DOWN = "VOLUME_DOWN"
    PREF_ORIGINAL = "PREF_ORIGINAL"
    PREF_TRANSLATE = "PREF_TRANSLATE"
    SET_LANGUAGE = "SET_LANGUAGE"
    UNKNOWN = "UNKNOWN"


@dataclass
class PageFrameAnalysis:
    """Per-frame page detection and stability metrics."""
    status: PageDetectionStatus
    contour: Optional[np.ndarray] = None
    quadrilateral: Optional[np.ndarray] = None
    area_ratio: float = 0.0
    aspect_ratio: float = 0.0
    stability_score: float = 0.0
    stable_frames_count: int = 0
    annotated_frame: Optional[np.ndarray] = None


@dataclass
class OCRQualityMetrics:
    """Multi-dimensional quality score for OCR output."""
    word_confidence_avg: float
    valid_word_ratio: float
    alpha_ratio: float
    symbol_penalty: float
    line_consistency: float
    composite_score: float
    total_words: int
    total_chars: int
    raw_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRVariantResult:
    """OCR result from a single preprocessing variant + PSM mode."""
    variant_name: str
    psm_mode: int
    language: str
    raw_text: str
    cleaned_text: str
    metrics: OCRQualityMetrics
    words_data: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class OCRResult:
    """Final selected best OCR output."""
    text: str
    cleaned_text: str
    sentences: List[str]
    detected_language: str
    detected_script: str
    composite_score: float
    average_confidence: float
    selected_variant: str
    selected_psm: int
    image_hash: str
    original_image_path: str
    processed_image_path: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_valid: bool = True
    error_message: Optional[str] = None


@dataclass
class ReadingSession:
    """Tracks current active reading session and user preferences."""
    session_id: str
    book_id: Optional[str] = None
    current_page_number: int = 0
    preferred_language: str = "en"
    read_mode: str = "original"  # "original" or "translated"
    current_sentence_index: int = 0
    sentences: List[str] = field(default_factory=list)
    current_ocr_result: Optional[OCRResult] = None
    is_paused: bool = False
    volume: float = 1.0
    speed: int = 175
