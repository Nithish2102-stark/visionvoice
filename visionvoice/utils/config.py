"""
Configuration Manager for VisionVoice.
Loads settings from environment variables and .env file with intelligent defaults.
"""

from __future__ import annotations
import os
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple
from dotenv import load_dotenv

# Automatically load .env from project root if present
load_dotenv()


@dataclass
class Config:
    # Project Root and Paths
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    DATA_DIR: Path = field(init=False)
    CAPTURES_DIR: Path = field(init=False)
    PROCESSED_DIR: Path = field(init=False)
    OCR_DIR: Path = field(init=False)
    LOGS_DIR: Path = field(init=False)
    DB_PATH: Path = field(init=False)

    # Platform & Hardware Target
    PLATFORM: str = field(default_factory=lambda: os.getenv("PLATFORM", "mac").lower())

    # Camera Configuration
    CAMERA_BACKEND: str = field(default_factory=lambda: os.getenv("CAMERA_BACKEND", "mac").lower())
    CAMERA_INDEX: int = field(default_factory=lambda: int(os.getenv("CAMERA_INDEX", "0")))
    CAMERA_WIDTH: int = field(default_factory=lambda: int(os.getenv("CAMERA_WIDTH", "1920")))
    CAMERA_HEIGHT: int = field(default_factory=lambda: int(os.getenv("CAMERA_HEIGHT", "1080")))
    CAMERA_FPS: int = field(default_factory=lambda: int(os.getenv("CAMERA_FPS", "30")))

    # Audio / TTS Configuration
    AUDIO_BACKEND: str = field(default_factory=lambda: os.getenv("AUDIO_BACKEND", "mac").lower())
    PI_AUDIO_DEVICE: str = field(default_factory=lambda: os.getenv("PI_AUDIO_DEVICE", "plughw:3,0"))
    SPEECH_SPEED: int = field(default_factory=lambda: int(os.getenv("SPEECH_SPEED", "175")))
    SPEECH_VOLUME: float = field(default_factory=lambda: float(os.getenv("SPEECH_VOLUME", "1.0")))

    # OCR Supported Languages (Targeted language candidates are determined via Stage-1 script probe)
    SUPPORTED_LANGUAGES: str = field(default_factory=lambda: os.getenv("SUPPORTED_LANGUAGES", os.getenv("OCR_LANGUAGES", "eng,tam,kan,tel,mal,hin")))
    OCR_CONFIDENCE_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "40.0")))
    OCR_FAST_MODE: bool = field(default_factory=lambda: os.getenv("OCR_FAST_MODE", "false").lower() in ("true", "1", "yes"))
    OCR_MAX_VARIANTS: int = field(default_factory=lambda: int(os.getenv("OCR_MAX_VARIANTS", "4")))
    OCR_MAX_PSM: int = field(default_factory=lambda: int(os.getenv("OCR_MAX_PSM", "2")))
    OCR_EARLY_STOP_SCORE: float = field(default_factory=lambda: float(os.getenv("OCR_EARLY_STOP_SCORE", "85.0")))
    TESSERACT_CMD: str = field(default_factory=lambda: os.getenv("TESSERACT_CMD", ""))
    TESSDATA_PREFIX: str = field(default_factory=lambda: os.getenv("TESSDATA_PREFIX", ""))

    # Page Alignment & Stability
    PAGE_STABILITY_FRAMES: int = field(default_factory=lambda: int(os.getenv("PAGE_STABILITY_FRAMES", "4")))
    PAGE_MIN_AREA_RATIO: float = field(default_factory=lambda: float(os.getenv("PAGE_MIN_AREA_RATIO", "0.15")))
    PAGE_MAX_MOVEMENT_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("PAGE_MAX_MOVEMENT_THRESHOLD", "0.03")))
    PAGE_DETECTION_FPS: int = field(default_factory=lambda: int(os.getenv("PAGE_DETECTION_FPS", "10")))

    # Language, Preferences & Translation
    DEFAULT_LANGUAGE: str = field(default_factory=lambda: os.getenv("DEFAULT_LANGUAGE", "en"))
    TRANSLATION_ENABLED: bool = field(default_factory=lambda: os.getenv("TRANSLATION_ENABLED", "false").lower() in ("true", "1", "yes"))
    TARGET_TRANSLATION_LANGUAGE: str = field(default_factory=lambda: os.getenv("TARGET_TRANSLATION_LANGUAGE", "en"))

    # Hardware / GPIO Pins (Raspberry Pi only)
    PIN_BUTTON_POWER: int = field(default_factory=lambda: int(os.getenv("PIN_BUTTON_POWER", "17")))
    PIN_BUTTON_VOL_UP: int = field(default_factory=lambda: int(os.getenv("PIN_BUTTON_VOL_UP", "27")))
    PIN_BUTTON_VOL_DOWN: int = field(default_factory=lambda: int(os.getenv("PIN_BUTTON_VOL_DOWN", "22")))
    PIN_LED_STATUS: int = field(default_factory=lambda: int(os.getenv("PIN_LED_STATUS", "24")))

    # Development & Debug Settings
    DEBUG_MODE: bool = field(default_factory=lambda: os.getenv("DEBUG_MODE", "true").lower() in ("true", "1", "yes"))
    SHOW_CAMERA_PREVIEW: bool = field(default_factory=lambda: os.getenv("SHOW_CAMERA_PREVIEW", "false").lower() in ("true", "1", "yes"))
    SAVE_INTERMEDIATE_IMAGES: bool = field(default_factory=lambda: os.getenv("SAVE_INTERMEDIATE_IMAGES", "true").lower() in ("true", "1", "yes"))

    def __post_init__(self) -> None:
        """Initialize and create necessary project directory structure."""
        self.DATA_DIR = self.BASE_DIR / "data"
        self.CAPTURES_DIR = self.DATA_DIR / "captures"
        self.PROCESSED_DIR = self.DATA_DIR / "processed"
        self.OCR_DIR = self.DATA_DIR / "ocr"
        self.LOGS_DIR = self.DATA_DIR / "logs"
        self.DB_PATH = self.DATA_DIR / "visionvoice.db"

        # Ensure all data subdirectories exist
        for d in [self.CAPTURES_DIR, self.PROCESSED_DIR, self.OCR_DIR, self.LOGS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        # Auto-detect Tesseract binary path if not explicitly provided
        if not self.TESSERACT_CMD:
            # Check common locations
            candidates = [
                "/opt/homebrew/bin/tesseract",
                "/usr/local/bin/tesseract",
                "/usr/bin/tesseract",
                shutil.which("tesseract") or ""
            ]
            for candidate in candidates:
                if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                    self.TESSERACT_CMD = candidate
                    break

    @property
    def resolution(self) -> Tuple[int, int]:
        """Returns the capture resolution as a (width, height) tuple."""
        return self.CAMERA_WIDTH, self.CAMERA_HEIGHT

    @property
    def supported_languages_list(self) -> List[str]:
        """Returns the list of supported OCR languages."""
        raw = self.SUPPORTED_LANGUAGES or "eng,tam,kan,tel,mal,hin"
        return [lang.strip() for lang in raw.replace("+", ",").split(",") if lang.strip()]

    @property
    def language_list(self) -> List[str]:
        """Returns the list of supported OCR language codes (alias)."""
        return self.supported_languages_list


_config_instance: Config | None = None


def get_config() -> Config:
    """Returns the singleton configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
