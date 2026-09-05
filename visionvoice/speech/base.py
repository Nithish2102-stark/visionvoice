"""
Text-to-Speech (TTS) Hardware Abstraction Interface.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from visionvoice.utils.config import get_config
from visionvoice.utils.logging import get_logger

logger = get_logger("BaseTTS")


class BaseTTS(ABC):
    """Abstract interface for speech synthesis engines."""

    @abstractmethod
    def speak(self, text: str, language: str = "en", blocking: bool = True) -> bool:
        """Synthesizes and speaks the full text string."""
        pass

    @abstractmethod
    def pause(self) -> None:
        """Pauses the current speech playback."""
        pass

    @abstractmethod
    def resume(self) -> None:
        """Resumes paused speech playback."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Immediately stops all ongoing audio playback."""
        pass

    @abstractmethod
    def set_speed(self, words_per_minute: int) -> None:
        """Configures speech rate (WPM)."""
        pass

    @abstractmethod
    def set_volume(self, volume: float) -> None:
        """Configures playback volume (0.0 to 1.0)."""
        pass

    @abstractmethod
    def is_speaking(self) -> bool:
        """Returns True if audio is currently playing."""
        pass


def get_tts(backend: Optional[str] = None) -> BaseTTS:
    """
    Factory to instantiate the appropriate TTS engine ('mac' or 'pi').
    """
    cfg = get_config()
    target_backend = (backend or cfg.AUDIO_BACKEND).lower()

    if target_backend == "pi":
        from visionvoice.speech.pi_tts import PiTTS
        logger.info(f"Initializing Raspberry Pi TTS (espeak-ng, audio_device={cfg.PI_AUDIO_DEVICE})...")
        return PiTTS(speed=cfg.SPEECH_SPEED, audio_device=cfg.PI_AUDIO_DEVICE)
    else:
        from visionvoice.speech.mac_tts import MacTTS
        logger.info("Initializing macOS Native TTS (say)...")
        return MacTTS(speed=cfg.SPEECH_SPEED)
