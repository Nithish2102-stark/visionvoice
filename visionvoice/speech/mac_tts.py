"""
macOS Native Text-to-Speech Engine using the built-in 'say' command.
Supports sentence playback, interruptible audio, rate adjustment, and Indian voices.
"""

from __future__ import annotations
import os
import signal
import subprocess
import threading
from typing import Optional, Dict, Set
from visionvoice.speech.base import BaseTTS
from visionvoice.utils.logging import get_logger

logger = get_logger("MacTTS")


class MacTTS(BaseTTS):
    """macOS TTS driver wrapping Apple's Speech Synthesis manager."""

    # Map language codes to available macOS voices
    VOICE_MAP: Dict[str, str] = {
        "en": "Samantha",
        "eng": "Samantha",
        "hi": "Rishi",
        "hin": "Rishi",
        "ta": "Veena",
        "tam": "Veena",
        "te": "Rishi",   # Fallback Indic voice
        "tel": "Rishi",
        "kn": "Rishi",
        "kan": "Rishi",
        "ml": "Rishi",
        "mal": "Rishi",
    }

    def __init__(self, speed: int = 175) -> None:
        self.speed = speed
        self.volume = 1.0
        self._current_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._is_paused = False
        self._available_voices = self._query_installed_voices()

    def _query_installed_voices(self) -> Set[str]:
        """Queries macOS for currently installed system voice names."""
        try:
            res = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True)
            installed = set()
            for line in res.stdout.splitlines():
                parts = line.split()
                if parts:
                    installed.add(parts[0])
            return installed
        except Exception:
            return set()

    def _get_voice_for_language(self, language: str) -> Optional[str]:
        """Selects the best installed voice for the target language."""
        desired = self.VOICE_MAP.get(language.lower(), "Samantha")
        if desired in self._available_voices:
            return desired
        if "Samantha" in self._available_voices:
            return "Samantha"
        return None

    def speak(self, text: str, language: str = "en", blocking: bool = True) -> bool:
        """Synthesizes speech on macOS with immediate interruption support."""
        if not text or not text.strip():
            return True

        self.stop()

        with self._lock:
            voice = self._get_voice_for_language(language)
            cmd = ["say", "-r", str(self.speed)]
            if voice:
                cmd.extend(["-v", voice])
            cmd.append(text)

            logger.debug(f"Speaking (lang={language}, voice={voice}): '{text[:50]}...'")
            try:
                self._current_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                )
                self._is_paused = False
            except Exception as e:
                logger.error(f"Failed to launch macOS 'say' process: {e}")
                return False

        if blocking and self._current_process:
            self._current_process.wait()
            return True
        return True

    def pause(self) -> None:
        """Pauses the current speech process using SIGSTOP."""
        with self._lock:
            if self._current_process and self._current_process.poll() is None and not self._is_paused:
                try:
                    os.kill(self._current_process.pid, signal.SIGSTOP)
                    self._is_paused = True
                    logger.info("Speech paused.")
                except Exception as e:
                    logger.warning(f"Failed to pause speech: {e}")

    def resume(self) -> None:
        """Resumes the paused speech process using SIGCONT."""
        with self._lock:
            if self._current_process and self._current_process.poll() is None and self._is_paused:
                try:
                    os.kill(self._current_process.pid, signal.SIGCONT)
                    self._is_paused = False
                    logger.info("Speech resumed.")
                except Exception as e:
                    logger.warning(f"Failed to resume speech: {e}")

    def stop(self) -> None:
        """Stops ongoing speech immediately."""
        with self._lock:
            if self._current_process and self._current_process.poll() is None:
                try:
                    self._current_process.terminate()
                    self._current_process.wait(timeout=0.5)
                except Exception:
                    try:
                        self._current_process.kill()
                    except Exception:
                        pass
                self._current_process = None
                self._is_paused = False
                logger.debug("Speech stopped.")

    def set_speed(self, words_per_minute: int) -> None:
        self.speed = max(80, min(350, words_per_minute))
        logger.info(f"Speech speed set to {self.speed} WPM")

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, volume))

    def is_speaking(self) -> bool:
        with self._lock:
            return self._current_process is not None and self._current_process.poll() is None
