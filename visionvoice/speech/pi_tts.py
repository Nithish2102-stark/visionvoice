"""
Raspberry Pi 4 Text-to-Speech Engine using eSpeak-ng and aplay.
Supports sentence synthesis, Indian language phonetics, and USB audio routing (e.g., plughw:3,0).
"""

from __future__ import annotations
import os
import signal
import subprocess
import threading
from typing import Optional, Dict
from visionvoice.speech.base import BaseTTS
from visionvoice.utils.logging import get_logger

logger = get_logger("PiTTS")


class PiTTS(BaseTTS):
    """eSpeak-ng driver for Raspberry Pi 4 with USB ALSA device support."""

    VOICE_MAP: Dict[str, str] = {
        "en": "en-us",
        "eng": "en-us",
        "hi": "hi",
        "hin": "hi",
        "ta": "ta",
        "tam": "ta",
        "te": "te",
        "tel": "te",
        "kn": "kn",
        "kan": "kn",
        "ml": "ml",
        "mal": "ml",
    }

    def __init__(self, speed: int = 175, audio_device: Optional[str] = "plughw:3,0") -> None:
        self.speed = speed
        self.volume = 1.0
        self.audio_device = audio_device
        self._current_process: Optional[subprocess.Popen] = None
        self._pipe_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._is_paused = False

    def speak(self, text: str, language: str = "en", blocking: bool = True) -> bool:
        """Synthesizes speech using espeak-ng piped to aplay for USB DAC hardware."""
        if not text or not text.strip():
            return True

        self.stop()

        with self._lock:
            voice = self.VOICE_MAP.get(language.lower(), "en-us")
            logger.debug(f"Pi TTS speaking ({voice}): '{text[:50]}...'")

            try:
                # If a custom ALSA audio device is configured, pipe stdout to aplay
                if self.audio_device:
                    espeak_cmd = ["espeak-ng", "-v", voice, "-s", str(self.speed), "--stdout", text]
                    aplay_cmd = ["aplay", "-D", self.audio_device, "-q"]

                    self._current_process = subprocess.Popen(
                        espeak_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )
                    self._pipe_process = subprocess.Popen(
                        aplay_cmd,
                        stdin=self._current_process.stdout,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                    )
                    # Allow espeak to receive SIGPIPE if aplay exits
                    if self._current_process.stdout:
                        self._current_process.stdout.close()
                else:
                    # Default audio device
                    espeak_cmd = ["espeak-ng", "-v", voice, "-s", str(self.speed), text]
                    self._pipe_process = subprocess.Popen(
                        espeak_cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                    )

                self._is_paused = False
            except Exception as e:
                logger.error(f"Failed to launch espeak-ng process: {e}")
                return False

        if blocking and self._pipe_process:
            self._pipe_process.wait()
            return True
        return True

    def pause(self) -> None:
        """Pauses audio playback."""
        with self._lock:
            proc = self._pipe_process or self._current_process
            if proc and proc.poll() is None and not self._is_paused:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGSTOP)
                    self._is_paused = True
                    logger.info("Pi Speech paused.")
                except Exception as e:
                    logger.warning(f"Failed to pause Pi speech: {e}")

    def resume(self) -> None:
        """Resumes paused audio playback."""
        with self._lock:
            proc = self._pipe_process or self._current_process
            if proc and proc.poll() is None and self._is_paused:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGCONT)
                    self._is_paused = False
                    logger.info("Pi Speech resumed.")
                except Exception as e:
                    logger.warning(f"Failed to resume Pi speech: {e}")

    def stop(self) -> None:
        """Immediately stops all ongoing Pi audio processes."""
        with self._lock:
            for proc in [self._current_process, self._pipe_process]:
                if proc and proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        proc.wait(timeout=0.3)
                    except Exception:
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except Exception:
                            pass
            self._current_process = None
            self._pipe_process = None
            self._is_paused = False

    def set_speed(self, words_per_minute: int) -> None:
        self.speed = max(80, min(350, words_per_minute))

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, volume))

    def is_speaking(self) -> bool:
        with self._lock:
            proc = self._pipe_process or self._current_process
            return proc is not None and proc.poll() is None
