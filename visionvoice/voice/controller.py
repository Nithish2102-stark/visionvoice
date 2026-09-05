"""
Controlled Voice Input Pipeline using SpeechRecognition.
Ensures strictly ONE microphone owner, thread safety, and seamless queue-based command dispatch.
"""

from __future__ import annotations
import queue
import threading
import time
from typing import Callable, Optional, Tuple
import speech_recognition as sr
from visionvoice.core.models import VoiceCommandIntent
from visionvoice.voice.commands import CommandParser
from visionvoice.utils.logging import get_logger

logger = get_logger("VoiceController")


class VoiceController:
    """
    Centralized Voice Controller for VisionVoice.
    Enforces strictly ONE microphone owner, preventing PyAudio hardware contention,
    while supporting synchronized command injection/simulation for automated testing.
    """

    def __init__(self) -> None:
        self.parser = CommandParser()
        self._recognizer = sr.Recognizer()
        self._recognizer.dynamic_energy_threshold = True
        self._recognizer.energy_threshold = 300
        self._recognizer.pause_threshold = 0.8

        self._mic: Optional[sr.Microphone] = None
        self._mic_lock = threading.Lock()
        self._command_queue: queue.Queue[Tuple[VoiceCommandIntent, Optional[str]]] = queue.Queue()
        self._has_microphone = self._init_microphone()

    def _init_microphone(self) -> bool:
        """Initializes and calibrates PyAudio microphone device if available."""
        try:
            self._mic = sr.Microphone()
            with self._mic_lock:
                with self._mic as source:
                    logger.info("Calibrating microphone for ambient noise...")
                    self._recognizer.adjust_for_ambient_noise(source, duration=0.8)
            logger.info("Microphone initialized successfully (single owner active).")
            return True
        except Exception as e:
            logger.warning(
                f"Microphone could not be initialized ({e}). "
                "Simulated / injected voice input fallback is active."
            )
            self._mic = None
            return False

    @property
    def has_microphone(self) -> bool:
        return self._has_microphone

    def inject_command(self, intent: VoiceCommandIntent, arg: Optional[str] = None) -> None:
        """Allows programmatic or CLI simulation of voice commands without microphone contention."""
        logger.info(f"Injecting simulated voice command: {intent.value} (arg={arg})")
        self._command_queue.put((intent, arg))

    def listen_single_utterance(
        self,
        timeout: float = 6.0,
        phrase_time_limit: float = 5.0
    ) -> Tuple[VoiceCommandIntent, Optional[str]]:
        """
        Listens synchronously for a single spoken utterance under microphone lock.
        Falls back to queued simulated commands first if present, or if microphone is unavailable.
        """
        # 1. Check if a command was injected via CLI / test queue first
        try:
            return self._command_queue.get_nowait()
        except queue.Empty:
            pass

        # 2. Return unknown if no hardware microphone is available
        if not self._has_microphone or self._mic is None:
            time.sleep(0.3)
            return VoiceCommandIntent.UNKNOWN, None

        # 3. Strictly acquire lock to ensure single microphone owner
        if not self._mic_lock.acquire(blocking=False):
            logger.debug("Microphone currently busy with another operation.")
            return VoiceCommandIntent.UNKNOWN, None

        try:
            with self._mic as source:
                logger.debug(f"Listening for voice command (timeout={timeout}s)...")
                audio = self._recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

            transcription = self._recognizer.recognize_google(audio)
            logger.info(f"Heard voice input: '{transcription}'")
            return self.parser.parse_command(transcription)

        except sr.WaitTimeoutError:
            return VoiceCommandIntent.UNKNOWN, None
        except sr.UnknownValueError:
            logger.debug("Audio detected but speech was not understood.")
            return VoiceCommandIntent.UNKNOWN, None
        except sr.RequestError as e:
            logger.warning(f"Speech recognition service error: {e}")
            return VoiceCommandIntent.UNKNOWN, None
        except Exception as e:
            logger.error(f"Voice listening error: {e}")
            return VoiceCommandIntent.UNKNOWN, None
        finally:
            self._mic_lock.release()

    def stop_listening(self) -> None:
        """Stops/clears pending voice operations gracefully."""
        # Clear pending injected queue on reset/shutdown
        while not self._command_queue.empty():
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("VoiceController reset/stopped.")
