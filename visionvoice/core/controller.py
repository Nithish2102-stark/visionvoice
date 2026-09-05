"""
VisionVoice Master Controller.
Integrates StateMachine, Camera, Advanced OCR, TTS, Voice Recognition,
Hardware Peripherals, and Session Persistence.
"""

from __future__ import annotations
import time
import uuid
import threading
from typing import Optional, List
import cv2
import numpy as np

from visionvoice.core.models import (
    DeviceState,
    PageDetectionStatus,
    PageFrameAnalysis,
    OCRResult,
    ReadingSession,
    VoiceCommandIntent,
)
from visionvoice.core.state_machine import StateMachine
from visionvoice.camera.base import BaseCamera, get_camera
from visionvoice.speech.base import BaseTTS, get_tts
from visionvoice.voice.controller import VoiceController
from visionvoice.hardware.base import BaseHardware, ButtonEvent, LEDState, get_hardware
from visionvoice.ocr.engine import OCREngine
from visionvoice.ocr.page_detector import PageDetector
from visionvoice.translation.translator import get_translator
from visionvoice.memory.database import get_database
from visionvoice.utils.config import get_config
from visionvoice.utils.logging import get_logger

logger = get_logger("Controller")


class VisionVoiceController:
    """Central orchestrator for the assistive reading device."""

    def __init__(self, show_preview: Optional[bool] = None) -> None:
        self.cfg = get_config()
        self.show_preview = show_preview if show_preview is not None else self.cfg.SHOW_CAMERA_PREVIEW

        # 1. State Machine
        self.state_machine = StateMachine(DeviceState.INITIALIZING)
        self.state_machine.add_listener(self._on_state_changed)

        # 2. Subsystems
        self.camera: BaseCamera = get_camera()
        self.tts: BaseTTS = get_tts()
        self.voice: VoiceController = VoiceController()
        self.hardware: BaseHardware = get_hardware()
        self.ocr_engine = OCREngine()
        self.page_detector = PageDetector()
        self.translator = get_translator()
        self.db = get_database()

        # 3. Active Session
        self.session = ReadingSession(
            session_id=str(uuid.uuid4())[:8],
            preferred_language=self.cfg.DEFAULT_LANGUAGE,
            read_mode="translated" if self.cfg.TRANSLATION_ENABLED else "original",
            speed=self.cfg.SPEECH_SPEED,
            volume=self.cfg.SPEECH_VOLUME,
        )

        # 4. Control flags & threads
        self._is_running = False
        self._stop_requested = threading.Event()
        self._pause_event = threading.Event()
        self._reading_thread: Optional[threading.Thread] = None
        self._current_page_sentences: List[str] = []
        self._last_detected_quad: Optional[np.ndarray] = None

        # 5. Register Hardware Callbacks
        self.hardware.register_button_callback(self._on_hardware_button)

    def start(self) -> None:
        """Initializes hardware and begins the main device control loop."""
        logger.info("==========================================")
        logger.info("   STARTING VISIONVOICE ASSISTIVE READER  ")
        logger.info("==========================================")

        self._is_running = True
        self._stop_requested.clear()

        # Initialize hardware
        self.hardware.start()
        self.hardware.set_led_state(LEDState.SOLID_ON)

        # Start Camera
        if not self.camera.start():
            logger.warning("Camera could not be opened at startup. Will retry during alignment.")

        # Announce readiness
        self.tts.speak("VisionVoice is ready. Say 'Hey VisionVoice' or press power to begin.", blocking=False)
        self.state_machine.transition_to(DeviceState.WAIT_WAKE_WORD, reason="Device initialized")

        # Start main state dispatcher loop
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Shutdown requested via KeyboardInterrupt.")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {e}", exc_info=True)
            self.state_machine.transition_to(DeviceState.ERROR, reason=str(e))
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Gracefully releases all hardware resources."""
        logger.info("Shutting down VisionVoice subsystems...")
        self._is_running = False
        self._stop_requested.set()

        self.tts.stop()
        self.voice.stop_listening()
        self.camera.stop()
        self.hardware.set_led_state(LEDState.OFF)
        self.hardware.stop()
        cv2.destroyAllWindows()
        logger.info("VisionVoice shutdown complete.")

    def _on_state_changed(self, old_state: DeviceState, new_state: DeviceState) -> None:
        """Updates LED patterns and audio queues on state transitions."""
        logger.info(f"==> System State Changed: {old_state.value} -> {new_state.value}")

        if new_state == DeviceState.WAIT_WAKE_WORD:
            self.hardware.set_led_state(LEDState.SLOW_PULSE)
        elif new_state == DeviceState.ASK_PREFERENCES:
            self.hardware.set_led_state(LEDState.SOLID_ON)
        elif new_state == DeviceState.WAIT_START:
            self.hardware.set_led_state(LEDState.SLOW_PULSE)
        elif new_state == DeviceState.ALIGNING_PAGE:
            self.hardware.set_led_state(LEDState.SLOW_PULSE)
        elif new_state == DeviceState.PROCESSING_PAGE:
            self.hardware.set_led_state(LEDState.FAST_BLINK)
        elif new_state == DeviceState.READING:
            self.hardware.set_led_state(LEDState.SOLID_ON)
        elif new_state == DeviceState.PAUSED:
            self.hardware.set_led_state(LEDState.SLOW_PULSE)
        elif new_state == DeviceState.PAGE_FINISHED:
            self.hardware.set_led_state(LEDState.SOLID_ON)
        elif new_state == DeviceState.ERROR:
            self.hardware.set_led_state(LEDState.ERROR_BLINK)

    def _main_loop(self) -> None:
        """Primary dispatcher executing state-specific actions."""
        while self._is_running and not self._stop_requested.is_set():
            state = self.state_machine.current_state

            if state == DeviceState.WAIT_WAKE_WORD:
                self._handle_wait_wake_word()
            elif state == DeviceState.ASK_PREFERENCES:
                self._handle_ask_preferences()
            elif state == DeviceState.WAIT_START:
                self._handle_wait_start()
            elif state == DeviceState.ALIGNING_PAGE:
                self._handle_aligning_page()
            elif state == DeviceState.PROCESSING_PAGE:
                self._handle_processing_page()
            elif state == DeviceState.READING:
                self._handle_reading_state()
            elif state == DeviceState.PAUSED:
                self._handle_paused_state()
            elif state == DeviceState.PAGE_FINISHED:
                self._handle_page_finished()
            elif state == DeviceState.STOPPED:
                logger.info("Device is in STOPPED state. Exiting main loop.")
                break
            elif state == DeviceState.ERROR:
                self._handle_error_state()
            else:
                time.sleep(0.1)

    def _handle_wait_wake_word(self) -> None:
        """Listens for 'Hey VisionVoice'."""
        intent, arg = self.voice.listen_single_utterance(timeout=5.0)
        if intent in (VoiceCommandIntent.WAKE_WORD, VoiceCommandIntent.START_READING):
            logger.info("Wake word detected!")
            self.tts.speak(
                "Hello! What is your preferred language, and would you like original or translated reading?",
                blocking=True
            )
            self.state_machine.transition_to(DeviceState.ASK_PREFERENCES, reason="Wake word heard")
        elif intent == VoiceCommandIntent.STOP_READING:
            self.state_machine.transition_to(DeviceState.STOPPED, reason="User requested stop")

    def _handle_ask_preferences(self) -> None:
        """Asks language and reading mode (original or translated)."""
        intent, arg = self.voice.listen_single_utterance(timeout=6.0)

        if intent == VoiceCommandIntent.SET_LANGUAGE and arg:
            self.session.preferred_language = arg
            logger.info(f"Set preferred language: {arg}")
            self.tts.speak(f"Language set to {arg}. Please say original or translated.", blocking=True)
            return

        if intent == VoiceCommandIntent.PREF_ORIGINAL:
            self.session.read_mode = "original"
            logger.info("User selected ORIGINAL language mode.")
            self.tts.speak("Got it, reading in original language. Place your book and say 'Start reading'.", blocking=True)
            self.state_machine.transition_to(DeviceState.WAIT_START, reason="Preferences set")
            return

        if intent == VoiceCommandIntent.PREF_TRANSLATE:
            self.session.read_mode = "translated"
            logger.info("User selected TRANSLATED mode.")
            self.tts.speak(f"Got it, translating to {self.session.preferred_language}. Place your book and say 'Start reading'.", blocking=True)
            self.state_machine.transition_to(DeviceState.WAIT_START, reason="Preferences set")
            return

        if intent == VoiceCommandIntent.START_READING:
            self.state_machine.transition_to(DeviceState.ALIGNING_PAGE, reason="User started reading directly")
            return

        # Default fallback if silence / timeout
        if intent == VoiceCommandIntent.UNKNOWN:
            self.tts.speak("Ready to read. Say 'Start reading' when your book is placed.", blocking=False)
            self.state_machine.transition_to(DeviceState.WAIT_START, reason="Default preferences accepted")

    def _handle_wait_start(self) -> None:
        """Waits for start reading command or power button."""
        intent, arg = self.voice.listen_single_utterance(timeout=5.0)
        if intent == VoiceCommandIntent.START_READING:
            self.tts.speak("Looking for page. Please hold still.", blocking=False)
            self.state_machine.transition_to(DeviceState.ALIGNING_PAGE, reason="Start reading command received")
        elif intent == VoiceCommandIntent.STOP_READING:
            self.state_machine.transition_to(DeviceState.STOPPED, reason="User stopped")

    def _handle_aligning_page(self) -> None:
        """Inspects live camera frames until a page is detected and stable."""
        self.page_detector.reset_stability()
        logger.info("Inspecting camera stream for page alignment...")

        while self.state_machine.current_state == DeviceState.ALIGNING_PAGE and not self._stop_requested.is_set():
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            analysis: PageFrameAnalysis = self.page_detector.analyze_frame(frame, draw_overlay=self.show_preview)

            if self.show_preview and analysis.annotated_frame is not None:
                cv2.imshow("VisionVoice Camera Preview", analysis.annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.state_machine.transition_to(DeviceState.STOPPED, reason="User quit preview window")
                    break
                elif key == 32:  # Spacebar forces manual capture in dev mode
                    self._last_detected_quad = analysis.quadrilateral
                    self.state_machine.transition_to(DeviceState.PROCESSING_PAGE, reason="Manual spacebar capture")
                    break

            if analysis.status == PageDetectionStatus.PAGE_STABLE:
                logger.info("Book page is STABLE! Transitioning to capture & OCR.")
                self._last_detected_quad = analysis.quadrilateral
                self.tts.speak("Page detected. Processing...", blocking=False)
                self.state_machine.transition_to(DeviceState.PROCESSING_PAGE, reason="Page stable")
                break

            time.sleep(0.04)

    def _handle_processing_page(self) -> None:
        """Captures still image, performs OCR, language detection, translation, and persistence."""
        logger.info("Capturing high-resolution still frame...")
        still_image = self.camera.capture_high_res()

        if still_image is None:
            logger.error("High-res capture returned None.")
            self.tts.speak("Failed to capture image. Retrying alignment.", blocking=True)
            self.state_machine.transition_to(DeviceState.ALIGNING_PAGE, reason="Capture failed")
            return

        # Execute Multi-Variant OCR Pipeline
        logger.info("Executing High-Accuracy OCR Pipeline...")
        ocr_result: OCRResult = self.ocr_engine.process_image(
            still_image,
            quadrilateral=self._last_detected_quad,
        )

        if not ocr_result.is_valid or not ocr_result.sentences:
            logger.warning(f"OCR failed or produced empty text: {ocr_result.error_message}")
            self.tts.speak("Could not read text clearly. Please reposition the book and hold still.", blocking=True)
            self.state_machine.transition_to(DeviceState.ALIGNING_PAGE, reason="OCR produced no text")
            return

        # Check duplicate page in current session
        if self.db.is_duplicate_page(self.session.session_id, ocr_result.image_hash):
            logger.info("Duplicate page detected (same hash).")
            self.tts.speak("This page was already read. Please turn to the next page.", blocking=True)
            self.state_machine.transition_to(DeviceState.ALIGNING_PAGE, reason="Duplicate page detected")
            return

        # Handle Translation if requested
        sentences_to_read = ocr_result.sentences
        if self.session.read_mode == "translated" and ocr_result.detected_language != self.session.preferred_language:
            logger.info(f"Translating {len(sentences_to_read)} sentences to '{self.session.preferred_language}'...")
            sentences_to_read = self.translator.translate_sentences(
                ocr_result.sentences,
                source_lang=ocr_result.detected_language,
                target_lang=self.session.preferred_language
            )

        # Update Session & Record in Database
        self.session.current_page_number += 1
        self.session.current_sentence_index = 0
        self.session.sentences = sentences_to_read
        self.session.current_ocr_result = ocr_result
        self._current_page_sentences = sentences_to_read

        self.db.create_or_update_session(self.session)
        self.db.record_page(self.session.session_id, self.session.current_page_number, ocr_result)

        logger.info(f"Ready to read Page {self.session.current_page_number} ({len(sentences_to_read)} sentences).")
        self.state_machine.transition_to(DeviceState.READING, reason="Page processed successfully")

    def _handle_reading_state(self) -> None:
        """Reads page sentences one-by-one with interruptibility."""
        sentences = self._current_page_sentences
        total_sentences = len(sentences)
        lang = self.session.preferred_language if self.session.read_mode == "translated" else (
            self.session.current_ocr_result.detected_language if self.session.current_ocr_result else "en"
        )

        logger.info(f"Reading {total_sentences} sentences in language '{lang}'...")

        while self.session.current_sentence_index < total_sentences:
            if self.state_machine.current_state != DeviceState.READING or self._stop_requested.is_set():
                break

            sentence = sentences[self.session.current_sentence_index]
            logger.info(f"Reading [{self.session.current_sentence_index + 1}/{total_sentences}]: {sentence}")

            # Speak sentence
            self.tts.speak(sentence, language=lang, blocking=True)

            # Check if an interrupt transitioned state during playback
            if self.state_machine.current_state != DeviceState.READING:
                break

            self.session.current_sentence_index += 1
            self.db.create_or_update_session(self.session)
            time.sleep(0.15)  # Natural breathing pause between sentences

        if self.session.current_sentence_index >= total_sentences:
            logger.info("Page reading completed.")
            self.state_machine.transition_to(DeviceState.PAGE_FINISHED, reason="All sentences read")

    def _handle_paused_state(self) -> None:
        """Handles paused reading."""
        time.sleep(0.2)

    def _handle_page_finished(self) -> None:
        """Prompts for next page, repeat, or stop."""
        self.tts.speak("Page finished. Say 'Next page', 'Repeat', or 'Stop reading'.", blocking=True)
        intent, arg = self.voice.listen_single_utterance(timeout=8.0)

        if intent in (VoiceCommandIntent.NEXT_PAGE, VoiceCommandIntent.START_READING):
            self.state_machine.transition_to(DeviceState.ALIGNING_PAGE, reason="Next page requested")
        elif intent == VoiceCommandIntent.REPEAT:
            self.session.current_sentence_index = 0
            self.state_machine.transition_to(DeviceState.READING, reason="Repeat page requested")
        elif intent == VoiceCommandIntent.STOP_READING:
            self.tts.speak("Session stopped. Goodbye.", blocking=True)
            self.state_machine.transition_to(DeviceState.STOPPED, reason="User stopped reading")
        else:
            # Re-prompt or stay in page finished
            time.sleep(0.5)

    def _handle_error_state(self) -> None:
        """Error recovery handler."""
        logger.warning("System in ERROR state. Attempting recovery in 2 seconds...")
        self.tts.speak("An error occurred. Recovering system...", blocking=True)
        time.sleep(2.0)
        self.state_machine.transition_to(DeviceState.WAIT_START, reason="Error recovered")

    def _on_hardware_button(self, event: ButtonEvent) -> None:
        """Handles physical or simulated hardware button presses."""
        logger.info(f"Hardware button received: {event.value}")

        if event == ButtonEvent.POWER_PRESS:
            state = self.state_machine.current_state
            if state in (DeviceState.WAIT_WAKE_WORD, DeviceState.WAIT_START):
                self.state_machine.transition_to(DeviceState.ALIGNING_PAGE, reason="Power button pressed to start")
            elif state == DeviceState.READING:
                self.tts.pause()
                self.state_machine.transition_to(DeviceState.PAUSED, reason="Power button pressed to pause")
            elif state == DeviceState.PAUSED:
                self.state_machine.transition_to(DeviceState.READING, reason="Power button pressed to resume")
                self.tts.resume()
            elif state == DeviceState.PAGE_FINISHED:
                self.state_machine.transition_to(DeviceState.ALIGNING_PAGE, reason="Power button pressed for next page")

        elif event == ButtonEvent.POWER_LONG_PRESS:
            logger.info("Power long press: stopping reading session.")
            self.tts.stop()
            self.state_machine.transition_to(DeviceState.STOPPED, reason="Power button long press")

        elif event == ButtonEvent.VOL_UP_PRESS:
            new_speed = min(350, self.session.speed + 25)
            self.session.speed = new_speed
            self.tts.set_speed(new_speed)
            logger.info(f"Increased reading speed to {new_speed} WPM")

        elif event == ButtonEvent.VOL_DOWN_PRESS:
            new_speed = max(80, self.session.speed - 25)
            self.session.speed = new_speed
            self.tts.set_speed(new_speed)
            logger.info(f"Decreased reading speed to {new_speed} WPM")

    def process_external_command(self, intent: VoiceCommandIntent, arg: Optional[str] = None) -> None:
        """Dispatches external voice/CLI commands into the active state machine."""
        logger.info(f"Dispatching external command: {intent.value} (arg={arg})")
        state = self.state_machine.current_state

        if intent == VoiceCommandIntent.PAUSE and state == DeviceState.READING:
            self.tts.pause()
            self.state_machine.transition_to(DeviceState.PAUSED, reason="Voice pause command")
        elif intent == VoiceCommandIntent.RESUME and state == DeviceState.PAUSED:
            self.state_machine.transition_to(DeviceState.READING, reason="Voice resume command")
            self.tts.resume()
        elif intent == VoiceCommandIntent.REPEAT:
            if state in (DeviceState.READING, DeviceState.PAUSED):
                self.session.current_sentence_index = max(0, self.session.current_sentence_index - 1)
                self.state_machine.transition_to(DeviceState.READING, reason="Repeat sentence")
            elif state == DeviceState.PAGE_FINISHED:
                self.session.current_sentence_index = 0
                self.state_machine.transition_to(DeviceState.READING, reason="Repeat page")
        elif intent == VoiceCommandIntent.NEXT_PAGE:
            self.tts.stop()
            self.state_machine.transition_to(DeviceState.ALIGNING_PAGE, reason="Voice next page")
        elif intent == VoiceCommandIntent.STOP_READING:
            self.tts.stop()
            self.state_machine.transition_to(DeviceState.STOPPED, reason="Voice stop reading")
        elif intent == VoiceCommandIntent.VOLUME_UP:
            self._on_hardware_button(ButtonEvent.VOL_UP_PRESS)
        elif intent == VoiceCommandIntent.VOLUME_DOWN:
            self._on_hardware_button(ButtonEvent.VOL_DOWN_PRESS)
