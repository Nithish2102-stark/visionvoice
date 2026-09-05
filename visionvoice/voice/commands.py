"""
Voice Command Parsing and Intent Extraction.
Maps transcribed spoken utterances to structured VoiceCommandIntents.
"""

from __future__ import annotations
import re
from typing import Tuple, Optional
from visionvoice.core.models import VoiceCommandIntent
from visionvoice.utils.logging import get_logger

logger = get_logger("CommandParser")


class CommandParser:
    """Parses transcribed audio strings into structured intents."""

    # Wake word patterns: "Hey Vision Voice", "VisionVoice", "Okay Vision", etc.
    WAKE_WORD_PATTERN = r"\b(?:hey|ok|okay|hi)?\s*(?:vision\s*voice|visionvoice|vision|voice)\b"

    COMMAND_PATTERNS = [
        (VoiceCommandIntent.START_READING, r"\b(?:start\s*reading|start|begin|read\s*now|read\s*book|read\s*page)\b"),
        (VoiceCommandIntent.PAUSE, r"\b(?:pause|hold|wait|stop\s*talking)\b"),
        (VoiceCommandIntent.RESUME, r"\b(?:resume|continue|keep\s*reading|go\s*on|unpause)\b"),
        (VoiceCommandIntent.REPEAT, r"\b(?:repeat|say\s*again|read\s*again|one\s*more\s*time|once\s*more)\b"),
        (VoiceCommandIntent.NEXT_PAGE, r"\b(?:next\s*page|next|turn\s*page|continue\s*reading|forward)\b"),
        (VoiceCommandIntent.PREVIOUS_PAGE, r"\b(?:previous\s*page|previous|prev|back|go\s*back)\b"),
        (VoiceCommandIntent.STOP_READING, r"\b(?:stop\s*reading|stop|quit|exit|cancel|terminate|shut\s*down)\b"),
        (VoiceCommandIntent.VOLUME_UP, r"\b(?:volume\s*up|louder|increase\s*volume|turn\s*up)\b"),
        (VoiceCommandIntent.VOLUME_DOWN, r"\b(?:volume\s*down|softer|quieter|decrease\s*volume|turn\s*down)\b"),
        (VoiceCommandIntent.PREF_ORIGINAL, r"\b(?:original\s*language|original|source|don'?t\s*translate)\b"),
        (VoiceCommandIntent.PREF_TRANSLATE, r"\b(?:translated\s*language|translate|translated|translation)\b"),
    ]

    LANGUAGE_MAP = {
        "english": "en",
        "tamil": "ta",
        "hindi": "hi",
        "kannada": "kn",
        "telugu": "te",
        "malayalam": "ml",
    }

    def is_wake_word(self, text: str) -> bool:
        """Checks if utterance contains the wake word phrase."""
        if not text:
            return False
        clean = text.lower().strip()
        return bool(re.search(self.WAKE_WORD_PATTERN, clean))

    def parse_command(self, text: str) -> Tuple[VoiceCommandIntent, Optional[str]]:
        """
        Parses an utterance and returns (VoiceCommandIntent, optional_arg).
        """
        if not text:
            return VoiceCommandIntent.UNKNOWN, None

        clean = text.lower().strip()

        # Check for language selection intent
        for lang_name, code in self.LANGUAGE_MAP.items():
            if lang_name in clean:
                return VoiceCommandIntent.SET_LANGUAGE, code

        # Check for matching command regex
        for intent, pattern in self.COMMAND_PATTERNS:
            if re.search(pattern, clean):
                logger.debug(f"Matched voice command intent: {intent.value} from text '{text}'")
                return intent, None

        # Check if it was purely wake word
        if self.is_wake_word(text):
            return VoiceCommandIntent.WAKE_WORD, None

        logger.debug(f"Unrecognized utterance: '{text}'")
        return VoiceCommandIntent.UNKNOWN, None
