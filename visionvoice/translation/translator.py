"""
Text Translation Service for Multilingual Reading.
Wraps deep-translator with caching and offline/network failure fallbacks.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from visionvoice.utils.logging import get_logger
from visionvoice.utils.text import TESS_TO_ISO

logger = get_logger("Translator")


class TextTranslator:
    """Translates recognized text to the user's preferred language."""

    def __init__(self) -> None:
        self._cache: Dict[str, str] = {}

    def translate_sentence(self, sentence: str, source_lang: str, target_lang: str) -> str:
        """
        Translates an individual sentence.
        If source and target languages match, returns original sentence immediately.
        """
        if not sentence or not sentence.strip():
            return ""

        src_iso = TESS_TO_ISO.get(source_lang.lower(), source_lang.lower())
        tgt_iso = TESS_TO_ISO.get(target_lang.lower(), target_lang.lower())

        if src_iso == tgt_iso:
            return sentence

        cache_key = f"{src_iso}->{tgt_iso}:{sentence}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source=src_iso if src_iso != "auto" else "auto", target=tgt_iso)
            translated = translator.translate(sentence)
            self._cache[cache_key] = translated
            logger.debug(f"Translated [{src_iso}->{tgt_iso}]: '{sentence[:30]}...' -> '{translated[:30]}...'")
            return translated
        except Exception as e:
            logger.warning(f"Translation failed ({src_iso}->{tgt_iso}): {e}. Falling back to original text.")
            return sentence

    def translate_sentences(self, sentences: List[str], source_lang: str, target_lang: str) -> List[str]:
        """Translates a batch list of sentences."""
        return [self.translate_sentence(s, source_lang, target_lang) for s in sentences]


_translator_instance: Optional[TextTranslator] = None


def get_translator() -> TextTranslator:
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = TextTranslator()
    return _translator_instance
