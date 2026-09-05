"""
Language and Script Detector for VisionVoice OCR.
Performs lightweight Stage-1 script probe, Unicode character block analysis,
and targeted Tesseract language selection.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pytesseract
from PIL import Image

from visionvoice.utils.text import UNICODE_RANGES, TESS_TO_ISO, ISO_TO_TESS, LANGUAGE_NAMES, is_script_or_alpha_char
from visionvoice.utils.logging import get_logger

logger = get_logger("LanguageDetector")


class LanguageDetector:
    """
    Detects script and language from text and images, generating targeted Tesseract candidates.
    Implements genuine lightweight Stage-1 script probe analyzing Unicode character blocks.
    """

    # Targeted language pairs mapping detected script to Stage-2 OCR models
    TARGETED_LANGUAGE_PAIRS: Dict[str, List[str]] = {
        "eng": ["eng"],
        "hin": ["hin+eng", "hin"],
        "tam": ["tam+eng", "tam"],
        "tel": ["tel+eng", "tel"],
        "kan": ["kan+eng", "kan"],
        "mal": ["mal+eng", "mal"],
    }

    # Controlled set of probe languages
    PROBE_LANGUAGES: List[str] = ["eng", "hin", "tam", "kan", "tel", "mal"]

    def detect_script(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Determines the dominant script based on Unicode block analysis.
        Returns: (dominant_language_code, dominant_ratio, full_distribution)
        """
        if not text or not text.strip():
            return "eng", 0.0, {k: 0.0 for k in UNICODE_RANGES}

        counts = {lang: 0 for lang in UNICODE_RANGES}
        total_valid_chars = 0

        for char in text:
            if not is_script_or_alpha_char(char):
                continue
            cp = ord(char)
            total_valid_chars += 1
            for lang, (low, high) in UNICODE_RANGES.items():
                if low <= cp <= high:
                    counts[lang] += 1
                    break

        if total_valid_chars == 0:
            return "eng", 0.0, {k: 0.0 for k in UNICODE_RANGES}

        dist = {lang: count / total_valid_chars for lang, count in counts.items()}
        dominant_lang, dominant_ratio = max(dist.items(), key=lambda item: item[1])

        return dominant_lang, dominant_ratio, dist

    def detect_language(self, text: str, default_lang: str = "eng") -> str:
        """
        Identifies the primary language code using Unicode blocks as primary signal
        and langdetect as fallback for Latin script.
        """
        if not text or not text.strip():
            return default_lang

        dominant_lang, ratio, dist = self.detect_script(text)

        # If an Indic script clearly dominates
        if ratio >= 0.20 and dominant_lang != "eng":
            return dominant_lang

        # If Latin script dominates, use langdetect to confirm English
        if dist.get("eng", 0.0) >= 0.40:
            try:
                from langdetect import detect
                iso_code = detect(text)
                return ISO_TO_TESS.get(iso_code, "eng")
            except Exception:
                return "eng"

        return dominant_lang if ratio > 0.10 else default_lang

    def _count_script_characters(self, text: str, script_code: str) -> Tuple[int, int, float]:
        """
        Counts characters strictly belonging to a specific script's Unicode range.
        Returns: (script_char_count, total_valid_chars, script_ratio)
        """
        if not text or script_code not in UNICODE_RANGES:
            return 0, 0, 0.0

        low, high = UNICODE_RANGES[script_code]
        script_chars = 0
        total_valid = 0

        for char in text:
            if not is_script_or_alpha_char(char):
                continue
            total_valid += 1
            cp = ord(char)
            if low <= cp <= high:
                script_chars += 1

        ratio = (script_chars / total_valid) if total_valid > 0 else 0.0
        return script_chars, total_valid, ratio

    def probe_image_script(
        self,
        image: np.ndarray,
        fallback_lang: str = "eng"
    ) -> Tuple[str, List[str], float]:
        """
        Stage-1 Fast Script Probe:
        Uses one fast baseline image and PSM 4 with a small controlled set of OCR probes:
        eng, hin, tam, kan, tel, mal.
        Analyzes returned Unicode characters and selects the script with the strongest evidence.

        Targeted mappings:
          Tamil Unicode      -> tam+eng
          Kannada Unicode    -> kan+eng
          Telugu Unicode     -> tel+eng
          Malayalam Unicode  -> mal+eng
          Devanagari Unicode -> hin+eng
          Latin Unicode      -> eng

        Returns: (detected_script_code, targeted_language_candidates, confidence_ratio)
        """
        # Downscale baseline image for rapid probing (<= 800px)
        h, w = image.shape[:2]
        scale = 800.0 / max(h, w) if max(h, w) > 800 else 1.0
        if scale < 1.0:
            probe_img = (image if len(image.shape) == 2 else image[:, :, 0])
            probe_pil = Image.fromarray(probe_img).resize((int(w * scale), int(h * scale)))
        else:
            probe_pil = Image.fromarray(image if len(image.shape) == 2 else image[:, :, 0])

        # Prioritize probe order: fallback/preferred first if Indic, else eng first
        fallback_clean = (fallback_lang or "eng").lower()
        probe_order = list(self.PROBE_LANGUAGES)
        if fallback_clean in self.PROBE_LANGUAGES and fallback_clean != "eng":
            probe_order.remove(fallback_clean)
            probe_order.insert(0, fallback_clean)

        probe_evidence: Dict[str, Tuple[int, float]] = {}  # lang -> (script_chars, script_ratio)

        for lang_code in probe_order:
            try:
                probe_text = pytesseract.image_to_string(
                    probe_pil,
                    lang=lang_code,
                    config="--oem 1 --psm 4",
                )
                if not probe_text or not probe_text.strip():
                    continue

                script_chars, total_valid, script_ratio = self._count_script_characters(probe_text, lang_code)
                probe_evidence[lang_code] = (script_chars, script_ratio)

                # High-confidence early termination criteria:
                if lang_code != "eng":
                    # Indic script: 8+ authentic Unicode characters with >= 35% ratio
                    if script_chars >= 8 and script_ratio >= 0.35:
                        logger.info(
                            f"Stage-1 Fast Probe matched Indic script '{lang_code}' "
                            f"(chars={script_chars}, ratio={script_ratio:.2f})"
                        )
                        return lang_code, self.TARGETED_LANGUAGE_PAIRS[lang_code], script_ratio
                else:
                    # English/Latin: 20+ Latin characters with >= 75% ratio
                    if script_chars >= 20 and script_ratio >= 0.75:
                        # Check if any Indic probe already showed evidence; if not, Latin is very strong
                        logger.info(
                            f"Stage-1 Fast Probe identified Latin script (chars={script_chars}, ratio={script_ratio:.2f})"
                        )
                        return "eng", self.TARGETED_LANGUAGE_PAIRS["eng"], script_ratio

            except Exception as e:
                logger.debug(f"Probe pass for {lang_code} note: {e}")

        # If no immediate early exit, evaluate strongest evidence across all tested probes
        if probe_evidence:
            # Rank Indic scripts first if they have authentic characters, as English probe can hallucinate ASCII
            indic_candidates = {
                k: v for k, v in probe_evidence.items()
                if k != "eng" and v[0] >= 5 and v[1] >= 0.20
            }
            if indic_candidates:
                best_indic, (best_chars, best_ratio) = max(
                    indic_candidates.items(),
                    key=lambda item: (item[1][0], item[1][1])
                )
                logger.info(
                    f"Stage-1 Probe selected Indic script '{best_indic}' "
                    f"(chars={best_chars}, ratio={best_ratio:.2f})"
                )
                return best_indic, self.TARGETED_LANGUAGE_PAIRS[best_indic], best_ratio

            # Otherwise check English
            if "eng" in probe_evidence and probe_evidence["eng"][0] >= 10:
                eng_chars, eng_ratio = probe_evidence["eng"]
                logger.info(f"Stage-1 Probe selected 'eng' (chars={eng_chars}, ratio={eng_ratio:.2f})")
                return "eng", self.TARGETED_LANGUAGE_PAIRS["eng"], eng_ratio

        # Controlled fallback
        fallback = fallback_clean if fallback_clean in self.TARGETED_LANGUAGE_PAIRS else "eng"
        candidates = self.TARGETED_LANGUAGE_PAIRS[fallback]
        logger.info(f"Stage-1 Probe fallback: script '{fallback}', candidates {candidates}")
        return fallback, candidates, 0.0

    def get_candidate_languages(self, detected_or_preferred_lang: Optional[str] = None) -> List[str]:
        """
        Returns targeted language strings to query with Tesseract based on detected script.
        """
        lang = (detected_or_preferred_lang or "eng").lower()
        if lang in self.TARGETED_LANGUAGE_PAIRS:
            return self.TARGETED_LANGUAGE_PAIRS[lang]
        
        # Controlled fallback
        return ["eng"]
