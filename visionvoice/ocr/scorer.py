"""
Language-Aware Composite OCR Quality Scorer.
Evaluates candidate OCR results using multi-dimensional lexical, script-specific Unicode,
and confidence metrics rather than relying on raw Tesseract confidence alone.
"""

from __future__ import annotations
import math
import re
from typing import Dict, List, Any
from visionvoice.core.models import OCRQualityMetrics
from visionvoice.utils.text import compute_text_statistics, detect_script_distribution, UNICODE_RANGES
from visionvoice.utils.logging import get_logger

logger = get_logger("OCRScorer")


class OCRScorer:
    """Computes transparent, script-aware holistic quality scores for OCR candidate outputs."""

    def __init__(self) -> None:
        pass

    def score_ocr_output(
        self,
        text: str,
        words_data: List[Dict[str, Any]],
        target_language: str = "eng",
        variant_name: str = "default",
        psm_mode: int = 4,
    ) -> OCRQualityMetrics:
        """
        Calculates composite OCR quality score from text and hierarchical word box metadata.
        Returns OCRQualityMetrics object with 0-100 composite score.
        """
        if not text or not text.strip():
            return OCRQualityMetrics(
                word_confidence_avg=0.0,
                valid_word_ratio=0.0,
                alpha_ratio=0.0,
                symbol_penalty=1.0,
                line_consistency=0.0,
                composite_score=0.0,
                total_words=0,
                total_chars=0,
                raw_details={
                    "reason": "empty_text",
                    "variant": variant_name,
                    "psm": psm_mode,
                    "language": target_language,
                },
            )

        # 1. Word Confidence Metrics
        valid_confidences: List[float] = []
        high_conf_words = 0
        line_heights: List[float] = []

        for word in words_data:
            conf = float(word.get("conf", -1))
            word_str = str(word.get("text", "")).strip()
            if conf >= 0 and word_str:
                valid_confidences.append(conf)
                if conf >= 50.0:
                    high_conf_words += 1
                h = float(word.get("height", 0))
                if h > 0:
                    line_heights.append(h)

        total_words_count = len(valid_confidences)
        avg_conf = sum(valid_confidences) / total_words_count if total_words_count > 0 else 0.0
        pct_reasonable_conf = high_conf_words / total_words_count if total_words_count > 0 else 0.0

        # 2. Text Statistics
        stats = compute_text_statistics(text)
        char_count = stats["char_count"]
        word_count = stats["word_count"]
        symbol_ratio = stats["symbol_ratio"]
        line_count = stats["line_count"]

        # 3. Script-Aware Character Validity
        script_dist = detect_script_distribution(text)
        primary_script_ratio = max(script_dist.values()) if script_dist else 0.0

        # Verify that the target language/script matches the actual recognized Unicode characters
        target_script_match = 0.2  # Low baseline if requested script not found in text
        for lang_code in target_language.split("+"):
            lang_clean = lang_code.strip().lower()
            if lang_clean in script_dist:
                ratio = script_dist[lang_clean]
                if ratio >= 0.30:
                    target_script_match = 1.0
                    break
                elif ratio >= 0.10:
                    target_script_match = max(target_script_match, 0.7)

        # 4. Valid Word Structure (at least one valid Unicode letter/conjunct)
        words = [w for w in re.split(r"\s+", text) if w]
        valid_word_count = 0
        sensible_lengths = 0
        for w in words:
            if any(c.isalpha() for c in w):
                valid_word_count += 1
            if 2 <= len(w) <= 25:
                sensible_lengths += 1

        valid_word_ratio = valid_word_count / len(words) if words else 0.0
        word_length_ratio = sensible_lengths / len(words) if words else 0.0

        # 5. Symbol and Garbage Penalty
        symbol_penalty = min(1.0, symbol_ratio * 2.0)
        garbage_patterns = len(re.findall(r"([^\w\s])\1{2,}", text))
        if garbage_patterns > 0:
            symbol_penalty = min(1.0, symbol_penalty + (garbage_patterns * 0.10))

        # 6. Line Consistency
        line_consistency = 0.5
        if line_count > 0:
            avg_words_per_line = word_count / line_count
            if 3 <= avg_words_per_line <= 20:
                line_consistency = 0.95
            elif avg_words_per_line > 0:
                line_consistency = 0.70

        if len(line_heights) >= 4:
            mean_h = sum(line_heights) / len(line_heights)
            var_h = sum((h - mean_h) ** 2 for h in line_heights) / len(line_heights)
            std_h = math.sqrt(var_h)
            height_consistency = max(0.0, 1.0 - (std_h / (mean_h + 1e-5)))
            line_consistency = (line_consistency * 0.6) + (height_consistency * 0.4)

        # 7. Volume Factor (normalizes short sentences vs multi-paragraph text)
        volume_factor = min(1.0, math.log10(max(1, word_count) + 1) / 1.8)

        # Composite Quality Score (0 - 100):
        # 30% Avg Confidence + 20% High-Confidence Word Ratio + 20% Valid Word Ratio +
        # 15% Script Purity & Target Match + 15% Line Consistency & Structure
        raw_score = (
            (avg_conf * 0.30)
            + (pct_reasonable_conf * 100.0 * 0.20)
            + (valid_word_ratio * 100.0 * 0.20)
            + (primary_script_ratio * target_script_match * 100.0 * 0.15)
            + (line_consistency * word_length_ratio * 100.0 * 0.15)
        )

        composite_score = raw_score * (1.0 - (symbol_penalty * 0.5)) * (0.6 + 0.4 * volume_factor)
        composite_score = max(0.0, min(100.0, composite_score))

        return OCRQualityMetrics(
            word_confidence_avg=round(avg_conf, 2),
            valid_word_ratio=round(valid_word_ratio, 3),
            alpha_ratio=round(primary_script_ratio, 3),
            symbol_penalty=round(symbol_penalty, 3),
            line_consistency=round(line_consistency, 3),
            composite_score=round(composite_score, 2),
            total_words=word_count,
            total_chars=char_count,
            raw_details={
                "language": target_language,
                "variant": variant_name,
                "psm": psm_mode,
                "pct_reasonable_conf": round(pct_reasonable_conf, 3),
                "volume_factor": round(volume_factor, 3),
                "line_count": line_count,
            },
        )
