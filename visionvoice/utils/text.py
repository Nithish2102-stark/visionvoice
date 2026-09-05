"""
Multilingual Text Utilities for VisionVoice.
Provides Unicode script detection, Indian-language sentence segmentation, and text quality analysis.
"""

from __future__ import annotations
import re
import unicodedata
from typing import Dict, List, Tuple

# Unicode Character Ranges for target scripts
UNICODE_RANGES = {
    "eng": (0x0020, 0x007F),        # Basic Latin / ASCII
    "hin": (0x0900, 0x097F),        # Devanagari (Hindi, Marathi, Sanskrit)
    "tam": (0x0B80, 0x0BFF),        # Tamil
    "tel": (0x0C00, 0x0C7F),        # Telugu
    "kan": (0x0C80, 0x0CFF),        # Kannada
    "mal": (0x0D00, 0x0D7F),        # Malayalam
}

# ISO 639-1 map for translation and TTS systems
TESS_TO_ISO = {
    "eng": "en",
    "hin": "hi",
    "tam": "ta",
    "tel": "te",
    "kan": "kn",
    "mal": "ml",
}

ISO_TO_TESS = {v: k for k, v in TESS_TO_ISO.items()}

# Human readable script and language names
LANGUAGE_NAMES = {
    "en": "English",
    "eng": "English",
    "hi": "Hindi",
    "hin": "Hindi",
    "ta": "Tamil",
    "tam": "Tamil",
    "te": "Telugu",
    "tel": "Telugu",
    "kn": "Kannada",
    "kan": "Kannada",
    "ml": "Malayalam",
    "mal": "Malayalam",
}


def detect_script_distribution(text: str) -> Dict[str, float]:
    """
    Computes the proportion of characters belonging to each supported Unicode script.
    Returns a dictionary mapping language code to percentage (0.0 - 1.0).
    """
    counts = {lang: 0 for lang in UNICODE_RANGES}
    total_letters = 0

    for char in text:
        if not char.isalpha():
            continue
        cp = ord(char)
        total_letters += 1
        for lang, (low, high) in UNICODE_RANGES.items():
            if low <= cp <= high:
                counts[lang] += 1
                break

    if total_letters == 0:
        return {lang: 0.0 for lang in UNICODE_RANGES}

    return {lang: count / total_letters for lang, count in counts.items()}


def detect_primary_language(text: str, default_lang: str = "eng") -> str:
    """
    Determines the primary language/script from text using Unicode character block analysis
    with secondary langdetect fallback.
    """
    if not text or not text.strip():
        return default_lang

    dist = detect_script_distribution(text)
    dominant_lang, max_ratio = max(dist.items(), key=lambda item: item[1])

    # If an Indian script or English clearly dominates (> 25% of alphabetic characters)
    if max_ratio >= 0.25:
        return dominant_lang

    # Fallback to langdetect if available
    try:
        from langdetect import detect
        detected_iso = detect(text)
        return ISO_TO_TESS.get(detected_iso, default_lang)
    except Exception:
        return default_lang


def split_into_sentences(text: str) -> List[str]:
    """
    Splits multilingual text into natural sentences.
    Supports Western sentence terminators (. ! ?) as well as Indic dandas (।, ॥),
    new lines, and quotation wrappers without breaking abbreviations or decimal numbers.
    """
    if not text or not text.strip():
        return []

    # Normalize newlines and excessive whitespace
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Protect common abbreviations (e.g., Dr., Mr., etc.)
    abbreviations = [r"Mr\.", r"Mrs\.", r"Ms\.", r"Dr\.", r"Prof\.", r"vs\.", r"e\.g\.", r"i\.e\.", r"etc\.", r"Vol\.", r"No\."]
    protected = text
    for i, abbr in enumerate(abbreviations):
        protected = re.sub(r"\b" + abbr, f"__ABBR_{i}__", protected, flags=re.IGNORECASE)

    # Split on sentence boundaries:
    # 1. Indic danda or double danda (।, ॥)
    # 2. Western punctuation (. ! ?) followed by space, quote, or newline
    # 3. Double newlines (paragraph boundaries)
    sentence_regex = r"(?<=[.!?।॥])\s+(?=[A-Z\u0900-\u0D7F\d\"\'\(])|(?:\n\s*\n)"
    raw_sentences = re.split(sentence_regex, protected)

    cleaned_sentences: List[str] = []
    for raw in raw_sentences:
        # Restore protected abbreviations
        s = raw
        for i, abbr in enumerate(abbreviations):
            token = f"__ABBR_{i}__"
            orig = abbr.replace(r"\.", ".")
            s = s.replace(token, orig)

        # Replace internal single newlines with spaces
        s = re.sub(r"\n+", " ", s).strip()
        if s and len(s) >= 2:
            cleaned_sentences.append(s)

    return cleaned_sentences


def is_script_or_alpha_char(c: str) -> bool:
    """Returns True if character is alphabetic or a valid Unicode combining script mark (matras, halant, pulli)."""
    if c.isalpha():
        return True
    cat = unicodedata.category(c)
    # Marks (Mn = nonspacing, Mc = spacing combining, Me = enclosing) are integral to Indic scripts
    return cat.startswith("M") or cat.startswith("L")


def compute_text_statistics(text: str) -> Dict[str, float]:
    """
    Calculates detailed character, word, and lexical statistics for OCR scoring.
    Correctly recognizes Indic combining marks, matras, and conjuncts as valid script components.
    """
    if not text or not text.strip():
        return {
            "char_count": 0,
            "word_count": 0,
            "alpha_ratio": 0.0,
            "digit_ratio": 0.0,
            "symbol_ratio": 0.0,
            "avg_word_length": 0.0,
            "line_count": 0,
        }

    total_chars = len(text)
    alpha_count = sum(1 for c in text if is_script_or_alpha_char(c))
    digit_count = sum(1 for c in text if c.isdigit())
    space_count = sum(1 for c in text if c.isspace())
    
    # Standard legitimate punctuation across Western and Indic scripts
    standard_punct = set(".,!?:;\"'()[]{}<>-–—/\\@#%&*+=|।॥‘’“”«»_")
    symbol_count = sum(
        1 for c in text
        if not (is_script_or_alpha_char(c) or c.isdigit() or c.isspace() or c in standard_punct)
    )

    words = [w for w in re.split(r"\s+", text) if w]
    word_count = len(words)
    lines = [ln for ln in text.split("\n") if ln.strip()]

    avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0

    return {
        "char_count": total_chars,
        "word_count": word_count,
        "alpha_ratio": alpha_count / total_chars if total_chars > 0 else 0.0,
        "digit_ratio": digit_count / total_chars if total_chars > 0 else 0.0,
        "symbol_ratio": symbol_count / total_chars if total_chars > 0 else 0.0,
        "avg_word_length": avg_word_len,
        "line_count": len(lines),
    }
