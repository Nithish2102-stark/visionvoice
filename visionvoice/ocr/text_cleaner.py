"""
Conservative Multilingual Text Cleaner.
Cleans OCR noise and line breaks while strictly preserving Indic scripts,
conjunct characters, numbers, and legitimate punctuation.
"""

from __future__ import annotations
import re
from typing import List


class TextCleaner:
    """Provides non-destructive, conservative text cleaning for OCR output."""

    def __init__(self) -> None:
        pass

    def clean(self, raw_text: str) -> str:
        """
        Cleans raw OCR output:
        - Resolves hyphenated line wraps.
        - Removes repeated garbage artifact lines.
        - Normalizes non-standard whitespace.
        - Removes standalone header/footer page numbers.
        - Preserves all Unicode Indic glyphs and proper sentence punctuation.
        """
        if not raw_text or not raw_text.strip():
            return ""

        text = raw_text

        # 1. Normalize line endings and tabs
        text = re.sub(r"\r\n|\r", "\n", text)
        text = text.replace("\t", " ")

        # 2. Re-join hyphenated line wraps (e.g. "com-\nputer" -> "computer")
        text = re.sub(r"(\b\w+)-\n\s*(\w+\b)", r"\1\2", text)

        # 3. Process line-by-line for header/footer page numbers and isolated artifacts
        lines = text.split("\n")
        cleaned_lines: List[str] = []

        total_lines = len(lines)
        for idx, line in enumerate(lines):
            line_str = line.strip()

            if not line_str:
                cleaned_lines.append("")
                continue

            # Check for header/footer page numbers on first 2 or last 2 lines
            # Matches "12", "- 12 -", "Page 12", "PAGE 45", "12 of 300"
            if idx <= 1 or idx >= total_lines - 2:
                if re.match(r"^[-–—\s]*(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?[-–—\s]*$", line_str, re.IGNORECASE):
                    continue

            # Remove lines that are purely repeated symbols (e.g. "----------", "......", "~~~~~~")
            if re.match(r"^[\W_]{3,}$", line_str):
                continue

            # Remove isolated single non-alphanumeric noise characters on a line (e.g., "|", "~", "'")
            if len(line_str) == 1 and not line_str.isalnum():
                continue

            # Clean duplicate consecutive punctuation inside the line (e.g. "Hello,,,, world" -> "Hello, world")
            line_str = re.sub(r",+", ",", line_str)
            line_str = re.sub(r";+", ";", line_str)
            line_str = re.sub(r":+", ":", line_str)
            line_str = re.sub(r"\|+", "", line_str)
            line_str = re.sub(r"[~^`]+", "", line_str)

            # Normalize multiple spaces
            line_str = re.sub(r" +", " ", line_str)

            cleaned_lines.append(line_str)

        # 4. Collapse 3 or more consecutive newlines to maximum 2 (paragraph break)
        result = "\n".join(cleaned_lines)
        result = re.sub(r"\n{3,}", "\n\n", result)

        return result.strip()
