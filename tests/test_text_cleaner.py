"""
Unit tests for TextCleaner and multilingual sentence segmentation.
"""

import unittest
from visionvoice.ocr.text_cleaner import TextCleaner
from visionvoice.utils.text import (
    split_into_sentences,
    detect_primary_language,
    detect_script_distribution,
    compute_text_statistics,
)


class TestTextCleaner(unittest.TestCase):

    def setUp(self):
        self.cleaner = TextCleaner()

    def test_clean_hyphenated_line_wraps(self):
        raw = "This is a demon-\nstration of hyphenated word recon-\nstruction in OCR."
        cleaned = self.cleaner.clean(raw)
        self.assertIn("demonstration", cleaned)
        self.assertIn("reconstruction", cleaned)

    def test_remove_header_footer_page_numbers(self):
        raw = "12\n\nChapter One\nThe journey begins here.\n\nPage 12"
        cleaned = self.cleaner.clean(raw)
        self.assertNotIn("Page 12", cleaned)
        self.assertTrue(cleaned.startswith("Chapter One"))

    def test_preserve_multilingual_indian_scripts(self):
        tamil_text = "பாரதிதாசன் ஒரு சிறந்த கவிஞர் ஆவார்."
        hindi_text = "यह एक बहुत ही सुंदर पुस्तक है।"
        
        cleaned_tam = self.cleaner.clean(tamil_text)
        cleaned_hin = self.cleaner.clean(hindi_text)
        
        self.assertEqual(cleaned_tam, tamil_text)
        self.assertEqual(cleaned_hin, hindi_text)

    def test_remove_repeated_garbage_symbols(self):
        raw = "Valid Title\n-----------\n^^^^^^^^^^^\nValid content follows here."
        cleaned = self.cleaner.clean(raw)
        self.assertNotIn("-----------", cleaned)
        self.assertNotIn("^^^^^^^^^^^", cleaned)
        self.assertIn("Valid Title", cleaned)
        self.assertIn("Valid content", cleaned)


class TestTextUtilities(unittest.TestCase):

    def test_sentence_segmentation_english(self):
        text = "Hello world! This is Dr. Smith. How are you today? We are testing VisionVoice."
        sentences = split_into_sentences(text)
        self.assertEqual(len(sentences), 4)
        self.assertEqual(sentences[0], "Hello world!")
        self.assertEqual(sentences[1], "This is Dr. Smith.")
        self.assertEqual(sentences[2], "How are you today?")
        self.assertEqual(sentences[3], "We are testing VisionVoice.")

    def test_sentence_segmentation_indic_dandas(self):
        text = "राम एक आदर्श बालक है। वह रोज विद्यालय जाता है॥ क्या आप भी पढ़ते हैं?"
        sentences = split_into_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "राम एक आदर्श बालक है।")
        self.assertEqual(sentences[1], "वह रोज विद्यालय जाता है॥")
        self.assertEqual(sentences[2], "क्या आप भी पढ़ते हैं?")

    def test_script_detection_multilingual(self):
        self.assertEqual(detect_primary_language("This is a simple English book page."), "eng")
        self.assertEqual(detect_primary_language("தமிழ் ஒரு மிகத் தொன்மையான திராவிட மொழியாகும்."), "tam")
        self.assertEqual(detect_primary_language("कबीरदास हिन्दी साहित्य के महान कवि थे।"), "hin")
        self.assertEqual(detect_primary_language("ಕನ್ನಡ ಸಾಹಿತ್ಯವು ಅತ್ಯಂತ ಪುರಾತನವಾಗಿದೆ."), "kan")
        self.assertEqual(detect_primary_language("తెలుగు భాష భారతదేశంలో ప్రముఖమైనది."), "tel")
        self.assertEqual(detect_primary_language("മലയാളം കേരളത്തിലെ പ്രധാന ഭാഷയാണ്."), "mal")

    def test_text_statistics(self):
        text = "VisionVoice is a smart 2.0 reading device!"
        stats = compute_text_statistics(text)
        self.assertGreater(stats["char_count"], 20)
        self.assertEqual(stats["word_count"], 7)
        self.assertGreater(stats["alpha_ratio"], 0.7)


if __name__ == "__main__":
    unittest.main()
