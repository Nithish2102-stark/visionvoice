"""
Unit tests for OCR composite quality scorer.
"""

import unittest
from visionvoice.ocr.scorer import OCRScorer


class TestOCRScorer(unittest.TestCase):

    def setUp(self):
        self.scorer = OCRScorer()

    def test_high_quality_clean_text_scoring(self):
        text = (
            "Artificial Intelligence is transforming assistive technology for visually impaired "
            "individuals around the world by providing real-time optical character recognition "
            "and natural speech synthesis."
        )
        words = text.split()
        words_data = [{"text": w, "conf": 92.0, "height": 18} for w in words]

        metrics = self.scorer.score_ocr_output(text, words_data, target_language="eng")
        self.assertGreater(metrics.composite_score, 70.0)
        self.assertGreater(metrics.valid_word_ratio, 0.9)
        self.assertGreater(metrics.word_confidence_avg, 90.0)

    def test_garbage_text_scoring_rejection(self):
        garbage_text = "|||| |||| ~~~~ ```` ^^^^ @@#$$ %^^&"
        words = garbage_text.split()
        words_data = [{"text": w, "conf": 25.0, "height": 8} for w in words]

        metrics = self.scorer.score_ocr_output(garbage_text, words_data, target_language="eng")
        self.assertLess(metrics.composite_score, 30.0)
        self.assertGreater(metrics.symbol_penalty, 0.5)

    def test_empty_text_scoring(self):
        metrics = self.scorer.score_ocr_output("", [])
        self.assertEqual(metrics.composite_score, 0.0)
        self.assertEqual(metrics.total_words, 0)


if __name__ == "__main__":
    unittest.main()
