"""
Unit and Integration tests for Advanced OCR components:
- ImageQualityChecker
- LanguageDetector
- TextReconstructor
- Preprocessing variants
"""

import unittest
import numpy as np
import cv2
from visionvoice.ocr.quality_check import ImageQualityChecker, ImageQualityStatus
from visionvoice.ocr.language_detector import LanguageDetector
from visionvoice.ocr.preprocessing import ImagePreprocessor
from visionvoice.ocr.reconstructor import TextReconstructor


class TestAdvancedOCR(unittest.TestCase):

    def setUp(self):
        self.quality_checker = ImageQualityChecker()
        self.lang_detector = LanguageDetector()
        self.preprocessor = ImagePreprocessor()
        self.reconstructor = TextReconstructor()

    def test_image_quality_sharp_vs_blurry(self):
        # Create a sharp image with high contrast text on standard page background
        sharp_img = np.full((400, 600, 3), 240, dtype=np.uint8)
        cv2.putText(sharp_img, "SHARP CLEAR TEXT 123", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3)

        # Create a blurred image
        blurry_img = cv2.GaussianBlur(sharp_img, (35, 35), 10.0)

        sharp_res = self.quality_checker.evaluate(sharp_img)
        blurry_res = self.quality_checker.evaluate(blurry_img)

        self.assertTrue(sharp_res.is_acceptable)
        self.assertEqual(sharp_res.status, ImageQualityStatus.OK)
        self.assertEqual(blurry_res.status, ImageQualityStatus.IMAGE_TOO_BLURRY)

    def test_image_quality_dark_and_bright(self):
        dark_img = np.full((100, 100, 3), 10, dtype=np.uint8)
        bright_img = np.full((100, 100, 3), 254, dtype=np.uint8)

        dark_res = self.quality_checker.evaluate(dark_img)
        bright_res = self.quality_checker.evaluate(bright_img)

        self.assertEqual(dark_res.status, ImageQualityStatus.IMAGE_TOO_DARK)
        self.assertEqual(bright_res.status, ImageQualityStatus.IMAGE_TOO_BRIGHT)

    def test_language_detector_targeted_candidates(self):
        self.assertEqual(self.lang_detector.get_candidate_languages("tam"), ["tam+eng", "tam"])
        self.assertEqual(self.lang_detector.get_candidate_languages("hin"), ["hin+eng", "hin"])
        self.assertEqual(self.lang_detector.get_candidate_languages("eng"), ["eng"])

    def test_language_detector_script_analysis(self):
        tam_text = "பாரதிதாசன் ஒரு சிறந்த கவிஞர்."
        hin_text = "मुंशी प्रेमचंद एक महान कथाकार थे।"
        eng_text = "This is a standard English paragraph."

        self.assertEqual(self.lang_detector.detect_language(tam_text), "tam")
        self.assertEqual(self.lang_detector.detect_language(hin_text), "hin")
        self.assertEqual(self.lang_detector.detect_language(eng_text), "eng")

    def test_preprocessing_10_variants_generation(self):
        sample_img = np.full((300, 300, 3), 200, dtype=np.uint8)
        cv2.putText(sample_img, "Test", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        variants = self.preprocessor.generate_all_variants(sample_img, fast_mode=False)
        self.assertGreaterEqual(len(variants), 8)
        self.assertIn("original_gray", variants)
        self.assertIn("clahe", variants)
        self.assertIn("illumination_corrected", variants)
        self.assertIn("otsu", variants)
        self.assertIn("adaptive_gaussian", variants)

    def test_hierarchical_text_reconstruction(self):
        data_dict = {
            "text": ["Title", "Word1", "Word2", "", "Par2Word1", "Par2Word2"],
            "conf": ["90", "95", "92", "-1", "88", "91"],
            "block_num": [1, 1, 1, 1, 2, 2],
            "par_num": [1, 1, 1, 1, 1, 1],
            "line_num": [1, 2, 2, 2, 1, 1],
            "word_num": [1, 1, 2, 0, 1, 2],
            "left": [10, 10, 50, 0, 10, 50],
            "top": [10, 30, 30, 0, 80, 80],
            "width": [40, 35, 35, 0, 40, 40],
            "height": [15, 15, 15, 0, 15, 15],
        }

        reconstructed, words_data = self.reconstructor.reconstruct(data_dict)
        self.assertEqual(len(words_data), 5)
        # Verify paragraph separation (\n\n)
        self.assertIn("\n\n", reconstructed)
        # Verify line break between Title (line 1) and Word1 Word2 (line 2)
        self.assertIn("Title\nWord1 Word2", reconstructed)
        self.assertIn("Par2Word1 Par2Word2", reconstructed)


if __name__ == "__main__":
    unittest.main()
