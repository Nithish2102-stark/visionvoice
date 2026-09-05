"""
Comprehensive Multilingual and Robustness Test Suite for VisionVoice OCR.
Evaluates:
1. English page
2. Tamil page
3. Kannada page
4. Telugu page
5. Malayalam page
6. Hindi page
7. Mixed English + Indic page
8. Page with shadows
9. Slightly rotated page
10. Perspective-distorted page
"""

import unittest
import numpy as np
import cv2
from visionvoice.ocr.engine import OCREngine
from visionvoice.ocr.language_detector import LanguageDetector
from visionvoice.ocr.scorer import OCRScorer
from visionvoice.ocr.page_detector import PageDetector
from visionvoice.utils.text import detect_primary_language, split_into_sentences


class TestMultilingualRobustness(unittest.TestCase):

    def setUp(self):
        self.engine = OCREngine()
        self.lang_detector = LanguageDetector()
        self.scorer = OCRScorer()
        self.page_detector = PageDetector()

    def test_01_english_script_detection_and_scoring(self):
        text = (
            "VisionVoice is an advanced voice-controlled assistive reading device. "
            "It transforms printed books into natural speech with high accuracy."
        )
        dominant_script, ratio, dist = self.lang_detector.detect_script(text)
        self.assertEqual(dominant_script, "eng")
        self.assertGreater(ratio, 0.90)

        # Test script-aware scoring
        words_data = [{"text": w, "conf": 90.0, "height": 18} for w in text.split()]
        metrics = self.scorer.score_ocr_output(text, words_data, target_language="eng")
        self.assertGreater(metrics.composite_score, 70.0)

    def test_02_tamil_script_detection_and_scoring(self):
        tamil_text = "பாரதிதாசன் ஒரு சிறந்த தமிழ் கவிஞர் மற்றும் எழுத்தாளர் ஆவார்."
        dominant_script, ratio, dist = self.lang_detector.detect_script(tamil_text)
        self.assertEqual(dominant_script, "tam")
        self.assertGreater(ratio, 0.80)

        # Ensure Indic script is NOT penalized for lacking English dictionary
        words_data = [{"text": w, "conf": 88.0, "height": 20} for w in tamil_text.split()]
        metrics = self.scorer.score_ocr_output(tamil_text, words_data, target_language="tam+eng")
        self.assertGreater(metrics.composite_score, 65.0)

    def test_03_kannada_script_detection_and_scoring(self):
        kannada_text = "ಕನ್ನಡ ಸಾಹಿತ್ಯ ಪರಿಷತ್ತು ಬೆಂಗಳೂರಿನಲ್ಲಿ ಪ್ರಮುಖ ಸಾಂಸ್ಕೃತಿಕ ಕೇಂದ್ರವಾಗಿದೆ."
        dominant_script, ratio, dist = self.lang_detector.detect_script(kannada_text)
        self.assertEqual(dominant_script, "kan")
        self.assertGreater(ratio, 0.80)

        words_data = [{"text": w, "conf": 86.0, "height": 20} for w in kannada_text.split()]
        metrics = self.scorer.score_ocr_output(kannada_text, words_data, target_language="kan+eng")
        self.assertGreater(metrics.composite_score, 65.0)

    def test_04_telugu_script_detection_and_scoring(self):
        telugu_text = "తెలుగు భాష భారతదేశంలో ఆంధ్ర ప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాలలో మాట్లాడబడుతుంది."
        dominant_script, ratio, dist = self.lang_detector.detect_script(telugu_text)
        self.assertEqual(dominant_script, "tel")
        self.assertGreater(ratio, 0.80)

        words_data = [{"text": w, "conf": 87.0, "height": 20} for w in telugu_text.split()]
        metrics = self.scorer.score_ocr_output(telugu_text, words_data, target_language="tel+eng")
        self.assertGreater(metrics.composite_score, 65.0)

    def test_05_malayalam_script_detection_and_scoring(self):
        malayalam_text = "മലയാള സാഹിത്യം കേരളത്തിന്റെ സാംസ്കാരിക പൈതൃകത്തിന്റെ പ്രധാന ഭാഗമാണ്."
        dominant_script, ratio, dist = self.lang_detector.detect_script(malayalam_text)
        self.assertEqual(dominant_script, "mal")
        self.assertGreater(ratio, 0.80)

        words_data = [{"text": w, "conf": 85.0, "height": 20} for w in malayalam_text.split()]
        metrics = self.scorer.score_ocr_output(malayalam_text, words_data, target_language="mal+eng")
        self.assertGreater(metrics.composite_score, 65.0)

    def test_06_hindi_devanagari_script_detection_and_scoring(self):
        hindi_text = "मुंशी प्रेमचंद आधुनिक हिन्दी साहित्य के सबसे प्रमुख कहानीकार और उपन्यासकार हैं।"
        dominant_script, ratio, dist = self.lang_detector.detect_script(hindi_text)
        self.assertEqual(dominant_script, "hin")
        self.assertGreater(ratio, 0.80)

        words_data = [{"text": w, "conf": 91.0, "height": 19} for w in hindi_text.split()]
        metrics = self.scorer.score_ocr_output(hindi_text, words_data, target_language="hin+eng")
        self.assertGreater(metrics.composite_score, 70.0)

    def test_07_mixed_english_indic_bilingual_page(self):
        bilingual_text = (
            "Chapter 1: Indian Literature (भारतीय साहित्य).\n"
            "Premchand wrote in Hindi (हिन्दी) and Urdu."
        )
        dominant_script, ratio, dist = self.lang_detector.detect_script(bilingual_text)
        self.assertIn(dominant_script, ("eng", "hin"))

        # Test targeted language candidate pair
        candidates = self.lang_detector.get_candidate_languages("hin")
        self.assertIn("hin+eng", candidates)

    def test_08_shadowed_page_illumination_correction(self):
        # Create image with artificial shadow gradient across book spine
        img = np.full((600, 800), 220, dtype=np.uint8)
        # Apply shadow on left half (mimicking spine shadow)
        for x in range(400):
            factor = 0.35 + (0.65 * (x / 400.0))
            img[:, x] = np.clip(img[:, x] * factor, 0, 255).astype(np.uint8)

        # Apply illumination normalization
        corrected = self.engine.preprocessor.normalize_illumination(img)
        self.assertEqual(corrected.shape, img.shape)
        # Check that shadow is brightened significantly
        self.assertGreater(float(np.mean(corrected[:, :200])), 120.0)

    def test_09_slightly_rotated_page_detection(self):
        # Create image with a rotated book page quadrilateral
        canvas = np.zeros((800, 800, 3), dtype=np.uint8)
        rect_pts = np.array([[200, 150], [650, 180], [620, 680], [170, 650]], dtype=np.int32)
        cv2.fillPoly(canvas, [rect_pts], (240, 240, 240))

        analysis = self.page_detector.analyze_frame(canvas)
        self.assertIsNotNone(analysis.quadrilateral)

        # Warp perspective
        warped = self.page_detector.warp_perspective(canvas, analysis.quadrilateral)
        self.assertGreater(warped.shape[0], 300)
        self.assertGreater(warped.shape[1], 300)

    def test_10_perspective_distorted_page_fallback(self):
        # When no contour is detected in a cluttered or dark scene, safe crop fallback succeeds
        flat_img = np.full((600, 800, 3), 180, dtype=np.uint8)
        cv2.putText(flat_img, "FULL FRAME FALLBACK TEXT", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)

        extracted, was_warped = self.page_detector.extract_page_region(flat_img, quadrilateral=None)
        # Verify 5% safe border crop without failing
        self.assertEqual(extracted.shape[0], int(600 * 0.90))
        self.assertEqual(extracted.shape[1], int(800 * 0.90))
        self.assertFalse(was_warped)


    def test_11_script_probe_targeted_selection_all_languages(self):
        from unittest.mock import patch

        dummy_img = np.zeros((200, 200), dtype=np.uint8)

        # Mock probe responses for each language
        test_cases = [
            ("eng", "This is a clean English book page text with several standard words.", "eng", ["eng"]),
            ("tam", "பாரதிதாசன் ஒரு சிறந்த தமிழ் கவிஞர் மற்றும் எழுத்தாளர்.", "tam", ["tam+eng", "tam"]),
            ("kan", "ಕನ್ನಡ ಸಾಹಿತ್ಯ ಪರಿಷತ್ತು ಬೆಂಗಳೂರಿನಲ್ಲಿ ಪ್ರಮುಖ ಕೇಂದ್ರವಾಗಿದೆ.", "kan", ["kan+eng", "kan"]),
            ("tel", "తెలుగు భాష భారతదేశంలో ప్రముఖ ప్రాచీన భాషలలో ఒకటి.", "tel", ["tel+eng", "tel"]),
            ("mal", "മലയാള സാഹിത്യം കേരളത്തിന്റെ സാംസ്കാരിക പൈതൃകമാണ്.", "mal", ["mal+eng", "mal"]),
            ("hin", "मुंशी प्रेमचंद हिन्दी साहित्य के प्रसिद्ध लेखक थे।", "hin", ["hin+eng", "hin"]),
        ]

        for probe_lang_code, mock_text, expected_script, expected_candidates in test_cases:
            def side_effect(img, lang=None, config=None):
                if lang == probe_lang_code:
                    return mock_text
                return "?? garbage text"

            with patch("pytesseract.image_to_string", side_effect=side_effect):
                script, candidates, score = self.lang_detector.probe_image_script(dummy_img, fallback_lang=probe_lang_code)
                self.assertEqual(script, expected_script, f"Failed for {probe_lang_code}")
                self.assertEqual(candidates, expected_candidates, f"Failed candidates for {probe_lang_code}")
                self.assertGreater(score, 0.30)


if __name__ == "__main__":
    unittest.main()
