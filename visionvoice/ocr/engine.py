"""
High-Accuracy Multilingual OCR Engine for VisionVoice Assistive Reader.
Implements a Two-Stage OCR Strategy:
  Stage 1: Fast script/layout probe inspecting Unicode character blocks.
  Stage 2: Targeted high-accuracy OCR evaluating the most effective preprocessing variants and PSMs
           with early-stopping when quality score threshold is achieved.
"""

from __future__ import annotations
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import pytesseract
from PIL import Image

from visionvoice.core.models import OCRResult, OCRVariantResult, OCRQualityMetrics
from visionvoice.ocr.quality_check import ImageQualityChecker, ImageQualityStatus
from visionvoice.ocr.preprocessing import ImagePreprocessor
from visionvoice.ocr.language_detector import LanguageDetector
from visionvoice.ocr.reconstructor import TextReconstructor
from visionvoice.ocr.scorer import OCRScorer
from visionvoice.ocr.text_cleaner import TextCleaner
from visionvoice.ocr.page_detector import PageDetector
from visionvoice.utils.config import get_config
from visionvoice.utils.logging import get_logger
from visionvoice.utils.text import split_into_sentences, LANGUAGE_NAMES

logger = get_logger("OCREngine")


class OCREngine:
    """
    Two-Stage Multilingual OCR Engine for English and Indian Languages
    (Tamil, Kannada, Telugu, Malayalam, Hindi) optimized for accuracy and Raspberry Pi 4 efficiency.
    """

    def __init__(self) -> None:
        self.cfg = get_config()
        self.quality_checker = ImageQualityChecker()
        self.preprocessor = ImagePreprocessor()
        self.language_detector = LanguageDetector()
        self.reconstructor = TextReconstructor()
        self.scorer = OCRScorer()
        self.cleaner = TextCleaner()
        self.page_detector = PageDetector()

        # Configure Tesseract binary and data paths
        if self.cfg.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = self.cfg.TESSERACT_CMD
            logger.info(f"Using custom Tesseract path: {self.cfg.TESSERACT_CMD}")

        if self.cfg.TESSDATA_PREFIX:
            os.environ["TESSDATA_PREFIX"] = self.cfg.TESSDATA_PREFIX
            logger.info(f"Using TESSDATA_PREFIX: {self.cfg.TESSDATA_PREFIX}")

        self._check_tesseract_availability()

    def _check_tesseract_availability(self) -> bool:
        """Verifies if Tesseract executable is accessible."""
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR is available (version: {version})")
            return True
        except Exception as e:
            logger.warning(
                f"Tesseract OCR is not accessible on system PATH. OCR calls may fail until installed: {e}"
            )
            return False

    def process_image(
        self,
        image: np.ndarray,
        quadrilateral: Optional[np.ndarray] = None,
        preferred_language: Optional[str] = None,
        fast_mode: Optional[bool] = None,
    ) -> OCRResult:
        """
        Executes the exact two-stage OCR pipeline:
        1. Page Detection / Perspective Correction (with safe full-frame crop fallback).
        2. Image Quality Assessment (sharpness, exposure, brightness).
        3. Stage 1: Fast Script/Layout Probe to identify actual script present on page.
        4. Unicode Script Analysis & Selection of Targeted Tesseract Language(s).
        5. Stage 2: Targeted High-Accuracy OCR on top preprocessing variants (always including original_gray baseline).
        6. Hierarchical Text Reconstruction (block -> paragraph -> line -> words).
        7. Language-Aware Composite Quality Scoring & Early Stopping.
        8. Conservative Text Cleaning & Indic Sentence Segmentation.
        9. Diagnostics & Artifacts Persistence.
        """
        start_time = time.time()
        if image is None or image.size == 0:
            return self._create_error_result("Empty or invalid image provided")

        is_fast = fast_mode if fast_mode is not None else self.cfg.OCR_FAST_MODE
        max_variants = 2 if is_fast else self.cfg.OCR_MAX_VARIANTS
        max_psms = 1 if is_fast else self.cfg.OCR_MAX_PSM
        early_stop_score = self.cfg.OCR_EARLY_STOP_SCORE
        image_hash = self._compute_image_hash(image)

        # 1. Page Detection & Perspective Rectification (with safe full-frame fallback)
        extracted_page, was_warped = self.page_detector.extract_page_region(image, quadrilateral)
        logger.debug(f"Page extraction complete: was_warped={was_warped}, shape={extracted_page.shape}")

        # 2. Image Quality Check
        quality = self.quality_checker.evaluate(extracted_page)
        if not quality.is_acceptable:
            logger.warning(f"Image quality check failed: {quality.status.value}")
            return self._create_error_result(
                f"Image quality issue: {quality.status.value}",
                image_hash=image_hash,
            )

        # 3. Stage 1: Fast Script / Layout Probe
        # Run probe on base grayscale to identify actual physical script on the page
        base_gray = self.preprocessor.to_grayscale(extracted_page)
        fallback_lang = preferred_language or self.cfg.DEFAULT_LANGUAGE
        detected_script, targeted_languages, probe_ratio = self.language_detector.probe_image_script(
            base_gray, fallback_lang=fallback_lang
        )
        logger.info(f"Stage 1 Result -> Script: '{detected_script}', Targeted Languages: {targeted_languages}")

        # 4. Generate Preprocessing Variants (Always preserving original_gray baseline)
        try:
            variants = self.preprocessor.generate_all_variants(extracted_page, fast_mode=is_fast)
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}", exc_info=True)
            return self._create_error_result(f"Preprocessing error: {e}", image_hash=image_hash)

        # 5. Stage 2: Targeted High-Accuracy OCR with Early Stopping
        candidates: List[OCRVariantResult] = []
        
        # Priority variants order: original_gray is ALWAYS evaluated first as the baseline
        priority_variant_names = ["original_gray", "clahe", "illumination_corrected", "otsu", "adaptive_gaussian"]
        eval_variants = [v for v in priority_variant_names if v in variants][:max_variants]

        # PSM modes: 4 (single column / book) and 6 (uniform block)
        eval_psms = [4, 6][:max_psms]

        early_stopped = False
        for var_name in eval_variants:
            if early_stopped:
                break
            var_img = variants[var_name]
            for psm in eval_psms:
                for lang in targeted_languages:
                    cand = self._run_single_ocr(var_img, var_name, psm, lang)
                    if cand is not None:
                        candidates.append(cand)
                        # Check early stop threshold
                        if cand.metrics.composite_score >= early_stop_score:
                            logger.info(
                                f"Early stopping threshold reached: {cand.metrics.composite_score:.1f} >= {early_stop_score}"
                            )
                            early_stopped = True
                            break
                if early_stopped:
                    break

        # Fallback if no candidate met minimum threshold
        if not candidates or max(c.metrics.composite_score for c in candidates) < self.cfg.OCR_CONFIDENCE_THRESHOLD:
            logger.debug("Testing fallback PSM 3 on baseline grayscale...")
            for psm in [3]:
                for lang in targeted_languages:
                    cand = self._run_single_ocr(variants["original_gray"], "original_gray", psm, lang)
                    if cand is not None:
                        candidates.append(cand)

        if not candidates:
            logger.warning("All OCR variants failed to return recognized text")
            return self._create_error_result("No readable text found on page", image_hash=image_hash)

        # 6. Score and Rank Candidates
        candidates.sort(key=lambda c: c.metrics.composite_score, reverse=True)
        best_candidate = candidates[0]

        logger.info(
            f"Selected Best OCR Candidate -> Variant: '{best_candidate.variant_name}', "
            f"PSM: {best_candidate.psm_mode}, Lang: '{best_candidate.language}', "
            f"Score: {best_candidate.metrics.composite_score:.2f}/100, "
            f"Avg Word Conf: {best_candidate.metrics.word_confidence_avg:.1f}%, "
            f"Words: {best_candidate.metrics.total_words}"
        )

        # 7. Conservative Text Cleanup & Sentence Segmentation
        cleaned_text = self.cleaner.clean(best_candidate.raw_text)
        final_lang = self.language_detector.detect_language(cleaned_text, default_lang=detected_script)
        detected_script_name = LANGUAGE_NAMES.get(final_lang, final_lang.upper())
        sentences = split_into_sentences(cleaned_text)

        logger.info(
            f"Final Detected Language: {detected_script_name} ({final_lang}), "
            f"Sentences: {len(sentences)}, Characters: {len(cleaned_text)}"
        )

        # 8. Persist Artifacts and Diagnostics
        orig_path, proc_path, meta_path = self._save_artifacts(
            image,
            extracted_page,
            variants.get(best_candidate.variant_name, extracted_page),
            best_candidate,
            cleaned_text,
            sentences,
            final_lang,
            image_hash,
            quality.details,
            was_warped,
        )

        elapsed = time.time() - start_time
        logger.info(f"Two-Stage OCR Pipeline completed in {elapsed:.2f}s")

        return OCRResult(
            text=best_candidate.raw_text,
            cleaned_text=cleaned_text,
            sentences=sentences,
            detected_language=final_lang,
            detected_script=detected_script_name,
            composite_score=best_candidate.metrics.composite_score,
            average_confidence=best_candidate.metrics.word_confidence_avg,
            selected_variant=best_candidate.variant_name,
            selected_psm=best_candidate.psm_mode,
            image_hash=image_hash,
            original_image_path=str(orig_path),
            processed_image_path=str(proc_path),
            timestamp=datetime.now(timezone.utc),
            is_valid=True,
        )

    def _run_single_ocr(
        self,
        img: np.ndarray,
        variant_name: str,
        psm: int,
        languages: str,
    ) -> Optional[OCRVariantResult]:
        """Runs Tesseract image_to_data using --oem 1 and hierarchical reconstruction."""
        try:
            pil_img = Image.fromarray(img)
            custom_config = f"--oem 1 --psm {psm}"
            
            # Execute image_to_data for detailed bounding boxes, blocks, lines, and confidences
            data_dict = pytesseract.image_to_data(
                pil_img,
                lang=languages,
                config=custom_config,
                output_type=pytesseract.Output.DICT,
            )

            # Hierarchical reconstruction preserving paragraph and line structure
            structured_text, words_data = self.reconstructor.reconstruct(data_dict)

            if not structured_text or not words_data:
                return None

            metrics = self.scorer.score_ocr_output(
                structured_text,
                words_data,
                target_language=languages,
                variant_name=variant_name,
                psm_mode=psm,
            )

            return OCRVariantResult(
                variant_name=variant_name,
                psm_mode=psm,
                language=languages,
                raw_text=structured_text,
                cleaned_text="",
                metrics=metrics,
                words_data=words_data,
            )
        except Exception as e:
            logger.debug(f"OCR evaluation note for {variant_name} (PSM {psm}, lang={languages}): {e}")
            return None

    def _compute_image_hash(self, image: np.ndarray) -> str:
        """Computes perceptual dHash to prevent duplicate reads."""
        small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
        diff = gray[:, 1:] > gray[:, :-1]
        return hex(int("".join(diff.flatten().astype(int).astype(str)), 2))[2:].zfill(16)

    def _save_artifacts(
        self,
        original_img: np.ndarray,
        cropped_img: np.ndarray,
        processed_img: np.ndarray,
        best_cand: OCRVariantResult,
        cleaned_text: str,
        sentences: List[str],
        detected_lang: str,
        image_hash: str,
        quality_details: Dict[str, Any],
        was_warped: bool,
    ) -> Tuple[Path, Path, Path]:
        """Saves debug captures, processed variants, and JSON diagnostics."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        base_name = f"page_{timestamp_str}_{image_hash[:8]}"

        orig_path = self.cfg.CAPTURES_DIR / f"{base_name}_orig.jpg"
        crop_path = self.cfg.PROCESSED_DIR / f"{base_name}_crop.jpg"
        proc_path = self.cfg.PROCESSED_DIR / f"{base_name}_proc_{best_cand.variant_name}.jpg"
        meta_path = self.cfg.OCR_DIR / f"{base_name}_meta.json"

        # Save images
        cv2.imwrite(str(orig_path), original_img)
        cv2.imwrite(str(crop_path), cropped_img)
        cv2.imwrite(str(proc_path), processed_img)

        # Save JSON metadata diagnostics
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_hash": image_hash,
            "was_perspective_warped": was_warped,
            "image_quality": quality_details,
            "selected_variant": best_cand.variant_name,
            "selected_psm": best_cand.psm_mode,
            "selected_language": best_cand.language,
            "composite_score": best_cand.metrics.composite_score,
            "average_confidence": best_cand.metrics.word_confidence_avg,
            "detected_language": detected_lang,
            "total_sentences": len(sentences),
            "sentences": sentences,
            "cleaned_text": cleaned_text,
            "original_image": str(orig_path),
            "cropped_image": str(crop_path),
            "processed_image": str(proc_path),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return orig_path, proc_path, meta_path

    def _create_error_result(self, error_msg: str, image_hash: str = "") -> OCRResult:
        """Helper to construct an invalid/error OCRResult."""
        return OCRResult(
            text="",
            cleaned_text="",
            sentences=[],
            detected_language="unknown",
            detected_script="Unknown",
            composite_score=0.0,
            average_confidence=0.0,
            selected_variant="none",
            selected_psm=0,
            image_hash=image_hash,
            original_image_path="",
            processed_image_path="",
            timestamp=datetime.now(timezone.utc),
            is_valid=False,
            error_message=error_msg,
        )
