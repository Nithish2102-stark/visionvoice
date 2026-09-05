"""
Advanced Modular Preprocessing Engine for VisionVoice Multilingual OCR.
Generates controlled image variants to handle uneven illumination,
spine shadows, low contrast, paper grain, and small Indian glyphs.
"""

from __future__ import annotations
from typing import Dict, Optional
import cv2
import numpy as np
from visionvoice.utils.logging import get_logger

logger = get_logger("Preprocessing")


class ImagePreprocessor:
    """
    Applies modular, non-destructive image enhancements tailored for printed text
    and complex Indian script conjuncts without over-processing.
    """

    def __init__(self) -> None:
        self.clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))

    def generate_all_variants(self, image: np.ndarray, fast_mode: bool = False) -> Dict[str, np.ndarray]:
        """
        Generates controlled image variants:
        1. original_gray
        2. contrast_enhanced
        3. clahe
        4. illumination_corrected (shadow removed)
        5. mild_denoised
        6. sharpened
        7. otsu
        8. adaptive_gaussian
        9. upscaled_gray
        10. upscaled_threshold
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid or empty image provided to ImagePreprocessor")

        variants: Dict[str, np.ndarray] = {}

        # 1. Base Grayscale
        gray = self.to_grayscale(image)
        variants["original_gray"] = gray

        # 2. Contrast Enhanced (Linear histogram stretching)
        contrast_enhanced = self.enhance_contrast(gray)
        variants["contrast_enhanced"] = contrast_enhanced

        # 3. CLAHE (Local adaptive contrast)
        clahe_img = self.apply_clahe(gray)
        variants["clahe"] = clahe_img

        # 4. Illumination / Spine Shadow Corrected
        illum_corrected = self.normalize_illumination(gray)
        variants["illumination_corrected"] = illum_corrected

        # 5. Otsu Threshold on CLAHE
        otsu_img = self.apply_otsu(clahe_img)
        variants["otsu"] = otsu_img

        # 6. Adaptive Gaussian Threshold on illumination-corrected image
        adaptive_gauss = self.apply_adaptive_gaussian(illum_corrected)
        variants["adaptive_gaussian"] = adaptive_gauss

        if not fast_mode:
            # 7. Mild Denoise (Bilateral filtering preserving font strokes)
            mild_denoised = self.apply_bilateral_filter(clahe_img)
            variants["mild_denoised"] = mild_denoised

            # 8. Mild Sharpened (Unsharp masking)
            sharpened = self.apply_unsharp_mask(illum_corrected, sigma=1.0, strength=1.2)
            variants["sharpened"] = sharpened

            # 9. Upscaled Grayscale (Bicubic 1.5x upscaling for small fonts)
            h, w = gray.shape[:2]
            if w < 2400 and h < 2400:
                upscaled_gray = self.upscale_image(clahe_img, scale=1.5)
                variants["upscaled_gray"] = upscaled_gray

                # 10. Upscaled Thresholded Image
                upscaled_thresh = self.apply_otsu(upscaled_gray)
                variants["upscaled_threshold"] = upscaled_thresh

        logger.debug(f"Generated {len(variants)} controlled preprocessing variants for OCR evaluation")
        return variants

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Safely converts BGR or RGB image to single-channel uint8 grayscale."""
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """Applies linear contrast stretching to span 0-255 range based on 1st and 99th percentiles."""
        p1, p99 = np.percentile(gray, (1, 99))
        if p99 > p1:
            stretched = np.clip((gray.astype(np.float32) - p1) * (255.0 / (p99 - p1)), 0, 255)
            return stretched.astype(np.uint8)
        return gray

    def normalize_illumination(self, gray: np.ndarray) -> np.ndarray:
        """
        Removes uneven page lighting and book-spine shadows via morphological background estimation
        and division normalization.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
        background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
        background = cv2.GaussianBlur(background, (41, 41), 0)

        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = np.divide(gray.astype(np.float32), background.astype(np.float32) + 1.0) * 255.0
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        return normalized

    def apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization."""
        return self.clahe.apply(gray)

    def apply_otsu(self, gray: np.ndarray) -> np.ndarray:
        """Applies Otsu's global thresholding with Gaussian pre-filter."""
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def apply_adaptive_gaussian(self, gray: np.ndarray, block_size: int = 25, c: int = 10) -> np.ndarray:
        """Applies Adaptive Gaussian local thresholding."""
        if block_size % 2 == 0:
            block_size += 1
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
        )

    def apply_unsharp_mask(self, gray: np.ndarray, sigma: float = 1.0, strength: float = 1.2) -> np.ndarray:
        """Sharpens text glyphs using mild unsharp masking without over-sharpening artifacts."""
        blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
        sharpened = cv2.addWeighted(gray, 1.0 + strength, blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def apply_bilateral_filter(self, gray: np.ndarray) -> np.ndarray:
        """Smooths paper grain while preserving sharp character edge gradients."""
        return cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)

    def upscale_image(self, gray: np.ndarray, scale: float = 1.5) -> np.ndarray:
        """Upscales image using bicubic interpolation for small printed text."""
        return cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
