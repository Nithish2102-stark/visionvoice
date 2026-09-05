"""
Image Quality Assessment for VisionVoice OCR.
Evaluates sharpness (Laplacian variance), luminance, contrast, overexposure, and underexposure.
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import cv2
import numpy as np
from visionvoice.utils.logging import get_logger

logger = get_logger("ImageQualityChecker")


class ImageQualityStatus(Enum):
    OK = "OK"
    IMAGE_TOO_BLURRY = "IMAGE_TOO_BLURRY"
    IMAGE_TOO_DARK = "IMAGE_TOO_DARK"
    IMAGE_TOO_BRIGHT = "IMAGE_TOO_BRIGHT"
    IMAGE_LOW_CONTRAST = "IMAGE_LOW_CONTRAST"


@dataclass
class QualityMetrics:
    status: ImageQualityStatus
    is_acceptable: bool
    blur_score: float         # Laplacian variance (higher = sharper)
    mean_brightness: float    # 0 - 255
    contrast_std: float       # Standard deviation of luminance
    overexposed_ratio: float  # Percentage of clipped white pixels (>250)
    underexposed_ratio: float # Percentage of clipped black pixels (<15)
    details: Dict[str, Any]


class ImageQualityChecker:
    """Evaluates whether an image has sufficient visual quality for accurate OCR."""

    def __init__(
        self,
        min_blur_score: float = 25.0,
        min_brightness: float = 30.0,
        max_brightness: float = 248.0,
        min_contrast: float = 12.0,
        max_underexposure_ratio: float = 0.65,
    ) -> None:
        self.min_blur_score = min_blur_score
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.max_underexposure_ratio = max_underexposure_ratio

    def evaluate(self, image: np.ndarray) -> QualityMetrics:
        """
        Analyzes image quality and returns diagnostic metrics.
        """
        if image is None or image.size == 0:
            return QualityMetrics(
                status=ImageQualityStatus.IMAGE_TOO_DARK,
                is_acceptable=False,
                blur_score=0.0,
                mean_brightness=0.0,
                contrast_std=0.0,
                overexposed_ratio=0.0,
                underexposed_ratio=1.0,
                details={"error": "Empty or null image"},
            )

        # Convert to grayscale for metric calculations
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 1. Brightness & Contrast
        mean_brightness = float(np.mean(gray))
        contrast_std = float(np.std(gray))

        # 2. Exposure clipping
        total_pixels = float(gray.size)
        overexposed_ratio = float(np.sum(gray >= 252) / total_pixels)
        underexposed_ratio = float(np.sum(gray <= 15) / total_pixels)

        # 3. Sharpness / Blur estimation via Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(laplacian.var())

        details = {
            "blur_score": round(blur_score, 2),
            "mean_brightness": round(mean_brightness, 2),
            "contrast_std": round(contrast_std, 2),
            "overexposed_ratio": round(overexposed_ratio, 3),
            "underexposed_ratio": round(underexposed_ratio, 3),
        }

        # Determine status with priority: Extreme darkness/brightness -> Blur -> Low contrast
        status = ImageQualityStatus.OK
        if mean_brightness < self.min_brightness or underexposed_ratio > self.max_underexposure_ratio:
            status = ImageQualityStatus.IMAGE_TOO_DARK
            logger.warning(f"Image is too dark: brightness = {mean_brightness:.1f}")
        elif mean_brightness > self.max_brightness and contrast_std < 15.0:
            status = ImageQualityStatus.IMAGE_TOO_BRIGHT
            logger.warning(f"Image is too bright/washed out: brightness = {mean_brightness:.1f}")
        elif blur_score < self.min_blur_score:
            status = ImageQualityStatus.IMAGE_TOO_BLURRY
            logger.warning(f"Image is too blurry: laplacian var = {blur_score:.1f} < {self.min_blur_score}")
        elif contrast_std < self.min_contrast:
            status = ImageQualityStatus.IMAGE_LOW_CONTRAST
            logger.warning(f"Image has low contrast: std = {contrast_std:.1f}")

        is_acceptable = status == ImageQualityStatus.OK

        return QualityMetrics(
            status=status,
            is_acceptable=is_acceptable,
            blur_score=blur_score,
            mean_brightness=mean_brightness,
            contrast_std=contrast_std,
            overexposed_ratio=overexposed_ratio,
            underexposed_ratio=underexposed_ratio,
            details=details,
        )
