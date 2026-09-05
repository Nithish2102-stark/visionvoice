"""
Page Detector and Alignment System using OpenCV.
Detects book page boundaries, calculates perspective transformations, tracks temporal stability,
and provides safe fallback cropping when boundaries are partially occluded or curved.
"""

from __future__ import annotations
from typing import Optional, Tuple, List
import cv2
import numpy as np
from visionvoice.core.models import PageDetectionStatus, PageFrameAnalysis
from visionvoice.utils.config import get_config
from visionvoice.utils.logging import get_logger

logger = get_logger("PageDetector")


class PageDetector:
    """
    Analyzes camera frames for book page contours, checks stability,
    and performs 4-point perspective dewarping with safe fallback cropping.
    """

    def __init__(
        self,
        min_area_ratio: Optional[float] = None,
        stability_threshold: Optional[float] = None,
        required_stable_frames: Optional[int] = None,
    ) -> None:
        cfg = get_config()
        self.min_area_ratio = min_area_ratio if min_area_ratio is not None else cfg.PAGE_MIN_AREA_RATIO
        self.stability_threshold = stability_threshold if stability_threshold is not None else cfg.PAGE_MAX_MOVEMENT_THRESHOLD
        self.required_stable_frames = required_stable_frames if required_stable_frames is not None else cfg.PAGE_STABILITY_FRAMES

        self._previous_corners: Optional[np.ndarray] = None
        self._stable_frames_count: int = 0
        logger.info(
            f"PageDetector initialized: min_area_ratio={self.min_area_ratio}, "
            f"stability_threshold={self.stability_threshold}, "
            f"required_stable_frames={self.required_stable_frames}"
        )

    def reset_stability(self) -> None:
        """Resets the consecutive stable frames counter."""
        self._previous_corners = None
        self._stable_frames_count = 0

    def analyze_frame(self, frame: np.ndarray, draw_overlay: bool = True) -> PageFrameAnalysis:
        """
        Processes a single camera frame to detect page contour, compute stability,
        and classify status as PAGE_NOT_FOUND, PAGE_DETECTED, or PAGE_STABLE.
        """
        if frame is None or frame.size == 0:
            return PageFrameAnalysis(status=PageDetectionStatus.PAGE_NOT_FOUND)

        h, w = frame.shape[:2]
        frame_area = float(h * w)

        # Downscale for fast edge detection and contour search
        scale = 640.0 / max(h, w) if max(h, w) > 640 else 1.0
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        quadrilateral = self._find_page_quadrilateral(small_frame)
        annotated = frame.copy() if draw_overlay else None

        if quadrilateral is None:
            self.reset_stability()
            if annotated is not None:
                self._draw_status_overlay(annotated, "SEARCHING FOR BOOK PAGE...", (0, 0, 255))
            return PageFrameAnalysis(
                status=PageDetectionStatus.PAGE_NOT_FOUND,
                annotated_frame=annotated,
            )

        # Scale quadrilateral back to original frame dimensions
        orig_quad = (quadrilateral / scale).astype(np.float32)
        quad_area = cv2.contourArea(orig_quad)
        area_ratio = quad_area / frame_area

        if area_ratio < self.min_area_ratio:
            self.reset_stability()
            if annotated is not None:
                self._draw_status_overlay(annotated, f"PAGE TOO SMALL ({area_ratio*100:.1f}%)", (0, 165, 255))
            return PageFrameAnalysis(
                status=PageDetectionStatus.PAGE_NOT_FOUND,
                area_ratio=area_ratio,
                annotated_frame=annotated,
            )

        # Stability tracking across consecutive frames
        corners_ordered = self.order_points(orig_quad.reshape(4, 2))

        if self._previous_corners is None:
            self._stable_frames_count = 1
            movement_score = 0.0
        else:
            movement_score = self._compute_corner_movement(corners_ordered, (w, h))
            if movement_score <= self.stability_threshold:
                self._stable_frames_count += 1
            else:
                self._stable_frames_count = max(0, self._stable_frames_count - 1)

        self._previous_corners = corners_ordered

        if self._stable_frames_count >= self.required_stable_frames:
            status = PageDetectionStatus.PAGE_STABLE
            color = (0, 255, 0)  # Green
            label = f"PAGE STABLE ({self._stable_frames_count}/{self.required_stable_frames})"
        else:
            status = PageDetectionStatus.PAGE_DETECTED
            color = (0, 255, 255)  # Yellow
            label = f"HOLD STILL ({self._stable_frames_count}/{self.required_stable_frames})"

        if annotated is not None:
            pts = corners_ordered.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=3)
            for pt in corners_ordered:
                cv2.circle(annotated, (int(pt[0]), int(pt[1])), 6, color, -1)
            self._draw_status_overlay(annotated, label, color)

        return PageFrameAnalysis(
            status=status,
            quadrilateral=corners_ordered,
            area_ratio=area_ratio,
            stability_score=movement_score,
            stable_frames_count=self._stable_frames_count,
            annotated_frame=annotated,
        )

    def _find_page_quadrilateral(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Finds the most prominent quadrilateral book/page contour."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Multi-strategy edge extraction
        edges1 = cv2.Canny(blurred, 30, 120)
        edges2 = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        combined_edges = cv2.bitwise_or(edges1, edges2)

        # Morphological dilation to close gaps in page edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(combined_edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
        img_area = float(img.shape[0] * img.shape[1])

        for c in contours:
            area = cv2.contourArea(c)
            if area < img_area * 0.10:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.025 * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2)
                rect = cv2.minAreaRect(approx)
                rw, rh = rect[1]
                if rw > 0 and rh > 0:
                    ar = max(rw, rh) / min(rw, rh)
                    if 1.0 <= ar <= 2.8:
                        return pts

        # Fallback to convex hull approximation
        if contours:
            hull = cv2.convexHull(contours[0])
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.04 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2)

        return None

    def _compute_corner_movement(self, current_corners: np.ndarray, frame_dim: Tuple[int, int]) -> float:
        """Calculates normalized average Euclidean displacement between consecutive corner positions."""
        if self._previous_corners is None:
            return 1.0

        w, h = frame_dim
        diag = np.sqrt(w * w + h * h)
        if diag == 0:
            return 1.0

        dist = np.linalg.norm(current_corners - self._previous_corners, axis=1)
        avg_movement = float(np.mean(dist) / diag)
        return avg_movement

    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """
        Orders coordinates in clockwise order:
        top-left, top-right, bottom-right, bottom-left.
        """
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def warp_perspective(self, image: np.ndarray, quad_pts: np.ndarray) -> np.ndarray:
        """
        Applies a 4-point perspective transform to extract a rectified, flat page image.
        """
        rect = self.order_points(quad_pts)
        (tl, tr, br, bl) = rect

        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        if max_width < 50 or max_height < 50:
            return image

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        transform_matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, transform_matrix, (max_width, max_height), flags=cv2.INTER_CUBIC)
        return warped

    def extract_page_region(self, image: np.ndarray, quadrilateral: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        """
        Extracts rectified page image if quadrilateral is valid, otherwise returns a safe
        central crop (5% margin) to eliminate outer desk clutter without failing OCR.
        Returns: (extracted_image, was_warped)
        """
        if quadrilateral is not None:
            try:
                warped = self.warp_perspective(image, quadrilateral)
                return warped, True
            except Exception as e:
                logger.warning(f"Perspective warp failed, applying safe crop: {e}")

        # Safe central crop fallback (removes 5% borders)
        h, w = image.shape[:2]
        margin_y = int(h * 0.05)
        margin_x = int(w * 0.05)
        safe_crop = image[margin_y:h - margin_y, margin_x:w - margin_x]
        return safe_crop, False

    def _draw_status_overlay(self, img: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
        """Draws semi-transparent HUD banner with status message."""
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.putText(
            img,
            text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
