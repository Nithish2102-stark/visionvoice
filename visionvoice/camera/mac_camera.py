"""
Mac Camera Driver using OpenCV VideoCapture.

Optimized for MacBook built-in FaceTime HD / Continuity cameras.
"""

from __future__ import annotations

import time
import threading
from typing import Optional

import cv2
import numpy as np

from visionvoice.camera.base import BaseCamera
from visionvoice.utils.logging import get_logger

logger = get_logger("MacCamera")


class MacCamera(BaseCamera):
    """OpenCV-based camera driver for macOS development."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        self.camera_index = camera_index
        self.target_width = width
        self.target_height = height

        self._cap: Optional[cv2.VideoCapture] = None

        # RLock is required because methods such as get_frame()
        # call is_opened(), which also uses this lock.
        self._lock = threading.RLock()

        self._is_running = False

    def start(self) -> bool:
        """Open the Mac camera using OpenCV."""

        with self._lock:

            # Already running
            if self._cap is not None and self._cap.isOpened():
                self._is_running = True
                return True

            logger.info(
                f"Opening macOS camera index {self.camera_index} "
                f"(target {self.target_width}x{self.target_height})..."
            )

            # Try AVFoundation first
            self._cap = cv2.VideoCapture(
                self.camera_index,
                cv2.CAP_AVFOUNDATION,
            )

            # Fallback to OpenCV default backend
            if not self._cap.isOpened():
                logger.warning(
                    "AVFoundation camera backend failed. "
                    "Trying default OpenCV backend..."
                )

                self._cap.release()

                self._cap = cv2.VideoCapture(self.camera_index)

            # Camera could not be opened
            if not self._cap.isOpened():
                logger.error(
                    f"Failed to open camera device at index "
                    f"{self.camera_index}"
                )

                self._cap = None
                self._is_running = False

                return False

            # Request resolution
            self._cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                self.target_width,
            )

            self._cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                self.target_height,
            )

            # Try to minimize buffering
            self._cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1,
            )

            actual_w = int(
                self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            actual_h = int(
                self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            logger.info(
                f"Mac Camera opened successfully: "
                f"{actual_w}x{actual_h}"
            )

            # Camera warm-up
            logger.info("Warming up camera...")

            for _ in range(5):
                ret, _ = self._cap.read()

                if not ret:
                    logger.warning(
                        "Camera warm-up frame failed"
                    )

                time.sleep(0.05)

            self._is_running = True

            logger.info("Mac Camera is ready.")

            return True

    def stop(self) -> None:
        """Release the camera."""

        with self._lock:

            self._is_running = False

            if self._cap is not None:

                self._cap.release()

                self._cap = None

                logger.info("Mac Camera released.")

    def is_opened(self) -> bool:
        """Return True if the camera is currently open."""

        with self._lock:

            return (
                self._cap is not None
                and self._cap.isOpened()
                and self._is_running
            )

    def get_frame(self) -> Optional[np.ndarray]:
        """Read the latest camera frame."""

        with self._lock:

            if not self.is_opened():
                return None

            ret, frame = self._cap.read()

            if not ret or frame is None:
                logger.warning(
                    "Failed to grab camera frame"
                )

                return None

            return frame

    def capture_high_res(self) -> Optional[np.ndarray]:
        """
        Capture a fresh high-resolution frame.

        Several frames are flushed first so that the returned
        image is as current as possible.
        """

        with self._lock:

            # Start camera if necessary
            if not self.is_opened():

                if not self.start():
                    return None

            # Flush old buffered frames
            for _ in range(3):

                if self._cap is not None:
                    self._cap.grab()

            # Capture latest frame
            if self._cap is None:
                return None

            ret, frame = self._cap.read()

            if not ret or frame is None:

                logger.error(
                    "Failed to capture high-res frame"
                )

                return None

            logger.info(
                f"Captured high-res still: "
                f"{frame.shape[1]}x{frame.shape[0]}"
            )

            return frame