"""
Raspberry Pi 4 Camera Driver using Picamera2 and Camera Module 3.
Features continuous autofocus and hardware-accelerated still captures.
Safe for Mac environment (guarded imports).
"""

from __future__ import annotations
import threading
from typing import Optional
import numpy as np
from visionvoice.camera.base import BaseCamera
from visionvoice.utils.logging import get_logger

logger = get_logger("PiCamera")


class PiCamera(BaseCamera):
    """Raspberry Pi Camera Module 3 driver powered by Picamera2."""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        self.width = width
        self.height = height
        self._picam2 = None
        self._lock = threading.Lock()
        self._is_running = False

    def _ensure_picamera2(self) -> bool:
        """Dynamically imports Picamera2 library."""
        try:
            from picamera2 import Picamera2  # type: ignore
            if self._picam2 is None:
                self._picam2 = Picamera2()
            return True
        except ImportError:
            logger.error(
                "picamera2 is not installed. This driver is designed for Raspberry Pi OS (Bullseye/Bookworm). "
                "For macOS testing, use CAMERA_BACKEND=mac."
            )
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Picamera2: {e}")
            return False

    def start(self) -> bool:
        """Starts the Picamera2 video/preview pipeline with continuous autofocus."""
        with self._lock:
            if self._is_running:
                return True

            if not self._ensure_picamera2():
                return False

            try:
                # Configure camera for preview and high-res capture
                camera_config = self._picam2.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"},
                )
                self._picam2.configure(camera_config)
                self._picam2.start()

                # Enable autofocus on Camera Module 3 if supported
                try:
                    from libcamera import controls  # type: ignore
                    self._picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
                    logger.info("Continuous autofocus enabled for Pi Camera Module 3")
                except Exception as e:
                    logger.debug(f"Autofocus control not applicable or unsupported: {e}")

                self._is_running = True
                logger.info(f"Raspberry Pi Camera started at {self.width}x{self.height}")
                return True
            except Exception as e:
                logger.error(f"Failed to start Picamera2 stream: {e}", exc_info=True)
                return False

    def stop(self) -> None:
        """Stops the Picamera2 stream."""
        with self._lock:
            self._is_running = False
            if self._picam2 is not None:
                try:
                    self._picam2.stop()
                    logger.info("Picamera2 stream stopped.")
                except Exception as e:
                    logger.error(f"Error stopping Picamera2: {e}")
                finally:
                    self._picam2 = None

    def is_opened(self) -> bool:
        with self._lock:
            return self._is_running and self._picam2 is not None

    def get_frame(self) -> Optional[np.ndarray]:
        """Captures a live frame as a BGR NumPy array."""
        with self._lock:
            if not self.is_opened():
                return None
            try:
                # picamera2 capture_array returns RGB, convert to BGR for OpenCV consistency
                rgb_frame = self._picam2.capture_array("main")
                if rgb_frame is None:
                    return None
                bgr_frame = rgb_frame[:, :, ::-1].copy()
                return bgr_frame
            except Exception as e:
                logger.warning(f"Error reading frame from Picamera2: {e}")
                return None

    def capture_high_res(self) -> Optional[np.ndarray]:
        """Captures high-resolution still frame."""
        with self._lock:
            if not self.is_opened():
                if not self.start():
                    return None
            try:
                rgb_frame = self._picam2.capture_array("main")
                if rgb_frame is None:
                    return None
                return rgb_frame[:, :, ::-1].copy()
            except Exception as e:
                logger.error(f"High-res capture failed on Picamera2: {e}")
                return None
