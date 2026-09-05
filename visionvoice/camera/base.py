"""
Camera Hardware Abstraction Interface.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from visionvoice.utils.config import get_config
from visionvoice.utils.logging import get_logger

logger = get_logger("BaseCamera")


class BaseCamera(ABC):
    """Abstract interface for hardware camera capture."""

    @abstractmethod
    def start(self) -> bool:
        """Initializes and opens the camera stream."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stops the camera stream and releases hardware resources."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Returns True if the camera device is currently active."""
        pass

    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Captures a live preview frame from the stream (BGR format)."""
        pass

    @abstractmethod
    def capture_high_res(self) -> Optional[np.ndarray]:
        """Captures a high-resolution still image optimized for OCR (BGR format)."""
        pass

    def __enter__(self) -> BaseCamera:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def get_camera(backend: Optional[str] = None) -> BaseCamera:
    """
    Factory function to instantiate the appropriate camera driver
    based on configuration or argument ('mac' or 'pi').
    """
    cfg = get_config()
    target_backend = (backend or cfg.CAMERA_BACKEND).lower()

    if target_backend == "pi":
        from visionvoice.camera.pi_camera import PiCamera
        logger.info("Initializing Raspberry Pi Camera (Picamera2)...")
        return PiCamera(width=cfg.CAMERA_WIDTH, height=cfg.CAMERA_HEIGHT)
    else:
        from visionvoice.camera.mac_camera import MacCamera
        logger.info(f"Initializing Mac Camera (OpenCV index={cfg.CAMERA_INDEX})...")
        return MacCamera(
            camera_index=cfg.CAMERA_INDEX,
            width=cfg.CAMERA_WIDTH,
            height=cfg.CAMERA_HEIGHT
        )
