"""Camera abstraction package supporting Mac OpenCV and Raspberry Pi Picamera2."""
from visionvoice.camera.base import BaseCamera, get_camera
from visionvoice.camera.mac_camera import MacCamera
from visionvoice.camera.pi_camera import PiCamera

__all__ = ["BaseCamera", "MacCamera", "PiCamera", "get_camera"]
