"""Hardware abstraction package for buttons and status LED."""
from visionvoice.hardware.base import BaseHardware, ButtonEvent, LEDState, get_hardware
from visionvoice.hardware.buttons import ButtonManager
from visionvoice.hardware.led import LEDManager
from visionvoice.hardware.mac_mock import MacMockHardware
from visionvoice.hardware.pi_gpio import PiGPIOHardware

__all__ = [
    "BaseHardware",
    "ButtonEvent",
    "LEDState",
    "get_hardware",
    "ButtonManager",
    "LEDManager",
    "MacMockHardware",
    "PiGPIOHardware",
]
