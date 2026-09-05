"""
Hardware Interface Definitions for Buttons and Status LED.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Optional
from visionvoice.utils.config import get_config
from visionvoice.utils.logging import get_logger

logger = get_logger("BaseHardware")


class ButtonEvent(Enum):
    POWER_PRESS = "POWER_PRESS"
    POWER_LONG_PRESS = "POWER_LONG_PRESS"
    VOL_UP_PRESS = "VOL_UP_PRESS"
    VOL_DOWN_PRESS = "VOL_DOWN_PRESS"


class LEDState(Enum):
    OFF = "OFF"
    SOLID_ON = "SOLID_ON"
    FAST_BLINK = "FAST_BLINK"     # Processing / Capturing
    SLOW_PULSE = "SLOW_PULSE"     # Waiting / Listening
    ERROR_BLINK = "ERROR_BLINK"   # Error state


class BaseHardware(ABC):
    """Abstract interface for physical hardware buttons and status LED."""

    @abstractmethod
    def start(self) -> bool:
        """Initializes hardware pins and starts listeners."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Cleans up hardware GPIO pins."""
        pass

    @abstractmethod
    def set_led_state(self, state: LEDState) -> None:
        """Updates the status LED pattern."""
        pass

    @abstractmethod
    def register_button_callback(self, callback: Callable[[ButtonEvent], None]) -> None:
        """Registers callback for hardware button events."""
        pass


def get_hardware(platform: Optional[str] = None) -> BaseHardware:
    """Factory to instantiate Mac Mock or Raspberry Pi GPIO hardware."""
    cfg = get_config()
    target = (platform or cfg.PLATFORM).lower()

    if target == "pi":
        from visionvoice.hardware.pi_gpio import PiGPIOHardware
        logger.info("Initializing Raspberry Pi GPIO Hardware...")
        return PiGPIOHardware(
            pin_power=cfg.PIN_BUTTON_POWER,
            pin_vol_up=cfg.PIN_BUTTON_VOL_UP,
            pin_vol_down=cfg.PIN_BUTTON_VOL_DOWN,
            pin_led=cfg.PIN_LED_STATUS,
        )
    else:
        from visionvoice.hardware.mac_mock import MacMockHardware
        logger.info("Initializing macOS Mock Hardware (simulated GPIO)...")
        return MacMockHardware()
