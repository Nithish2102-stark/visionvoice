"""
Mac Mock Hardware Driver.
Simulates GPIO buttons and logs LED status patterns during macOS development.
"""

from __future__ import annotations
from typing import Callable, List
from visionvoice.hardware.base import BaseHardware, ButtonEvent, LEDState
from visionvoice.utils.logging import get_logger

logger = get_logger("MacMockHardware")


class MacMockHardware(BaseHardware):
    """Software simulation of Raspberry Pi GPIO peripherals."""

    def __init__(self) -> None:
        self._led_state = LEDState.OFF
        self._button_callbacks: List[Callable[[ButtonEvent], None]] = []
        self._is_active = False

    def start(self) -> bool:
        self._is_active = True
        logger.info("Mac Mock Hardware initialized. (Physical GPIO simulated in software)")
        return True

    def stop(self) -> None:
        self._is_active = False
        self._led_state = LEDState.OFF
        logger.info("Mac Mock Hardware stopped.")

    def set_led_state(self, state: LEDState) -> None:
        if self._led_state != state:
            self._led_state = state
            logger.info(f"💡 [MOCK LED] State changed to -> {state.value}")

    def register_button_callback(self, callback: Callable[[ButtonEvent], None]) -> None:
        self._button_callbacks.append(callback)

    def trigger_button(self, event: ButtonEvent) -> None:
        """Allows testing button events programmatically or via CLI."""
        logger.info(f"🔘 [MOCK BUTTON] Triggered: {event.value}")
        for cb in self._button_callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Error in mock button callback: {e}")
