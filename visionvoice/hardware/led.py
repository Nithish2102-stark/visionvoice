"""
LED Pattern Manager.
"""

from __future__ import annotations
from visionvoice.hardware.base import LEDState
from visionvoice.utils.logging import get_logger

logger = get_logger("LEDManager")


class LEDManager:
    """Helper for managing current LED pattern state."""

    def __init__(self) -> None:
        self._current_state = LEDState.OFF

    @property
    def state(self) -> LEDState:
        return self._current_state

    def set_state(self, state: LEDState) -> None:
        if self._current_state != state:
            self._current_state = state
            logger.debug(f"Status LED pattern changed to: {state.value}")
