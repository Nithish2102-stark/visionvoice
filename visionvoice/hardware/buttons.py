"""
Button Event Management Helper.
"""

from __future__ import annotations
from typing import Callable, List
from visionvoice.hardware.base import ButtonEvent
from visionvoice.utils.logging import get_logger

logger = get_logger("ButtonManager")


class ButtonManager:
    """Manages button event subscribers."""

    def __init__(self) -> None:
        self._callbacks: List[Callable[[ButtonEvent], None]] = []

    def add_callback(self, cb: Callable[[ButtonEvent], None]) -> None:
        self._callbacks.append(cb)

    def dispatch(self, event: ButtonEvent) -> None:
        logger.debug(f"Dispatching button event: {event.value}")
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Error in button event handler: {e}")
