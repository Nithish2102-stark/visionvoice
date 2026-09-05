"""
State Machine implementation for VisionVoice assistive reader.
Enforces valid transitions and provides event hooks and state recovery.
"""

from __future__ import annotations
import threading
from typing import Callable, Dict, List, Optional, Set
from visionvoice.core.models import DeviceState
from visionvoice.utils.logging import get_logger

logger = get_logger("StateMachine")


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


class StateMachine:
    """Thread-safe Finite State Machine for VisionVoice workflow."""

    VALID_TRANSITIONS: Dict[DeviceState, Set[DeviceState]] = {
        DeviceState.INITIALIZING: {
            DeviceState.WAIT_WAKE_WORD,
            DeviceState.ERROR,
            DeviceState.STOPPED,
        },
        DeviceState.WAIT_WAKE_WORD: {
            DeviceState.ASK_PREFERENCES,
            DeviceState.WAIT_START,
            DeviceState.STOPPED,
            DeviceState.ERROR,
        },
        DeviceState.ASK_PREFERENCES: {
            DeviceState.WAIT_START,
            DeviceState.ALIGNING_PAGE,
            DeviceState.WAIT_WAKE_WORD,
            DeviceState.STOPPED,
            DeviceState.ERROR,
        },
        DeviceState.WAIT_START: {
            DeviceState.ALIGNING_PAGE,
            DeviceState.WAIT_WAKE_WORD,
            DeviceState.STOPPED,
            DeviceState.ERROR,
        },
        DeviceState.ALIGNING_PAGE: {
            DeviceState.PROCESSING_PAGE,
            DeviceState.WAIT_START,
            DeviceState.WAIT_WAKE_WORD,
            DeviceState.STOPPED,
            DeviceState.ERROR,
        },
        DeviceState.PROCESSING_PAGE: {
            DeviceState.READING,
            DeviceState.ALIGNING_PAGE,  # Retry if OCR quality low or no page
            DeviceState.WAIT_START,
            DeviceState.STOPPED,
            DeviceState.ERROR,
        },
        DeviceState.READING: {
            DeviceState.PAUSED,
            DeviceState.PAGE_FINISHED,
            DeviceState.ALIGNING_PAGE,  # Fast next page
            DeviceState.STOPPED,
            DeviceState.ERROR,
        },
        DeviceState.PAUSED: {
            DeviceState.READING,
            DeviceState.ALIGNING_PAGE,
            DeviceState.PAGE_FINISHED,
            DeviceState.STOPPED,
            DeviceState.ERROR,
        },
        DeviceState.PAGE_FINISHED: {
            DeviceState.ALIGNING_PAGE,
            DeviceState.READING,        # Repeat page
            DeviceState.WAIT_START,
            DeviceState.WAIT_WAKE_WORD,
            DeviceState.STOPPED,
            DeviceState.ERROR,
        },
        DeviceState.STOPPED: {
            DeviceState.INITIALIZING,
            DeviceState.WAIT_WAKE_WORD,
        },
        DeviceState.ERROR: {
            DeviceState.INITIALIZING,
            DeviceState.WAIT_WAKE_WORD,
            DeviceState.WAIT_START,
            DeviceState.STOPPED,
        },
    }

    def __init__(self, initial_state: DeviceState = DeviceState.INITIALIZING) -> None:
        self._state = initial_state
        self._lock = threading.RLock()
        self._listeners: List[Callable[[DeviceState, DeviceState], None]] = []
        logger.info(f"StateMachine initialized at state: {self._state.value}")

    @property
    def current_state(self) -> DeviceState:
        with self._lock:
            return self._state

    def add_listener(self, callback: Callable[[DeviceState, DeviceState], None]) -> None:
        """Register a callback for state transition notifications (old_state, new_state)."""
        with self._lock:
            self._listeners.append(callback)

    def transition_to(self, new_state: DeviceState, reason: str = "") -> bool:
        """
        Attempts a transition to new_state.
        Returns True on success, raises StateTransitionError on invalid transition.
        """
        with self._lock:
            old_state = self._state
            if old_state == new_state:
                return True

            allowed = self.VALID_TRANSITIONS.get(old_state, set())
            if new_state not in allowed:
                msg = f"Invalid transition from {old_state.value} to {new_state.value} (reason: {reason})"
                logger.error(msg)
                raise StateTransitionError(msg)

            self._state = new_state
            reason_str = f" [{reason}]" if reason else ""
            logger.info(f"State transition: {old_state.value} -> {new_state.value}{reason_str}")

            # Notify listeners outside the lock to prevent deadlock
            listeners_copy = list(self._listeners)

        for listener in listeners_copy:
            try:
                listener(old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state transition listener callback: {e}", exc_info=True)

        return True

    def reset(self) -> None:
        """Resets the state machine to INITIALIZING."""
        with self._lock:
            self._state = DeviceState.INITIALIZING
            logger.info("StateMachine reset to INITIALIZING")
