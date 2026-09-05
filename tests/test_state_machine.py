"""
Unit tests for StateMachine transitions and callbacks.
"""

import unittest
from visionvoice.core.models import DeviceState
from visionvoice.core.state_machine import StateMachine, StateTransitionError


class TestStateMachine(unittest.TestCase):

    def setUp(self):
        self.sm = StateMachine(DeviceState.INITIALIZING)

    def test_valid_forward_workflow(self):
        self.sm.transition_to(DeviceState.WAIT_WAKE_WORD)
        self.assertEqual(self.sm.current_state, DeviceState.WAIT_WAKE_WORD)

        self.sm.transition_to(DeviceState.ASK_PREFERENCES)
        self.assertEqual(self.sm.current_state, DeviceState.ASK_PREFERENCES)

        self.sm.transition_to(DeviceState.WAIT_START)
        self.assertEqual(self.sm.current_state, DeviceState.WAIT_START)

        self.sm.transition_to(DeviceState.ALIGNING_PAGE)
        self.assertEqual(self.sm.current_state, DeviceState.ALIGNING_PAGE)

        self.sm.transition_to(DeviceState.PROCESSING_PAGE)
        self.assertEqual(self.sm.current_state, DeviceState.PROCESSING_PAGE)

        self.sm.transition_to(DeviceState.READING)
        self.assertEqual(self.sm.current_state, DeviceState.READING)

        self.sm.transition_to(DeviceState.PAUSED)
        self.assertEqual(self.sm.current_state, DeviceState.PAUSED)

        self.sm.transition_to(DeviceState.READING)
        self.assertEqual(self.sm.current_state, DeviceState.READING)

        self.sm.transition_to(DeviceState.PAGE_FINISHED)
        self.assertEqual(self.sm.current_state, DeviceState.PAGE_FINISHED)

        self.sm.transition_to(DeviceState.STOPPED)
        self.assertEqual(self.sm.current_state, DeviceState.STOPPED)

    def test_invalid_transition_raises_error(self):
        # INITIALIZING cannot jump directly to READING
        with self.assertRaises(StateTransitionError):
            self.sm.transition_to(DeviceState.READING)

    def test_transition_listener_notification(self):
        recorded = []
        self.sm.add_listener(lambda old, new: recorded.append((old, new)))

        self.sm.transition_to(DeviceState.WAIT_WAKE_WORD)
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0], (DeviceState.INITIALIZING, DeviceState.WAIT_WAKE_WORD))


if __name__ == "__main__":
    unittest.main()
