"""
Raspberry Pi 4 Physical GPIO Driver.
Implements button interrupts with hardware pull-ups (GPIO17, GPIO27, GPIO22)
and status LED patterns (GPIO24). Safe for Mac imports.
"""

from __future__ import annotations
import threading
import time
from typing import Callable, List, Optional
from visionvoice.hardware.base import BaseHardware, ButtonEvent, LEDState
from visionvoice.utils.logging import get_logger

logger = get_logger("PiGPIOHardware")


class PiGPIOHardware(BaseHardware):
    """
    Physical Raspberry Pi 4 GPIO Controller.
    GPIO17 = Power / Start button (Active LOW)
    GPIO27 = Volume Up button (Active LOW)
    GPIO22 = Volume Down button (Active LOW)
    GPIO24 = Status LED (Active HIGH)
    """

    def __init__(
        self,
        pin_power: int = 17,
        pin_vol_up: int = 27,
        pin_vol_down: int = 22,
        pin_led: int = 24,
    ) -> None:
        self.pin_power = pin_power
        self.pin_vol_up = pin_vol_up
        self.pin_vol_down = pin_vol_down
        self.pin_led = pin_led

        self._led_state = LEDState.OFF
        self._button_callbacks: List[Callable[[ButtonEvent], None]] = []
        self._gpio = None
        self._is_active = False
        self._led_thread: Optional[threading.Thread] = None

    def _ensure_gpio(self) -> bool:
        """Dynamically imports and initializes RPi.GPIO."""
        try:
            import RPi.GPIO as GPIO  # type: ignore
            self._gpio = GPIO
            return True
        except ImportError:
            logger.error(
                "RPi.GPIO is not available. This module is meant for Raspberry Pi OS. "
                "For macOS development, set PLATFORM=mac."
            )
            return False
        except Exception as e:
            logger.error(f"Failed to import RPi.GPIO: {e}")
            return False

    def start(self) -> bool:
        """Configures Raspberry Pi GPIO pin modes and registers interrupt handlers."""
        if self._is_active:
            return True

        if not self._ensure_gpio():
            return False

        try:
            GPIO = self._gpio
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # Setup buttons with internal pull-up resistors (pressed connects to GND)
            for pin in [self.pin_power, self.pin_vol_up, self.pin_vol_down]:
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

            # Setup LED pin
            GPIO.setup(self.pin_led, GPIO.OUT, initial=GPIO.LOW)

            # Add debounced event detection on falling edges (button press)
            GPIO.add_event_detect(
                self.pin_power, GPIO.FALLING, callback=self._handle_power_pin, bouncetime=300
            )
            GPIO.add_event_detect(
                self.pin_vol_up, GPIO.FALLING, callback=lambda ch: self._dispatch(ButtonEvent.VOL_UP_PRESS), bouncetime=200
            )
            GPIO.add_event_detect(
                self.pin_vol_down, GPIO.FALLING, callback=lambda ch: self._dispatch(ButtonEvent.VOL_DOWN_PRESS), bouncetime=200
            )

            self._is_active = True

            # Start LED worker thread
            self._led_thread = threading.Thread(target=self._led_pattern_worker, daemon=True)
            self._led_thread.start()

            logger.info(
                f"Raspberry Pi GPIO active -> Power: GPIO{self.pin_power}, "
                f"Vol+: GPIO{self.pin_vol_up}, Vol-: GPIO{self.pin_vol_down}, LED: GPIO{self.pin_led}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to configure Pi GPIO: {e}", exc_info=True)
            return False

    def _handle_power_pin(self, channel: int) -> None:
        """Distinguishes between short press and long press on power button."""
        time.sleep(0.05)
        if self._gpio and self._gpio.input(self.pin_power) == self._gpio.LOW:
            # Check duration
            start = time.time()
            while self._gpio.input(self.pin_power) == self._gpio.LOW and (time.time() - start) < 2.0:
                time.sleep(0.05)
            duration = time.time() - start
            if duration >= 1.5:
                self._dispatch(ButtonEvent.POWER_LONG_PRESS)
            else:
                self._dispatch(ButtonEvent.POWER_PRESS)

    def _dispatch(self, event: ButtonEvent) -> None:
        logger.info(f"Hardware button pressed: {event.value}")
        for cb in self._button_callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Error in button callback: {e}")

    def stop(self) -> None:
        """Cleans up GPIO resources."""
        self._is_active = False
        if self._gpio:
            try:
                self._gpio.output(self.pin_led, self._gpio.LOW)
                self._gpio.cleanup()
                logger.info("Raspberry Pi GPIO cleaned up.")
            except Exception as e:
                logger.error(f"Error cleaning up GPIO: {e}")
            finally:
                self._gpio = None

    def set_led_state(self, state: LEDState) -> None:
        self._led_state = state

    def register_button_callback(self, callback: Callable[[ButtonEvent], None]) -> None:
        self._button_callbacks.append(callback)

    def _led_pattern_worker(self) -> None:
        """Background thread controlling status LED blinking patterns."""
        while self._is_active and self._gpio:
            try:
                GPIO = self._gpio
                if self._led_state == LEDState.OFF:
                    GPIO.output(self.pin_led, GPIO.LOW)
                    time.sleep(0.1)
                elif self._led_state == LEDState.SOLID_ON:
                    GPIO.output(self.pin_led, GPIO.HIGH)
                    time.sleep(0.1)
                elif self._led_state == LEDState.FAST_BLINK:
                    GPIO.output(self.pin_led, GPIO.HIGH)
                    time.sleep(0.1)
                    GPIO.output(self.pin_led, GPIO.LOW)
                    time.sleep(0.1)
                elif self._led_state == LEDState.SLOW_PULSE:
                    GPIO.output(self.pin_led, GPIO.HIGH)
                    time.sleep(0.6)
                    GPIO.output(self.pin_led, GPIO.LOW)
                    time.sleep(0.6)
                elif self._led_state == LEDState.ERROR_BLINK:
                    GPIO.output(self.pin_led, GPIO.HIGH)
                    time.sleep(0.05)
                    GPIO.output(self.pin_led, GPIO.LOW)
                    time.sleep(0.05)
            except Exception:
                time.sleep(0.2)
