from __future__ import annotations

from typing import List, Optional, Callable, Dict, Tuple, Any, Union
import time
import threading
import gpiozero


class DigitalInputDevice(gpiozero.DigitalInputDevice):
    def __init__(
            self,
            pin: int,
            name: Optional[str] = None,
            *,
            pull_up: bool = False,
            active_state: Optional[bool] = None,
            bounce_time: Optional[float] = None,
            pin_factory=None,
    ):
        super().__init__(
            pin,
            pull_up=pull_up,
            active_state=active_state,
            bounce_time=bounce_time,
            pin_factory=pin_factory,
        )
        self.name = name or f'Pin{pin}'

    def __str__(self) -> str:
        return f"{self.name}:{int(self.value)}"


class DigitalOutputDevice(gpiozero.DigitalOutputDevice):
    def __init__(self, pin=None, name=None, *, active_high=True, initial_value=False, pin_factory=None):
        super().__init__(pin, active_high=active_high, initial_value=initial_value, pin_factory=pin_factory)
        self.name = name or f'Pin{pin}'

    def __str__(self) -> str:
        return f"{self.name}:{int(self.value)}"


class Inputs:
    def __init__(self, poll_interval: float = 0.05):
        self.inputs: List[DigitalInputDevice] = []
        self.poll_interval = poll_interval
        self._running = False

        self._edge_callbacks: List[Callable[[DigitalInputDevice, int], None]] = []
        self._handlers_attached: bool = False

        self._simul_groups: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(
            self,
            pin: int,
            name: str,
            *,
            pull_up: bool = False,
            active_state: Optional[bool] = None,
            bounce_time: Optional[float] = None,
            pin_factory=None
    ) -> DigitalInputDevice:
        device = DigitalInputDevice(
            pin=pin,
            name=name,
            pull_up=pull_up,
            active_state=active_state,
            bounce_time=bounce_time,
            pin_factory=pin_factory,
        )
        self.inputs.append(device)
        return device

    def get(self, key: Union[int, str]) -> DigitalInputDevice:
        """
        Find a device by Pin Number (int) or Name (str).
        """
        for device in self.inputs:
            if isinstance(key, int):
                # Compare with the integer number of the pin
                if device.pin.number == key:
                    return device
            elif isinstance(key, str):
                if device.name == key:
                    return device

        raise ValueError(f"Input Device not found: {key}")

    def _attach_handlers_once(self):
        if self._handlers_attached:
            return
        for device in self.inputs:
            device.when_activated = lambda d=device: self._handle_edge(d, 1)
            device.when_deactivated = lambda d=device: self._handle_edge(d, 0)

        self._handlers_attached = True

    def on_change(self, callback: Callable[[DigitalInputDevice, int], None]) -> None:
        """
        Register a callback called on every edge:
            callback(device, value)
        """
        self._edge_callbacks.append(callback)
        self._attach_handlers_once()

    def simultaneous_events(
            self,
            callback: Callable[[List[Tuple[str, int]]], None],
            duration: float
    ) -> None:
        """
        Register a new simultaneous-events group.

        For this group:
        - When first edge in window occurs:
            * start window (t0)
            * for t0..t0+duration: collect SEQUENCE of (name, value)
        - After 'duration', call callback(events) once with
            events: List[(name, value)] in time order
        """
        if duration <= 0:
            raise ValueError("duration must be > 0")

        self._attach_handlers_once()

        with self._lock:
            group = {
                "duration": duration,
                "callback": callback,
                "events": [],
                "window_start": None,  # type: Optional[float]
                "window_id": 0,  # type: int
            }
            self._simul_groups.append(group)

    def _handle_edge(self, device: DigitalInputDevice, value: int):
        for cb in self._edge_callbacks:
            cb(device, value)
        self._handle_simultaneous_edge(device, value)

    def _handle_simultaneous_edge(self, device: DigitalInputDevice, value: int):
        now = time.monotonic()
        with self._lock:
            if not self._simul_groups:
                return

            for group in self._simul_groups:
                duration: float = group["duration"]
                if duration <= 0:
                    continue
                window_start: Optional[float] = group["window_start"]
                if window_start is None or (now - window_start) > duration:
                    group["window_start"] = now
                    group["events"] = []
                    group["window_id"] += 1
                    window_id = group["window_id"]
                    timer = threading.Timer(
                        duration,
                        self._flush_simultaneous_events_group,
                        args=(group, window_id),
                    )
                    timer.daemon = True
                    timer.start()

                group["events"].append((device.name, value))

    def _flush_simultaneous_events_group(self, group: Dict[str, Any], window_id: int):
        with self._lock:
            if window_id != group["window_id"]:
                return

            events: List[Tuple[str, int]] = group["events"]
            if not events:
                group["window_start"] = None
                return

            # copy + reset
            events_to_send = list(events)
            group["events"] = []
            group["window_start"] = None
            cb = group["callback"]

        if cb:
            cb(events_to_send)

    def run(self):
        print("Starting input monitor...  (Ctrl+C to stop)")
        self._running = True
        try:
            while self._running:
                states = " | ".join(str(device) for device in self.inputs)
                print(f"\r{states}", end="", flush=True)  # Print on same line
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        for device in self.inputs:
            device.close()
        print("All Input GPIO pins closed.")


class Outputs:
    def __init__(self):
        self.outputs: List[DigitalOutputDevice] = []

    def add(self, device: DigitalOutputDevice) -> DigitalOutputDevice:
        self.outputs.append(device)
        return device

    def get(self, key: Union[int, str]) -> DigitalOutputDevice:
        """
        Find a device by Pin Number (int) or Name (str).
        """
        for device in self.outputs:
            if isinstance(key, int):
                # Check pin number
                if device.pin.number == key:
                    return device
            elif isinstance(key, str):
                # Check name
                if device.name == key:
                    return device

        raise ValueError(f"Output Device not found: {key}")

    def close(self):
        for device in self.outputs:
            device.close()
        print("All Output GPIO pins closed.")


if __name__ == "__main__":

    def edge_callback(device: DigitalInputDevice, value: int):
        print(f"\n[EDGE] {device.name} changed to {value}")

        # Example: Using the 'get' method on Inputs
        # If 'Switch L' is pressed (value 0 because pull_up=True means pressed drives it low usually,
        # but check your wiring. Assuming 1=Active here based on your prev code):
        pass


    def handle_events(events: List[Tuple[str, int]]):
        print("\n[SIMUL] Window:", events)

        # If both switches triggered
        # Note: Depending on wiring (pull_up), pressed might be 0 or 1.
        # Assuming 1 is the trigger state here:
        if ('Switch L', 1) in events and ('Switch R', 1) in events:
            outputs.get('Cylinder 1+').on()
            time.sleep(0.1)
            outputs.get('Cylinder 1+').off()
            time.sleep(1)
            outputs.get('Cylinder 1-').on()
            time.sleep(0.1)
            outputs.get('Cylinder 1-').off()

    inputs = Inputs(poll_interval=0.1)
    outputs = Outputs()
    
    inputs.add(5, "EM", pull_up=True, bounce_time=0.02)
    inputs.add(12, "Switch L", pull_up=True, bounce_time=0.02)
    inputs.add(16, "Switch R", pull_up=True, bounce_time=0.02)
    inputs.add(20, "Area 1", pull_up=True, bounce_time=0.02)
    inputs.add(6, "Area 2", pull_up=True, bounce_time=0.02)
    inputs.add(13, "Check Part", pull_up=True, bounce_time=0.02)
    inputs.add(19, "Cylinder 1 Reed Switch", pull_up=True, bounce_time=0.02)
    inputs.add(21, "Cylinder 2 Reed Switch", pull_up=True, bounce_time=0.02)
    
    outputs.add(DigitalOutputDevice(4, 'Switch Lamp'))
    outputs.add(DigitalOutputDevice(18, 'Buzzer'))
    outputs.add(DigitalOutputDevice(22, 'Cylinder 1+'))
    outputs.add(DigitalOutputDevice(24, 'Cylinder 1-'))
    outputs.add(DigitalOutputDevice(17, 'Cylinder 2+'))
    outputs.add(DigitalOutputDevice(27, 'Cylinder 2-'))
    outputs.add(DigitalOutputDevice(23))
    outputs.add(DigitalOutputDevice(25))

    inputs.on_change(edge_callback)
    inputs.simultaneous_events(handle_events, duration=0.2)


    input_thread = threading.Thread(target=inputs.run, daemon=True)
    input_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        inputs.stop()
        outputs.close()
