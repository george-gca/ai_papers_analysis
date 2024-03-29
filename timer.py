import logging
import time
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

@dataclass
class Timer(ContextDecorator):
    """Time your code using a class, context manager, or decorator"""

    timers: ClassVar[dict[str, float]] = dict()
    name: None | str = None
    text: str = "Elapsed time: {:0.4f} seconds"
    logger: None | Callable[[str], None] = logging.info
    _start_time: None | float = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialization: add timer to dict of timers"""
        if self.name:
            self.timers.setdefault(self.name, 0)

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time only if it is greater than 0.5s
        if self.logger and elapsed_time > 0.5:
            if self.name:
                text = f'{self.name} {self.text.lower()}'
            else:
                text = self.text

            try:
                from colorama import Fore
                self.logger(f'{Fore.RED}{text.format(elapsed_time)}{Fore.RESET}')
            except Exception:
                self.logger(f'{text.format(elapsed_time)}')

        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time

    def __enter__(self) -> "Timer":
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        """Stop the context manager timer"""
        self.stop()
