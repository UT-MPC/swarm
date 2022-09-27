import functools
from typing_extensions import Self
import time

class BaseDevice():
    def on_discover(self, other: Self) -> None:
        raise NotImplementedError()

    def on_disconnect(self, other: Self) -> None:
        raise NotImplementedError()

    def collaborate(self) -> None:
        raise NotImplementedError()

    # def train(self, dataFrom: Self) -> None:
        # raise NotImplementedError()

    def measure(func):
        @functools.wraps(func)
        def wrapper_measure(*args, **kwargs):
            start_time = time.time()
            func(*args, **kwargs)
            return time.time() - start_time
        return wrapper_measure

    def delay(seconds: float):
        def decorator_delay(func):
            @functools.wraps(func)
            def wrapper_delay(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper_delay
        return decorator_delay