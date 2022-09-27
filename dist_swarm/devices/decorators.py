import functools
import time

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