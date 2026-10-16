"""Pure diagnostic intra-op guard. No training arithmetic may be wrapped."""
import contextlib
import functools
import pickle
import random
import threading
import time
import numpy as np
import torch

_local = threading.local()
_records = []

def rng_bytes():
    """Bit-exact immutable RNG evidence; no draws and no floating-point reduction."""
    n = np.random.get_state()
    def raw(t):
        return (str(t.dtype), tuple(t.shape), t.detach().cpu().contiguous().numpy().tobytes())
    value = (random.getstate(), n[0], n[1].dtype.str, n[1].shape,
             n[1].tobytes(), tuple(n[2:]), raw(torch.get_rng_state()),
             tuple(raw(s) for s in torch.cuda.get_rng_state_all())
             if torch.cuda.is_available() else ())
    return pickle.dumps(value, protocol=4)

@contextlib.contextmanager
def diagnostic_threads(label="diagnostic"):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("diagnostic guard requires the training main thread")
    depth = getattr(_local, "depth", 0)
    if depth:
        _local.depth = depth + 1
        try:
            yield
        finally:
            _local.depth = depth
        return
    native_threads = torch.get_num_threads()
    native_interop = torch.get_num_interop_threads()
    before = rng_bytes()
    started = time.perf_counter()
    _local.depth = 1
    completed = False
    verified = False
    try:
        torch.set_num_threads(1)
        if torch.get_num_threads() != 1:
            raise RuntimeError("diagnostic thread setting failed")
        yield
        completed = True
    finally:
        try:
            torch.set_num_threads(native_threads)
            if (torch.get_num_threads() != native_threads or
                    torch.get_num_interop_threads() != native_interop):
                raise RuntimeError("native thread configuration was not restored")
            if rng_bytes() != before:
                raise RuntimeError("diagnostic operation consumed or changed RNG")
            verified = True
        finally:
            _local.depth = 0
            _records.append({"operation": label, "seconds": time.perf_counter() - started,
                             "native_threads": native_threads, "diagnostic_threads": 1,
                             "native_interop": native_interop, "completed": completed and verified})

def diagnostic_only(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with diagnostic_threads(function.__name__):
            return function(*args, **kwargs)
    return wrapped

def drain_records():
    result = list(_records)
    _records.clear()
    return result
