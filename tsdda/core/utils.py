import numpy as np
from .types import SignalType


def validate_and_convert_input(signal: SignalType, dtype=None) -> np.ndarray:
    """Validate and convert the input signal to a numpy array if necessary.

    Args:
        signal (SignalType): Input signal (numpy array or list).
        dtype: Desired dtype for the output array. If None, infer from input.

    Returns:
        np.ndarray: Validated and converted input signal.
    """
    if isinstance(signal, list):
        try:
            return np.array(signal, dtype=dtype)
        except Exception as e:
            raise ValueError("Invalid input: Can't convert to numpy array.") from e
    elif isinstance(signal, np.ndarray):
        if dtype is not None and signal.dtype != dtype:
            return signal.astype(dtype)
        return signal
    else:
        raise TypeError("Input must be a list or numpy array.")
