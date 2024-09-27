import numpy as np
import pywt


def wave_mask(
    signal: np.ndarray,
    rates: list[float],
    wavelet: str = "db1",
    level: int = 2,
    clip_min: float | int | None = None,
    clip_max: float | int | None = None,
    offset: float = 0.0,
) -> np.ndarray:
    """
    Apply wavelet-based masking to input data.

    Args:
    - signal: A numpy array representing the time series signal.
              Shape can be (time_steps,) for a single feature or
              (time_steps, features) for multiple features.
    - rates: List of mask rates for each wavelet level.
    - wavelet: Type of wavelet to use.
    - level: Number of decomposition levels.
    - clip_min: Minimum value for clipping the output signal.
    - clip_max: Maximum value for clipping the output signal.
    - offset: A float value to offset the signal after augmentation.

    Returns:
    - A numpy array of the augmented signal, same shape as the input.
    """
    original_shape = signal.shape
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    time_steps, num_features = signal.shape
    s_mask = np.empty_like(signal, dtype=np.float32)

    for col in range(num_features):
        coeffs = pywt.wavedec(
            signal[:, col], wavelet=wavelet, mode="symmetric", level=level
        )

        S = []
        for i in range(level + 1):
            coeffs_array = coeffs[i]

            m = (
                np.random.uniform(0, 1, coeffs_array.shape)
                < rates[min(i, len(rates) - 1)]
            )
            C = np.where(m, 0, coeffs_array)
            S.append(C)
        s = pywt.waverec(S, wavelet=wavelet, mode="symmetric")
        s_mask[:, col] = s[:time_steps]

    # Apply offset
    s_mask += offset

    # Apply clipping if specified
    if clip_min is not None or clip_max is not None:
        s_mask = np.clip(s_mask, clip_min, clip_max)

    # Restore original shape
    if len(original_shape) == 1:
        s_mask = s_mask.flatten()

    return s_mask


def wave_mix(
    signal1: np.ndarray,
    signal2: np.ndarray,
    rates: list[float],
    wavelet: str = "db1",
    level: int = 2,
    clip_min: float | int | None = None,
    clip_max: float | int | None = None,
    offset: float = 0.0,
) -> np.ndarray:
    """
    Mix two input signals using wavelet transformation.

    Args:
    - signal1, signal2: Numpy arrays representing the time series signals.
                        Shape can be (time_steps,) for a single feature or
                        (time_steps, features) for multiple features.
    - rates: List of mix rates for each wavelet level.
    - wavelet: Type of wavelet to use.
    - level: Number of decomposition levels.
    - clip_min: Minimum value for clipping the output signal.
    - clip_max: Maximum value for clipping the output signal.
    - offset: A float value to offset the signal after augmentation.

    Returns:
    - A numpy array of the mixed signal, same shape as the inputs.
    """
    raise NotImplementedError
