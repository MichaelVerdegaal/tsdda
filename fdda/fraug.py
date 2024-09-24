import numpy as np


def frequency_masking(
    signal: np.ndarray,
    mask_rate: float = 0.2,
    clip_min: float | int | None = None,
    clip_max: float | int | None = None,
    offset: float = 0.0,
) -> np.ndarray:
    """
    Apply frequency masking to the input signal.

    This function converts the input signal to the frequency domain,
    randomly masks some frequency components, and then converts
    the signal back to the time domain. It treats the last 'forecast_horizon'
    points as the forecasting horizon.

    Args:
        signal: A 1D numpy array representing the time series signal.
        mask_rate: The rate at which frequency components are masked. Should be
         between 0 and 1. Default is 0.5.
        clip_min: Clip signal to a minimum value. If None then nothing happens.
        clip_max: Clip signal to a maximum value. If None then nothing happens.
        offset: A float value to offset the signal after augmentation. Default is 0.0
         (no offset).

    Returns:
        A 1D numpy array of the augmented signal, same length as the input.

    Raises:
        ValueError: If the input signal is not a 1D numpy array, if
                    forecast_horizon is invalid, or if mask_rate is not
                    between 0 and 1.
    """
    if signal.ndim != 1:
        raise ValueError("Input signal must be a 1D numpy array.")
    if not 0 <= mask_rate <= 1:
        raise ValueError("mask_rate must be between 0 and 1")

    # Convert to frequency domain
    signal_f = np.fft.rfft(signal)

    # Create random mask
    mask = np.random.rand(len(signal_f)) < mask_rate

    # Apply mask
    masked_signal_f = signal_f * mask

    # Convert back to time domain
    augmented_signal = np.fft.irfft(masked_signal_f, n=len(signal))

    # Apply offset
    augmented_signal += offset

    # Apply clipping if specified
    if clip_min is not None or clip_max is not None:
        augmented_signal = np.clip(augmented_signal, clip_min, clip_max)

    return augmented_signal


def frequency_mixing(
    signal1: np.ndarray,
    signal2: np.ndarray,
    mix_rate: float = 0.5,
    clip_min: float | int | None = None,
    clip_max: float | int | None = None,
    offset: float = 0.0,
) -> np.ndarray:
    """
    Apply frequency mixing to the input signals.

    This function converts both input signals to the frequency domain,
    mixes their frequency components based on the mix_rate, and then converts
    the result back to the time domain. It treats the last 'forecast_horizon'
    points as the forecasting horizon.

    Args:
        signal1: A 1D numpy array representing the first time series signal.
        signal2: A 1D numpy array representing the second time series signal.
        mix_rate: The rate at which frequency components are mixed from signal2.
                  Should be between 0 and 1. Default is 0.5.
        clip_min: Clip signal to a minimum value. If None then nothing happens.
        clip_max: Clip signal to a maximum value. If None then nothing happens.
        offset: A float value to offset the signal after augmentation. Default is 0.0
                (no offset).

    Returns:
        A 1D numpy array of the augmented signal, same length as the input.

    Raises:
        ValueError: If the input signals are not 1D numpy arrays of the same length,
                    if forecast_horizon is invalid, or if mix_rate is not
                    between 0 and 0.5.
    """
    if signal1.ndim != 1 or signal2.ndim != 1:
        raise ValueError("Input signals must be 1D numpy arrays.")
    if len(signal1) != len(signal2):
        raise ValueError("Input signals must have the same length.")
    if not 0 < mix_rate <= 0.5:
        raise ValueError("mix_rate must be between 0 and 0.5")

    # Convert to frequency domain
    signal1_f = np.fft.rfft(signal1)
    signal2_f = np.fft.rfft(signal2)

    # Create mixing masks
    mask1 = np.random.rand(len(signal1_f)) < mix_rate
    mask2 = np.bitwise_invert(mask1)

    # Mix frequency components
    mixed_signal_f = signal1_f * mask1 + signal2_f * mask2

    # Convert back to time domain
    augmented_signal = np.fft.irfft(mixed_signal_f, n=len(signal1))

    # Apply offset
    augmented_signal += offset

    # Apply clipping if specified
    if clip_min is not None or clip_max is not None:
        augmented_signal = np.clip(augmented_signal, clip_min, clip_max)

    return augmented_signal
