import numpy as np
from PyEMD import EMD


def emd_augmentation(
    signal: np.ndarray,
    n_imf: int = 10,
    clip_min: float | int | None = None,
    clip_max: float | int | None = None,
    offset: float = 0.0,
) -> np.ndarray:
    """
    Apply EMD-based augmentation to the input signal.

    Args:
        signal: A numpy array representing the time series signal.
                Shape can be (time_steps,) for a single feature or
                (time_steps, features) for multiple features.
        n_imf: Number of IMFs to compute. Default is 10.
        clip_min: Minimum value for clipping the output signal.
        clip_max: Maximum value for clipping the output signal.
        offset: A float value to offset the signal after augmentation.

    Returns:
        A numpy array of the augmented signal, same shape as the input.
    """
    original_shape = signal.shape
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    time_steps, features = signal.shape

    # Perform EMD and augmentation for each feature
    augmented_signal = np.zeros_like(signal)
    for i in range(features):
        s = signal[:, i]
        IMF = EMD().emd(s, max_imf=n_imf)

        # If EMD fails to decompose, use the original signal
        if len(IMF) == 0:
            augmented_signal[:, i] = s
            continue

        # Apply random weights to IMFs
        if np.random.rand() >= 0.5:
            weights = 2 * np.random.rand(len(IMF))
        else:
            weights = np.ones(len(IMF))

        # Reconstruct signal with weighted IMFs
        augmented_signal[:, i] = np.sum(weights[:, np.newaxis] * IMF, axis=0)

    # Apply offset
    augmented_signal += offset

    # Apply clipping if specified
    if clip_min is not None or clip_max is not None:
        augmented_signal = np.clip(augmented_signal, clip_min, clip_max)

    # Restore original shape
    if len(original_shape) == 1:
        augmented_signal = augmented_signal.flatten()

    return augmented_signal


def mix_augmentation(
    signal1: np.ndarray,
    signal2: np.ndarray,
    alpha: float = 0.5,
    clip_min: float | int | None = None,
    clip_max: float | int | None = None,
    offset: float = 0.0,
) -> np.ndarray:
    """
    Apply mixup augmentation to two input signals.

    Args:
        signal1: First input signal. Shape can be (time_steps,) or (time_steps, features).
        signal2: Second input signal. Must have the same shape as signal1.
        alpha: Parameter for Beta distribution to sample mixing coefficient.
        clip_min: Minimum value for clipping the output signal.
        clip_max: Maximum value for clipping the output signal.
        offset: A float value to offset the signal after augmentation.

    Returns:
        A numpy array of the augmented signal, same shape as the inputs.
    """
    if signal1.shape != signal2.shape:
        raise ValueError("Input signals must have the same shape.")

    # Sample mixing coefficient
    lam = np.random.beta(alpha, alpha)

    # Mix signals
    mixed_signal = lam * signal1 + (1 - lam) * signal2

    # Apply offset
    mixed_signal += offset

    # Apply clipping if specified
    if clip_min is not None or clip_max is not None:
        mixed_signal = np.clip(mixed_signal, clip_min, clip_max)

    return mixed_signal
