import numpy as np
from PyEMD import EMD

from pdda.core import AugmentationTechnique, SignalType


class STAug(AugmentationTechnique):
    def __init__(self, method: str = "combined"):
        self.method = method

    def augment(self, signal: SignalType, **params) -> np.ndarray:
        """
        Apply STAug augmentation to the input signal.

        Args:
            signal (np.ndarray): Input signal to be augmented.
            **params: Additional parameters for the augmentation.

        Returns:
            np.ndarray: Augmented signal.
        """
        raise NotImplementedError("This technique is temporarily disabled")

    def emd_augmentation(
        self,
        signal: SignalType,
        n_imf: int = 10,
        random_weight_prob: float = 0.5,
        imf_rate: float = 1.0,
    ) -> np.ndarray:
        """
        Apply EMD-based augmentation to the input signal.

        Args:
            signal: A numpy array representing the time series signal.
                    Shape can be (time_steps,) for a single feature or
                    (time_steps, features) for multiple features.
            n_imf: Maximum number of IMFs to compute. Default is 10.
            random_weight_prob: Probability of using random weights instead of uniform weights.
            imf_rate: Proportion of IMFs to use in reconstruction, similar to rate in dominant_shuffle.

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

            # Determine number of IMFs to use
            n_imf_use = max(1, int(len(IMF) * imf_rate))
            IMF = IMF[:n_imf_use]

            # Apply random or uniform weights to IMFs
            if np.random.rand() < random_weight_prob:
                weights = 2 * np.random.rand(len(IMF))
            else:
                weights = np.ones(len(IMF))

            # Reconstruct signal with weighted IMFs
            augmented_signal[:, i] = np.sum(weights[:, np.newaxis] * IMF, axis=0)

        # Restore original shape
        if len(original_shape) == 1:
            augmented_signal = augmented_signal.flatten()

        return augmented_signal

    def mix_augmentation(
        self,
        signal1: SignalType,
        signal2: SignalType,
        alpha: float = 0.5,
        mix_rate: float = 1.0,
    ) -> np.ndarray:
        """
        Apply mixup augmentation to two input signals.

        Args:
            signal1: First input signal. Shape can be (time_steps,) or (time_steps, features).
            signal2: Second input signal. Must have the same shape as signal1.
            alpha: Parameter for Beta distribution to sample mixing coefficient.
            mix_rate: Proportion of the signal to apply mixing. Default is 1.0 (whole signal).

        Returns:
            A numpy array of the augmented signal, same shape as the inputs.
        """
        if signal1.shape != signal2.shape:
            raise ValueError("Input signals must have the same shape.")

        # Sample mixing coefficient
        lam = np.random.beta(alpha, alpha)

        # Determine the portion of the signal to mix
        mix_length = int(signal1.shape[0] * mix_rate)

        # Initialize mixed signal with signal1
        mixed_signal = signal1.copy()

        # Apply mixing to the determined portion
        mixed_signal[:mix_length] = (
            lam * signal1[:mix_length] + (1 - lam) * signal2[:mix_length]
        )

        return mixed_signal
