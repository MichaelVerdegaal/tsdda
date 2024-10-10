import random

import numpy as np
from PyEMD import EMD

from pdda.core import AugmentationTechnique, SignalType


class STAug(AugmentationTechnique):
    def __init__(self):
        super().__init__()
        self.supports_combination = True

    def augment(
        self,
        signal: SignalType,
        n_imf: int = 10,
        random_weight_prob: float = 0.5,
        imf_rate: float = 1.0,
    ) -> np.ndarray:
        """
        Apply STAug augmentation to the input signal.

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

    def augment_staug(
        self,
        signals: list[SignalType],
        n_imf: int = 10,
        random_weight_prob: float = 0.5,
        imf_rate: float = 1.0,
        alpha: float = 0.5,
    ) -> list[SignalType]:
        """
        Apply both frequency and time domain augmentations to a collection of signals.

        Args:
            signals: A list of input signals. Each signal can be 1D or 2D numpy array.
            n_imf: Maximum number of IMFs to compute. Default is 10.
            random_weight_prob: Probability of using random weights instead of uniform weights.
            imf_rate: Proportion of IMFs to use in reconstruction, similar to rate in dominant_shuffle.
            alpha: Parameter for time domain augmentation.

        Returns:
            A single augmented signal or a list of augmented signals if num_combinations > 1.
        """
        # Apply frequency domain augmentation to each signal
        augmented_signals = [
            self.augment(signal, n_imf, random_weight_prob, imf_rate)
            for signal in signals
        ]

        # Apply time domain augmentation to random pairs
        results = []
        for _ in range(len(signals)):
            signal1, signal2 = random.sample(augmented_signals, 2)
            results.append(self.augment_multi(signal1, signal2, alpha))

        return results
