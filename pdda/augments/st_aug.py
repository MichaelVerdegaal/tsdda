import random

import numpy as np
from PyEMD import EMD

from pdda.core import AugmentationTechnique, SignalType


class STAug(AugmentationTechnique):
    """Spectral and Time Augmentation (STAug)

    This is an augmentation technique that applies augmentations in both the
    frequency and the time domain, via EMD decomposition and mixup.

    References:
        - Implementation: https://github.com/xiyuanzh/STAug
        - Paper: https://arxiv.org/abs/2303.14254
    """

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
        """Apply STAug frequency-domain augmentation

        This part of the technique uses Empirical Mode Decomposition (EMD)

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
        # Input validation
        signal = self._validate_input(signal)

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

    def augment_multi(
        self,
        signal1: SignalType,
        signal2: SignalType,
        signals: list[SignalType],
        n_imf: int = 10,
        random_weight_prob: float = 0.5,
        imf_rate: float = 1.0,
        alpha: float = 0.5,
        mix_rate: float = 1.0,
    ) -> SignalType:
        """Apply STAug time-domain augmentation

        This part of the technique uses mixup augmentation.

        Args:
            signal1: First input signal. Shape can be (time_steps,) or (time_steps, features).
            signal2: Second input signal. Must have the same shape as signal1.
            alpha: Parameter for Beta distribution to sample mixing coefficient.
            mix_rate: Proportion of the signal to apply mixing. Default is 1.0 (whole signal).
            signals: A list of input signals. Each signal can be 1D or 2D numpy array.
            n_imf: Maximum number of IMFs to compute. Default is 10.
            random_weight_prob: Probability of using random weights instead of uniform weights.
            imf_rate: Proportion of IMFs to use in reconstruction, similar to rate in dominant_shuffle.
            alpha: Parameter for time domain augmentation.

        Returns:
            A numpy array of the augmented signal, same shape as the inputs.
            A single augmented signal or a list of augmented signals if num_combinations > 1.
        """
        # Input validation
        signal1 = self._validate_input(signal1)
        signal2 = self._validate_input(signal2)

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

    def augment_staug(
        self,
        signals: list[SignalType],
        n_imf: int = 10,
        random_weight_prob: float = 0.5,
        imf_rate: float = 1.0,
        alpha: float = 0.5,
    ) -> list[SignalType]:
        """Apply both frequency and time domain augmentations to a collection of signals.'

        Signals are first individually augmented, and then randomly combined.

        Args:
            signals: A list of input signals. Each signal can be 1D or 2D numpy array.
            n_imf: Maximum number of IMFs to compute. Default is 10.
            random_weight_prob: Probability of using random weights instead of uniform weights.
            imf_rate: Proportion of IMFs to use in reconstruction, similar to rate in dominant_shuffle.
            alpha: Parameter for time domain augmentation.

        Returns:
            A single augmented signal or a list of augmented signals if num_combinations > 1.
        """
        # Input validation
        signals = [self._validate_input(signal) for signal in signals]

        if len(signals) == 1:
            raise ValueError("Must provide at least two signals to augment.")

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
