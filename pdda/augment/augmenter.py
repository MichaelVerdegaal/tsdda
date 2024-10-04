from abc import ABC
import numpy as np
from .wave_aug import wave_mask, wave_mix
from .d_shuffle import dominant_shuffle
from .fr_aug import frequency_masking, frequency_mixing
from .st_aug import emd_augmentation, mix_augmentation
from typing import TypeAlias

SignalType: TypeAlias = np.ndarray


class Augmenter(ABC):
    """Base class for all augmentation methods."""

    @staticmethod
    def transform(
        signal: SignalType,
        clip_min: float | None = None,
        clip_max: float | None = None,
        offset: float = 0.0,
    ) -> np.ndarray:
        """Apply post-processing to the signal.

        Args:
            signal: Input signal.
            clip_min: Minimum value for clipping the output signal.
            clip_max: Maximum value for clipping the output signal.
            offset: A float value to offset the signal after augmentation.
        """
        signal += offset
        if clip_min is not None or clip_max is not None:
            signal = np.clip(signal, clip_min, clip_max)
        return signal


class SingleSigAugmenter(Augmenter):
    """Base class for augmentations that operate on a single signal."""

    @staticmethod
    def st_aug(
        signal: SignalType,
        n_imf: int = 10,
        random_weight_prob: float = 0.5,
        imf_rate: float = 1.0,
    ) -> np.ndarray:
        """Execute STAug method on single signal.

        Args:
            signal: Input signal to be augmented.
            n_imf: Maximum number of IMFs to compute.
            random_weight_prob: Probability of using random weights instead of uniform weights.
            imf_rate: Proportion of IMFs to use in reconstruction.

        Returns:
            The augmented signal.
        """
        signal = emd_augmentation(signal, n_imf, random_weight_prob, imf_rate)
        return signal

    @staticmethod
    def fr_aug(signal: SignalType, mask_rate: float = 0.2) -> np.ndarray:
        """Execute FRAug method on single signal.

        Args:
            signal: Input signal to be augmented.
            mask_rate: Rate of frequency masking.

        Returns:
            The augmented signal.
        """
        signal = frequency_masking(signal, mask_rate)
        return signal

    @staticmethod
    def d_shuffle(signal: SignalType, rate: int) -> np.ndarray:
        """Execute DShuffle method on single signal.

        Args:
            signal: Input signal to be augmented.
            rate: Number of top dominant frequencies to shuffle.

        Returns:
            The augmented signal.
        """
        signal = dominant_shuffle(signal, rate)
        return signal

    @staticmethod
    def wave_aug(
        signal: SignalType, rates: list[float], wavelet: str = "db1", level: int = 2
    ) -> np.ndarray:
        """Execute WaveAug method on single signal.

        Args:
            signal: Input signal to be augmented.
            rates: List of mask rates for each wavelet level.
            wavelet: Type of wavelet to use.
            level: Number of decomposition levels.

        Returns:
            The augmented signal.
        """
        signal = wave_mask(signal, rates, wavelet, level)
        return signal


class DualSignalAugmenter(Augmenter):
    """Base class for augmentations that operate on two signals.

    In case of method chaining, output signal is always combined with
    original second signal.
    """

    @staticmethod
    def st_aug(
        signal1: SignalType,
        signal2: SignalType,
        alpha: float = 0.5,
        mix_rate: float = 1.0,
    ) -> np.ndarray:
        """
        Execute STAug method on two signals.

        Args:
            signal1: The first signal.
            signal2: The second signal.
            alpha: The alpha value. Defaults to 0.5.
            mix_rate: The mix rate. Defaults to 1.0.

        Returns:
            np.ndarray: The augmented signal.
        """
        return mix_augmentation(signal1, signal2, alpha, mix_rate)

    @staticmethod
    def fr_aug(
        signal1: SignalType, signal2: SignalType, mask_rate: float = 0.2
    ) -> np.ndarray:
        """
        Execute FRAug method on two signals.

        Args:
            signal1: The first signal.
            signal2: The second signal.
            mask_rate: The mask rate. Defaults to 0.2.

        Returns:
            np.ndarray: The augmented signal.
        """
        return frequency_mixing(signal1, signal2, mask_rate)

    @staticmethod
    def wave_aug(
        signal1: SignalType,
        signal2: SignalType,
        rates: list[float],
        wavelet: str = "db1",
        level: int = 2,
    ) -> np.ndarray:
        """
        Execute WaveAug method on two signals.

        Args:
            signal1: The first signal.
            signal2: The second signal.
            rates: The list of rates.
            wavelet: The wavelet. Defaults to "db1".
            level: The level. Defaults to 2.

        Returns:
            np.ndarray: The augmented signal.
        """
        return wave_mix(signal1, signal2, rates, wavelet, level)
