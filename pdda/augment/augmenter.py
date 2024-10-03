from abc import ABC
import numpy as np
from .wave_aug import wave_mask, wave_mix
from .d_shuffle import dominant_shuffle
from .fr_aug import frequency_masking, frequency_mixing
from .st_aug import emd_augmentation, mix_augmentation


class Augmenter(ABC):
    """Base class for all augmentation methods."""

    def __init__(self, signal: np.ndarray):
        """Augmenter abstract class."""
        # Won't be modified
        self.signal = signal
        # Will be modified after augmentation call
        self.output = signal

    def transform(
        self,
        clip_min: float | None = None,
        clip_max: float | None = None,
        offset: float = 0.0,
    ) -> "Augmenter":
        """Apply post-processing to the signal.

        Args:
            clip_min: Minimum value for clipping the output signal.
            clip_max: Maximum value for clipping the output signal.
            offset: A float value to offset the signal after augmentation.
        """
        self.output += offset
        if clip_min is not None or clip_max is not None:
            self.output = np.clip(self.output, clip_min, clip_max)
        return self


class SingleSigAugmenter(Augmenter):
    """Base class for augmentations that operate on a single signal."""

    def __init__(self, signal: np.ndarray):
        """SingleSignalAugmenter abstract class."""
        super().__init__(signal)

    def st_aug(
        self, n_imf: int = 10, random_weight_prob: float = 0.5, imf_rate: float = 1.0
    ) -> "SingleSigAugmenter":
        self.output = emd_augmentation(self.output, n_imf, random_weight_prob, imf_rate)
        return self

    def fr_aug(self, mask_rate: float = 0.2) -> "SingleSigAugmenter":
        self.output = frequency_masking(self.output, mask_rate)
        return self

    def d_shuffle(self, rates: list[float]) -> "SingleSigAugmenter":
        self.output = dominant_shuffle(self.output, rates)
        return self

    def wave_aug(
        self, rates: list[float], wavelet: str = "db1", level: int = 2
    ) -> "SingleSigAugmenter":
        self.output = wave_mask(self.output, rates, wavelet, level)
        return self


class DualSignalAugmenter(Augmenter):
    """Base class for augmentations that operate on two signals.

    In case of method chaining, output signal is always combined with
    original second signal.
    """

    def __init__(self, signal_1: np.ndarray, signal_2: np.ndarray):
        """DualSignalAugmenter abstract class."""
        super().__init__(signal_1)
        self.signal2 = signal_2

    def st_aug(
        self, alpha: float = 0.5, mix_rate: float = 1.0
    ) -> "DualSignalAugmenter":
        self.output = mix_augmentation(self.output, self.signal2, alpha, mix_rate)
        return self

    def fr_aug(self, mask_rate: float = 0.2) -> "DualSignalAugmenter":
        self.output = frequency_mixing(self.output, self.signal2, mask_rate)
        return self

    def wave_aug(
        self, rates: list[float], wavelet: str = "db1", level: int = 2
    ) -> "DualSignalAugmenter":
        self.output = wave_mix(self.output, self.signal2, rates, wavelet, level)
        return self
