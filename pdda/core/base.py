from abc import ABC, abstractmethod
import numpy as np

from .types import SignalType
from .utils import validate_and_convert_input


class AugmentationTechnique(ABC):
    """Base class for all augmentation techniques."""

    supports_combination = False

    @abstractmethod
    def augment(self, signal: SignalType) -> np.ndarray:
        """Augment a single signal."""
        pass

    def augment_multi(self, signals: list[SignalType], **params) -> np.ndarray:
        """Combine multiple signals into one. Default implementation raises NotImplementedError."""
        raise NotImplementedError("This technique does not support signal combination.")

    @staticmethod
    def _validate_input(signal: SignalType) -> np.ndarray:
        """Validate and convert input signal."""
        return validate_and_convert_input(signal)

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
