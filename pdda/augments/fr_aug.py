import numpy as np

from pdda.core import AugmentationTechnique, SignalType


class FRAug(AugmentationTechnique):
    def augment(
        self,
        signal: SignalType,
        mask_rate: float = 0.2,
    ) -> np.ndarray:
        """Apply frequency masking to the input signal.

        This function converts the input signal to the frequency domain,
        randomly masks some frequency components, and then converts
        the signal back to the time domain. It treats the last 'forecast_horizon'
        points as the forecasting horizon.

        Args:
            signal: A 1D numpy array representing the time series signal.
            mask_rate: The rate at which frequency components are masked. Should be
             between 0 and 1. Default is 0.5.

        Returns:
            A 1D numpy array of the augmented signal, same length as the input.

        Raises:
            ValueError: If the input signal is not a 1D numpy array, if
                        forecast_horizon is invalid, or if mask_rate is not
                        between 0 and 1.
        """
        # Input validation
        signal = self._validate_input(signal)

        # Convert to frequency domain
        signal_f = np.fft.rfft(signal)

        # Create random mask
        mask = np.random.rand(len(signal_f)) < mask_rate

        # Apply mask
        masked_signal_f = signal_f * mask

        # Convert back to time domain
        augmented_signal = np.fft.irfft(masked_signal_f, n=len(signal))

        return augmented_signal

    def augment_multi(
        self,
        signal1: SignalType,
        signal2: SignalType,
        mix_rate: float = 0.5,
    ) -> np.ndarray:
        """Apply frequency mixing to the input signals.

        This function converts both input signals to the frequency domain,
        mixes their frequency components based on the mix_rate, and then converts
        the result back to the time domain. It treats the last 'forecast_horizon'
        points as the forecasting horizon.

        Args:
            signal1: A 1D numpy array representing the first time series signal.
            signal2: A 1D numpy array representing the second time series signal.
            mix_rate: The rate at which frequency components are mixed from signal2. Should be between 0 and 1. Default is 0.5.

        Returns:
            A 1D numpy array of the augmented signal, same length as the input.
        """
        # Input validation
        signal1 = self._validate_input(signal1)
        signal2 = self._validate_input(signal2)

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

        return augmented_signal
