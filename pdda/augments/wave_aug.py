import numpy as np
import pywt
from pdda.core import AugmentationTechnique, SignalType


class WaveAug(AugmentationTechnique):
    def augment(
        self,
        signal: SignalType,
        rates: list[float],
        wavelet: str = "db1",
        level: int = 2,
    ) -> np.ndarray:
        """Applies wave-mask augment

        Args:
        - signal: A numpy array representing the time series signal.
                  Shape can be (time_steps,) for a single feature or
                  (time_steps, features) for multiple features.
        - rates: List of mask rates for each wavelet level.
        - wavelet: Type of wavelet to use.
        - level: Number of decomposition levels.

        Returns:
        - A numpy array of the augmented signal, same shape as the input.
        """
        # Input validation
        signal = self._validate_input(signal)

        # TODO: move to utils
        original_shape = signal.shape
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        time_steps, num_features = signal.shape
        s_mask = np.empty_like(signal, dtype=np.float32)

        for col in range(num_features):
            coeffs = pywt.wavedec(
                signal[:, col], wavelet=wavelet, mode="symmetric", level=level
            )

            # Randomly mask coefficients
            for i in range(1, len(coeffs)):
                c_i = coeffs[i]
                selected_rate = rates[min(i, len(rates) - 1)]
                m = np.random.uniform(0, 1, c_i.shape) < selected_rate
                coeffs[i] = np.where(m, 0, c_i)
            s = pywt.waverec(coeffs, wavelet=wavelet, mode="symmetric")
            s_mask[:, col] = s[:time_steps]

        # Restore original shape
        if len(original_shape) == 1:
            s_mask = s_mask.flatten()

        return s_mask

    def augment_multi(
        self,
        signal1: SignalType,
        signal2: SignalType,
        rates: list[float],
        wavelet: str = "db1",
        level: int = 2,
    ) -> np.ndarray:
        """Applies wave-mix augment

        Args:
            - signal1, signal2: Numpy arrays representing the time series signals.
                                Shape can be (time_steps,) for a single feature or
                                (time_steps, features) for multiple features.
            - rates: List of mix rates for each wavelet level.
            - wavelet: Type of wavelet to use.
            - level: Number of decomposition levels.

        Returns:
            - A numpy array of the mixed signal, same shape as the inputs.
        """
        # Input validation
        signal1 = self._validate_input(signal1)
        signal2 = self._validate_input(signal2)

        # TODO: move to utils
        original_shape = signal1.shape
        if signal1.ndim == 1:
            signal1 = signal1.reshape(-1, 1)
            signal2 = signal2.reshape(-1, 1)

        time_steps, num_features = signal1.shape
        s_mixed = np.empty_like(signal1, dtype=np.float32)

        for col in range(num_features):
            coeffs1 = pywt.wavedec(
                signal1[:, col], wavelet=wavelet, mode="symmetric", level=level
            )
            coeffs2 = pywt.wavedec(
                signal2[:, col], wavelet=wavelet, mode="symmetric", level=level
            )

            # Mix coefficients
            mixed_coeffs = []
            for i in range(len(coeffs1)):
                c1, c2 = coeffs1[i], coeffs2[i]
                selected_rate = rates[min(i, len(rates) - 1)]
                mask = np.random.uniform(0, 1, c1.shape) < selected_rate
                mixed_c = np.where(mask, c1, c2)
                mixed_coeffs.append(mixed_c)

            # Reconstruct mixed signal
            s = pywt.waverec(mixed_coeffs, wavelet=wavelet, mode="symmetric")
            s_mixed[:, col] = s[:time_steps]

        # Restore original shape
        if len(original_shape) == 1:
            s_mixed = s_mixed.flatten()

        return s_mixed
