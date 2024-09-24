# python-fdda

This package implements Frequency Domain Data Augmentation (FDDA) methods in Python
using NumPy.

FDDA is an augmentation technique used to enhance and diversify your training dataset,
done by manipulating the frequency components of (time-series) signals.

The motivation for this package is that i desired a library that could be used
without much effort, and with minimal dependencies. All of our current methods have
implementations are available, but they tend to require Pytorch and/or are designed
to be injected during the training process, making them less broadly applicable. This
package solves this, at the cost of being slightly less faithful to the proposed
implementations.

## Techniques

- [FrAug (10.48550/arXiv.2302.09292)](https://arxiv.org/abs/2302.09292)
  - Frequency Masking
  - Frequency Mixing
- [Dominant Shuffle (10.48550/arXiv.2405.16456)](https://arxiv.org/abs/2405.16456v1)

## Usage

```python
from fdda import frequency_masking, frequency_mixing, dominant_shuffle

# Example: Frequency Masking
signal = ...
augmented_signal = frequency_masking(signal, forecast_horizon=12)

# Example: Frequency Mixing
signal_2 = ...
augmented_signal_2 = frequency_mixing(signal, signal_2, forecast_horizon=12)

# Example: Dominant Shuffle
augmented_signal_3 = dominant_shuffle(signal)
```

## TODO

- Frequency Mixing fails if 1 of the signals contains NaN values
- Better input validation
- Allow for more input types?
- Add post-processing features, to improve augmentations
  - Modify amplitude
  - Offset presets (i.e. calculate median/mean offset for the user)
- Consider [STAug](https://arxiv.org/abs/2303.14254)
- Consider [Wave-Mask/Mix](https://arxiv.org/abs/2408.10951)
