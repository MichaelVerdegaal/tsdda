# Time Series Domain Data Augmentation (TSDDA)

This package implements Domain specific Data Augmentation methods for Time Series in Python (NumPy).

Dataset augmentation is a technique used to create and diversify your dataset,
primarily to improve machine learning models. When we're speaking of data
augmentation techniques for time series, we can broadly categorize them into specific categories (domains):

- The time domain, which represents how a signal changes over time
- The frequency domain, which represents the same signal, but in its underlying frequencies
- The time-frequency domain, where we use both time and frequency domain representation of the signal

______________________________________________________________________

Pre-existing time series augmentation packages do exist, but (to my awareness) none of them contain the techniques i was most interested in, which are the cutting-edge augmentation techniques in the frequency domain. This was the main driver behind the creation of this package.

This package will be designed with these goals in mind:

- Easy to expand | It should be easy to implement new techniques into this package,
  so you can rapidly integrate cutting-edge solutions into your own project.
- Freedom | The user should have the freedom to make use of these techniques however they want. This is primarily referring to not including arbitrary dependencies.
- Minimalism | The user should be able to rapidly augment their data with minimal
  code. This also means no ML-based augmentations for now. If you're
  interested in that, see [TSGM](https://github.com/AlexanderVNikitin/tsgm) perhaps.

This does come with some caveats. For example, the original STAug
implementation augments series as a part of a neural network training loop,
splitting the data into a 'train' and 'forecast' set. As we want our implementations
to be more broadly applicable, we work on the basis here that we input a single
signal as a whole, and output a single signal as a whole. This slight modification
stops the techniques from being limited to just neural net forecasting models.

## Techniques

Although a generalization, the techniques can be categorized based
on in which domains they have an implementation available right now. The time domain, the frequency domain, or a combination of both.

| Technique                               | Time domain | Freq domain | Time-freq domain |
|-----------------------------------------|-------------|-------------|------------------|
| [STAug](tsdda/augments/st_aug.py)       | Yes         | Yes         | Yes              |
| [FRAug](tsdda/augments/fr_aug.py)       | Yes         | Yes         | No               |
| [DShuffle](tsdda/augments/d_shuffle.py) | No          | Yes         | No               |
| [WaveAug](tsdda/augments/wave_aug.py)   | Yes         | Yes         | No               |

## Usage

```python
from tsdda import STAug

# Create augmenter class
augmenter = STAug()

# Apply augmentation
my_signal = ...  # your list or numpy array
augmented_signal = augmenter.augment(my_signal)

# (Optional) Post-process augmented signal
augmented_signal = augmenter.transform(augmented_signal,
                                       clip_min=0,
                                       clip_max=1000,
                                       offset=100)
```
