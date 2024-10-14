# Python-DDA

This package implements Domain Data Augmentation (DDA) methods in Python, for
time-series data.

Dataset augmentation is a technique used to create and diversify your dataset,
primarily to improve machine learning models. When we're speaking of data
augmentation techniques for time-series, we can categorize them into specific categories (domains):

- The time domain, which represents how a signal changes over time
- The frequency domain, which represents the same singal, but in its underlying frequencies

This package implements augmentation methods for domains, all in NumPy.

______________________________________________________________________

Pre-existing time-series augmentation packages do exist, but (to my awareness) none
of them contain the most cutting-edge techniques for augmentation, which is the motivation
of this library.

This package will be designed with these goals:

- Easy to expand | It should be easy to implement new techniques into this package,
  so you can rapidly integrate cutting-edge solutions into your own project.
- Freedom | The user should have the freedom to make use of these techniques however
  they want, with the only restraints being the ones that are strictly necessary
  for to make each technique work. This also means a minimal amount of
  dependencies (i.e. no PyTorch requirements for your technique if it's not really
  needed, looking at you here researchers!).
- Minimalism | The user should be able to rapidly augment their data with a minimal
  amount of code. This probably means no ML-based augmentations for now, if you're
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

| Technique                              | Time domain | Freq domain | Time-freq domain |
|----------------------------------------|-------------|-------------|------------------|
| [STAug](pdda/augments/st_aug.py)       | Yes         | Yes         | Yes              |
| [FRAug](pdda/augments/fr_aug.py)       | Yes         | Yes         | No               |
| [DShuffle](pdda/augments/d_shuffle.py) | No          | Yes         | No               |
| [WaveAug](pdda/augments/wave_aug.py)   | Yes         | Yes         | No               |

## Usage

```python
from pdda import STAug

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
