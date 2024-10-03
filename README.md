# Python-DDA

This package implements Domain Data Augmentation (DDA) methods in Python, for 
time-series data. 

Dataset augmentation is a technique used to create and diversify your dataset, 
primarily to improve machine learning models. When we're speaking of data 
augmentation techniques for time-series, we can categorize them into specific categories:
- The time domain, which represents how a signal changes over time
- The frequency domain, which represents the same singal, but in its underlying frequencies

This package implements augmentation methods for domains, all in NumPy.
___
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


## Techniques

Techniques will be categorized in two ways, single-augment and dual-augment. Single-augment 
augmentations input 1 time-series, and will output 1 time-series. Dual augment 
augmentations input 2 time-series, and output 1 time-series. 

Both time-domain and frequency-domain augmentations can be found in one of these 
categories.

### Single augmentations

- [STAug](https://doi.org/10.48550/arXiv.2303.14254) - Spectrum augmentation
- [FrAug](https://doi.org/10.48550/arXiv.2302.09292) - Frequency masking
- [Dominant Shuffle](https://doi.org/10.48550/arXiv.2405.16456)
- [Wave-aug](https://arxiv.org/abs/2408.10951) - Wavemask

### Dual augmentations

- [STAug](https://doi.org/10.48550/arXiv.2303.14254) - Time augmentation
- [FrAug](https://doi.org/10.48550/arXiv.2302.09292) - Frequency mixing
- [Wave-aug](https://arxiv.org/abs/2408.10951) - Wavemix

## Usage

```python
# todo
```

## TODO

- Frequency Mixing fails if 1 of the signals contains NaN values
- Make into a proper package (code-structure, PyPI)
- Standardized input types and dimensions (+validation)
- Unit-tests
- Add more post-processing features, to improve augmentations
  - Modify amplitude
  - Offset presets (i.e. calculate median/mean offset for the user)
- Update docs