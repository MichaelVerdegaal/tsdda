from fdda.augment.augmenter import SingleSigAugmenter, DualSignalAugmenter
from fdda.augment.st_aug import emd_augmentation, mix_augmentation
from fdda.augment.fr_aug import frequency_mixing, frequency_masking
from fdda.augment.wave_aug import wave_mix, wave_mask
from fdda.augment.d_shuffle import dominant_shuffle

__all__ = [
    "SingleSigAugmenter",
    "DualSignalAugmenter",
    "emd_augmentation",
    "mix_augmentation",
    "frequency_mixing",
    "frequency_masking",
    "wave_mix",
    "wave_mask",
    "dominant_shuffle",
]
