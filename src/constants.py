"""
Shared project constants.
"""

import torch

MODEL_TYPES = ["mlp", "rnn", "lstm"]
FREQUENCY_LABELS = [5, 15, 30, 50]
SPLIT_NAMES = ["train", "val", "test"]

TENSOR_DTYPES = {
    "float": torch.float32,
    "int": torch.long
}

DISPLAY_MODES = ["LINE", "DOTS"]
NOISE_MODELS = ["None", "Gaussian", "Uniform"]
FILTER_MODELS = ["None", "Lowpass", "Highpass", "Bandpass"]
