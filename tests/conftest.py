"""
Shared pytest fixtures.
"""
import pytest
import os
import torch
import numpy as np
import random
from unittest.mock import patch

@pytest.fixture(autouse=True)
def set_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@pytest.fixture
def tmp_config_dir(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir

@pytest.fixture(autouse=True)
def isolated_gatekeeper(tmp_path):
    # This automatically resets gatekeepers for tests, if necessary
    from src.shared import gatekeeper
    original_gatekeepers = gatekeeper._gatekeepers.copy()
    gatekeeper._gatekeepers.clear()
    
    # Mock where file_io saves things
    yield
    
    gatekeeper._gatekeepers.clear()
    gatekeeper._gatekeepers.update(original_gatekeepers)

@pytest.fixture
def sample_signal():
    fs = 100
    t = np.linspace(0, 1, fs, endpoint=False)
    y = np.sin(2 * np.pi * 10 * t)
    return y

@pytest.fixture
def small_dataset_factory():
    def _make_factory():
        return {
            "data": np.random.randn(80, 10),
            "frequency_label": np.zeros((80, 4))
        }
    return _make_factory

@pytest.fixture
def sample_setup_config():
    return {
        "project_name": "freq_extractor",
        "version": "1.00",
        "data": {
            "frequencies": [5, 15, 30, 50],
            "sampling_rate": 200,
            "duration_seconds": 10,
            "noise_sigma": 0.50,
            "eval_noise_sigmas": [0.05, 0.10, 0.20, 0.30, 0.50],
            "window_size": 10,
            "test_split": 0.15,
            "val_split": 0.15,
            "base_seed": 42
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 0.001,
            "max_epochs": 2,
            "patience": 2
        },
        "mlp": { "hidden_sizes": [16, 16] },
        "rnn": { "hidden_size": 16, "num_layers": 1 },
        "lstm": { "hidden_size": 16, "num_layers": 1 }
    }
