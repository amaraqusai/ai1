"""
Tests for DataService, SignalGenerator, DatasetBuilder, and Normalizer.
"""
import pytest
import numpy as np
from src.services.data_generator import SignalGenerator
from src.services.data_pipeline import DatasetBuilder, DatasetSplitter, DataNormalizer, one_hot_encode

def test_sg_t12_nyquist_violation():
    with pytest.raises(ValueError):
        SignalGenerator(frequencies=[150], sampling_rate=200, duration=10)

def test_sg_t13_zero_duration():
    with pytest.raises(ValueError):
        SignalGenerator(frequencies=[5], sampling_rate=100, duration=0)

def test_sg_t11_negative_noise():
    with pytest.raises(ValueError):
        SignalGenerator(frequencies=[5], sampling_rate=100, duration=10, noise_sigma=-0.1)

def test_sg_t1_freq_match():
    gen = SignalGenerator([10], 100, 10, 0)
    t, clean, phi = gen.generate_clean(10, 1.0, 0.0)
    spectrum = np.abs(np.fft.rfft(clean))
    freqs = np.fft.rfftfreq(len(clean), 1/100)
    peak_f = freqs[np.argmax(spectrum)]
    assert abs(peak_f - 10) < 0.5

def test_sg_t2_noise_snr():
    # SNR ratio for σ=0.1
    gen = SignalGenerator([5], 100, 10, noise_sigma=0.1)
    t, clean, noisy = gen.generate_noisy(5)
    noise = noisy - clean
    signal_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    # Expected power of A*sin is A^2/2 = 0.5. Noise power is (0.1)^2 = 0.01. SNR = 50 -> 10*log10(50) ≈ 17 dB
    assert abs(snr_db - 16.98) < 2.0

def test_sg_t3_sliding_window_count():
    gen = SignalGenerator([5], 200, 10, 0.1) # 2000 samples
    t, clean, noisy = gen.generate_noisy(5)
    
    signals = {5: (clean, noisy)}
    b = DatasetBuilder(window_size=10)
    ds = b.build_from_signals(signals, [5])
    assert len(ds["noisy_samples"]) == 1990

def test_sg_t4_reproducibility():
    np.random.seed(42)
    gen = SignalGenerator([5], 200, 1, 0.5)
    _, _, noisy1 = gen.generate_noisy(5)
    
    np.random.seed(42)
    _, _, noisy2 = gen.generate_noisy(5)
    
    assert np.array_equal(noisy1, noisy2)

def test_sg_t5_zero_noise():
    gen = SignalGenerator([5], 200, 1, 0.0)
    _, clean, noisy = gen.generate_noisy(5)
    np.testing.assert_array_equal(clean, noisy)

def test_sg_t8_one_hot():
    oh = one_hot_encode(2, 4)
    assert np.sum(oh) == 1.0
    assert np.argmax(oh) == 2

def test_sg_t7_t9_splits_sum_AND_stratified():
    n_samples = 400
    dataset = {
        "frequency_label": np.concatenate([
            np.tile(one_hot_encode(i, 4), (100, 1)) for i in range(4)
        ]),
        "data": np.zeros(400)
    }
    splitter = DatasetSplitter(val_split=0.15, test_split=0.15)
    splits = splitter.split(dataset)
    
    ntr = len(splits["train"]["data"])
    nval = len(splits["val"]["data"])
    nts = len(splits["test"]["data"])
    assert ntr + nval + nts == 400
    
    for s in ["train", "val", "test"]:
        labels = splits[s]["frequency_label"]
        unique_labels = np.unique(labels, axis=0)
        assert len(unique_labels) == 4

def test_sg_t10_normalizer():
    norm = DataNormalizer()
    train_data = np.random.normal(loc=5.0, scale=2.0, size=1000)
    val_data = np.random.normal(loc=-1.0, scale=1.0, size=100)
    
    norm.fit(train_data)
    t_train = norm.transform(train_data)
    t_val = norm.transform(val_data)
    
    assert abs(np.mean(t_train)) < 0.1
    assert abs(np.std(t_train) - 1.0) < 0.1
    
    assert abs(np.mean(t_val)) > 0.5 # should have different stats

def test_sg_t14_noise_mean():
    gen = SignalGenerator([5], 200, 100, noise_sigma=0.5)
    _, clean, noisy = gen.generate_noisy(5)
    noise = noisy - clean
    # std=0.5, N=20000 -> 3*0.5/sqrt(20000) ≈ 0.0106
    assert abs(np.mean(noise)) < 3 * 0.5 / np.sqrt(20000)

def test_sg_t15_noise_std():
    gen = SignalGenerator([5], 200, 100, noise_sigma=0.5)
    _, clean, noisy = gen.generate_noisy(5)
    noise = noisy - clean
    assert abs(np.std(noise) - 0.5) / 0.5 < 0.05
