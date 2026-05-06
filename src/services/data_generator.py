"""
Signal generation module.
"""
import numpy as np

class SignalGenerator:
    def __init__(self, frequencies, sampling_rate, duration, noise_sigma=0.5):
        self.frequencies = frequencies
        self.fs = sampling_rate
        self.duration = duration
        self.noise_sigma = noise_sigma
        self._check_nyquist()

    def _check_nyquist(self):
        max_f = max(self.frequencies)
        if max_f > self.fs / 2:
            raise ValueError(f"Nyquist violation: max freq {max_f} > fs/2 ({self.fs/2})")
        if self.duration <= 0 or self.fs <= 0:
            raise ValueError("Duration and sampling rate must be > 0.")
        if self.noise_sigma < 0:
            raise ValueError("Noise sigma must be >= 0.")

    def generate_clean(self, f, A=1.0, phi=None):
        if f <= 0:
            raise ValueError("Frequency must be > 0.")
        if phi is None:
            phi = np.random.uniform(-2 * np.pi, 2 * np.pi)
        t = np.linspace(0, self.duration, int(self.fs * self.duration), endpoint=False)
        return t, A * np.sin(2 * np.pi * f * t + phi), phi

    def generate_noisy(self, f, A=1.0, phi=None):
        t, clean, used_phi = self.generate_clean(f, A, phi)
        # Apply noise to amplitude and phase as per user request
        # (A +- noise) * sin(2*pi*f*t + phi +- noise)
        noise_A = np.random.normal(0, self.noise_sigma, size=clean.shape)
        noise_phi = np.random.normal(0, self.noise_sigma, size=clean.shape)
        
        noisy = (A + noise_A) * np.sin(2 * np.pi * f * t + used_phi + noise_phi)
        return t, clean, noisy
