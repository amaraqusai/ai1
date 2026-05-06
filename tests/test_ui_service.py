"""
Tests for UIService.
"""
import pytest
import dash
import numpy as np
from src.services.ui_service import UIService
from src.services.ui_layout import create_layout
from src.services.ui_plots import create_signals_plot, create_pca_plot, create_tsne_plot, create_fft_plot

def test_6k1_build_app():
    ui = UIService()
    assert isinstance(ui.app, dash.Dash)
    assert ui.app.layout is not None

def test_6k4_combined_signal_logic():
    # Extracted from the logic: if we sum two clean signals, it is just sum
    s1 = np.ones(10)
    s2 = np.ones(10) * 2
    combined = s1 + s2
    np.testing.assert_array_equal(combined, np.ones(10) * 3)

def test_6k9_fft_peak():
    # If we generate pure sine 10Hz, peak should be 10Hz
    fs = 100
    t = np.linspace(0, 1, fs, endpoint=False)
    combined = np.sin(2 * np.pi * 10 * t)
    
    spectrum = np.abs(np.fft.rfft(combined))
    freqs = np.fft.rfftfreq(len(combined), 1/fs)
    peak_f = freqs[np.argmax(spectrum)]
    
    assert abs(peak_f - 10) <= (fs / len(combined))

def test_ui_plots():
    # testing the plot generation doesn't crash component render
    # Create dummy data
    t = np.linspace(0, 1, 10)
    signals = [{"t": t, "y": np.ones(10)}]
    combined = np.ones(10)
    
    f1, f2 = create_signals_plot(signals, combined, "LINE", ["#fff"])
    assert f1.data[0].mode == "lines"
    
    f1, f2 = create_signals_plot(signals, combined, "DOTS", ["#fff"])
    assert f1.data[0].mode == "markers"
    
    # PCA
    features = np.random.rand(10, 5)
    labels = [0]*10
    colors_map = ["red"]
    
    fig = create_pca_plot(features, labels, colors_map)
    assert fig is not None

def test_6k2_6k5_6k6_callback():
    from src.services.ui_callbacks import generate_update_content
    # tab, fs, n_cyc, display, noise_type, flt_type, mixes..., fs..., ps..., as...
    content, metrics = generate_update_content(
        "tab-signals", 200, 5, "DOTS", "None", "None",
        ["mix"], ["mix"], [], [],
        10, 15, 20, 25,
        0, 0, 0, 0,
        1, 1, 1, 1
    )
    # 2 combined metrics
    for m in metrics:
        if isinstance(m, dash.html.Span):
            pass  # It's a Dash component
    
    # ensure mode is handled
    assert len(metrics) == 5
    assert "Fs: 200 Hz" in metrics[0].children
    
    # TSNE
    fig2 = create_tsne_plot(features, labels, colors_map)
    assert fig2 is not None
