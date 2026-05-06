"""
Callbacks for the Sinusoid Explorer Dash app.
"""
from dash import Input, Output, State, html, dcc
import numpy as np
from .ui_plots import create_signals_plot, create_pca_plot, create_tsne_plot, create_fft_plot
from .data_generator import SignalGenerator

COLORS_HEX = ["#ff7b72", "#79c0ff", "#d2a8ff", "#a5d6ff"]

def generate_update_content(tab, fs, n_cyc, display, noise_type, flt_type, 
                       mix0, mix1, mix2, mix3, 
                       f0, f1, f2, f3, 
                       p0, p1, p2, p3, 
                       a0, a1, a2, a3):
    # Generate signals
    duration = n_cyc / max(0.1, min(f0, f1, f2, f3)) # approximation
    freqs = [f0, f1, f2, f3]
    phis = [p0, p1, p2, p3]
    amps = [a0, a1, a2, a3]
    mixes = [mix0, mix1, mix2, mix3]
    
    try:
        gen = SignalGenerator(frequencies=[f for i, f in enumerate(freqs) if "mix" in (mixes[i] or [])],
                              sampling_rate=fs, duration=duration, noise_sigma=0.5 if noise_type!="None" else 0)
    except ValueError:
        return html.Div("Invalid Parameters (e.g. Nyquist violation)"), ""

    signals = []
    combined = None
    features = []
    labels = []
    t = np.linspace(0, duration, int(fs*duration))
    
    for i in range(4):
        if mixes[i] and "mix" in mixes[i]:
            # clean
            _, clean, _ = gen.generate_clean(freqs[i], amps[i], phis[i])
            signals.append({"t": t, "y": clean})
            if combined is None:
                combined = np.zeros_like(clean)
            combined += clean
            
            # windowing for features
            for w in range(0, len(clean)-10, max(1, len(clean)//50)):
                features.append(clean[w:w+10])
                labels.append(i)

    if noise_type == "Gaussian" and combined is not None:
        combined += np.random.normal(0, 0.5, size=combined.shape)
    elif noise_type == "Uniform" and combined is not None:
        combined += np.random.uniform(-0.5, 0.5, size=combined.shape)

    metrics = [
        html.Span(f"Fs: {fs} Hz"),
        html.Span(f"N-CYC: {n_cyc}"),
        html.Span(f"T: {duration:.2f}s"),
        html.Span(f"Samples: {len(t)}"),
        html.Span(f"F_MIN: {fs/len(t) if len(t)>0 else 0:.2f} Hz")
    ]

    if tab == "tab-signals":
        fig1, fig2 = create_signals_plot(signals, combined, display, COLORS_HEX)
        return html.Div([dcc.Graph(figure=fig1), dcc.Graph(figure=fig2)]), metrics
    elif tab == "tab-tsne":
        fig = create_tsne_plot(np.array(features), labels, COLORS_HEX)
        return dcc.Graph(figure=fig, style={"height": "600px"}), metrics
    elif tab == "tab-pca":
        fig = create_pca_plot(np.array(features), labels, COLORS_HEX)
        return dcc.Graph(figure=fig, style={"height": "600px"}), metrics
    elif tab == "tab-fft":
        fig = create_fft_plot(combined, fs, False)
        return dcc.Graph(figure=fig, style={"height": "400px"}), metrics

    return html.Div(), metrics

def register_callbacks(app):
    @app.callback(
        Output("per-sinusoid-controls", "children"),
        Input("fs-slider", "value") # Dummy trigger to initialize, actual controls generation logic can be here or static
    )
    def init_controls(_):
        from .ui_layout import generate_sinusoid_controls
        return generate_sinusoid_controls()

    @app.callback(
        Output("tab-content", "children"),
        Output("metrics-bar", "children"),
        Input("tabs", "value"),
        Input("fs-slider", "value"),
        Input("n-cycles-input", "value"),
        Input("display-mode", "value"),
        Input("noise-dropdown", "value"),
        Input("filter-dropdown", "value"),
        [Input(f"mix-check-{i}", "value") for i in range(4)],
        [Input(f"f-slider-{i}", "value") for i in range(4)],
        [Input(f"phi-slider-{i}", "value") for i in range(4)],
        [Input(f"a-slider-{i}", "value") for i in range(4)]
    )
    def update_content(tab, fs, n_cyc, display, noise_type, flt_type, 
                       mix0, mix1, mix2, mix3, 
                       f0, f1, f2, f3, 
                       p0, p1, p2, p3, 
                       a0, a1, a2, a3):
        return generate_update_content(tab, fs, n_cyc, display, noise_type, flt_type, 
                       mix0, mix1, mix2, mix3, 
                       f0, f1, f2, f3, 
                       p0, p1, p2, p3, 
                       a0, a1, a2, a3)
