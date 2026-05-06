"""
UI Layout for Sinusoid Explorer.
"""
from dash import html, dcc

def create_layout() -> html.Div:
    return html.Div(
        style={"backgroundColor": "#0d1117", "color": "#e6edf3", "fontFamily": "sans-serif", "padding": "20px"},
        children=[
            html.H1("SINUSOID EXPLORER", style={"textAlign": "center"}),
            html.Div(id="metrics-bar", style={"display": "flex", "justifyContent": "space-around", "borderBottom": "1px solid #30363d", "paddingBottom": "10px", "marginBottom": "20px"}),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "gap": "20px"},
                children=[
                    # Sidebar
                    html.Div(
                        style={"width": "300px", "borderRight": "1px solid #30363d", "paddingRight": "20px"},
                        children=[
                            html.H3("Global Controls"),
                            html.Label("Fs (Hz)"),
                            dcc.Slider(10, 2000, 10, value=200, id="fs-slider"),
                            html.Label("N-Cycles"),
                            dcc.Input(type="number", min=1, max=20, value=5, id="n-cycles-input"),
                            html.Label("BW (Hz)"),
                            dcc.Slider(0.1, 100, 0.1, value=10.0, id="bw-slider"),
                            html.Label("Display Mode"),
                            dcc.RadioItems(["LINE", "DOTS"], "LINE", id="display-mode"),
                            html.Label("Noise Model"),
                            dcc.Dropdown(["None", "Gaussian", "Uniform"], "None", id="noise-dropdown"),
                            html.Label("Global Filter"),
                            dcc.Dropdown(["None", "Lowpass", "Highpass", "Bandpass"], "None", id="filter-dropdown"),
                            html.Button("SWEEP NOISE", id="sweep-noise-btn", style={"marginTop": "20px"}),
                            dcc.Interval(id="sweep-interval", interval=100, disabled=True)
                        ]
                    ),
                    # Main content
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            dcc.Tabs(
                                id="tabs",
                                value="tab-signals",
                                colors={"border": "#30363d", "primary": "#58a6ff", "background": "#0d1117"},
                                children=[
                                    dcc.Tab(label="Signals", value="tab-signals"),
                                    dcc.Tab(label="T-SNE 3D", value="tab-tsne"),
                                    dcc.Tab(label="PCA 3D", value="tab-pca"),
                                    dcc.Tab(label="FFT Spectrum", value="tab-fft")
                                ]
                            ),
                            html.Div(id="tab-content", style={"marginTop": "20px"})
                        ]
                    )
                ]
            ),
            # Per-sinusoid controls appended dynamically or statically
            html.Div(id="per-sinusoid-controls", style={"display": "flex", "marginTop": "20px", "gap": "10px"})
        ]
    )

def generate_sinusoid_controls() -> list:
    panels = []
    colors = ["#ff7b72", "#79c0ff", "#d2a8ff", "#a5d6ff"]
    for i in range(4):
        panels.append(
            html.Div(
                style={"border": f"2px solid {colors[i]}", "padding": "10px", "borderRadius": "5px", "flex": "1"},
                children=[
                    html.H4(f"Sinusoid {i+1}", style={"color": colors[i]}),
                    dcc.Checklist([{"label": "MIX", "value": "mix"}], ["mix"] if i < 2 else [], id=f"mix-check-{i}"),
                    dcc.Checklist([{"label": "BPF", "value": "bpf"}], [], id=f"bpf-check-{i}"),
                    html.Label("Freq (Hz)"),
                    dcc.Slider(0.1, 100, 1.0, value=(i+1)*5, id=f"f-slider-{i}"),
                    html.Label("Phase (rad)"),
                    dcc.Slider(0, 6.28, 0.1, value=0.0, id=f"phi-slider-{i}"),
                    html.Label("Amp"),
                    dcc.Slider(0, 2.0, 0.1, value=1.0, id=f"a-slider-{i}")
                ]
            )
        )
    return panels
