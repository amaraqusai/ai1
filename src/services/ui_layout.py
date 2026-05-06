"""
UI Layout for Sinusoid Explorer.
"""
from dash import html, dcc

def create_layout() -> html.Div:
    return html.Div(
        style={"backgroundColor": "#ffffff", "color": "#1f2328", "fontFamily": "Segoe UI, Tahoma, Geneva, Verdana, sans-serif", "padding": "20px"},
        children=[
            html.H1("SINUSOID EXPLORER", style={"textAlign": "center", "color": "#0969da", "fontWeight": "bold"}),
            html.Div(id="metrics-bar", style={"display": "flex", "justifyContent": "space-around", "borderBottom": "2px solid #d0d7de", "paddingBottom": "15px", "marginBottom": "25px", "fontWeight": "bold", "fontSize": "1.1em"}),
            html.Div(
                style={"display": "flex", "flexDirection": "row", "gap": "20px"},
                children=[
                    # Sidebar
                    html.Div(
                        style={"width": "300px", "borderRight": "2px solid #d0d7de", "paddingRight": "20px", "backgroundColor": "#f6f8fa", "padding": "15px", "borderRadius": "8px"},
                        children=[
                            html.H3("Global Controls", style={"borderBottom": "1px solid #d0d7de", "paddingBottom": "5px"}),
                            html.Label("Fs (Hz)", style={"fontWeight": "bold", "marginTop": "10px", "display": "block"}),
                            dcc.Slider(10, 2000, 10, value=100, id="fs-slider"),
                            html.Label("N-Cycles", style={"fontWeight": "bold", "marginTop": "10px", "display": "block"}),
                            dcc.Input(type="number", min=1, max=20, value=5, id="n-cycles-input", style={"width": "100%", "padding": "5px"}),
                            html.Label("BW (Hz)", style={"fontWeight": "bold", "marginTop": "10px", "display": "block"}),
                            dcc.Slider(0.1, 100, 0.1, value=10.0, id="bw-slider"),
                            html.Label("Display Mode", style={"fontWeight": "bold", "marginTop": "10px", "display": "block"}),
                            dcc.RadioItems(["LINE", "DOTS"], "LINE", id="display-mode", labelStyle={"marginRight": "10px"}),
                            html.Label("Noise Model", style={"fontWeight": "bold", "marginTop": "10px", "display": "block"}),
                            dcc.Dropdown(["None", "Gaussian", "Uniform"], "None", id="noise-dropdown"),
                            html.Label("Global Filter", style={"fontWeight": "bold", "marginTop": "10px", "display": "block"}),
                            dcc.Dropdown(["None", "Lowpass", "Highpass", "Bandpass"], "None", id="filter-dropdown"),
                            html.Button("SWEEP NOISE", id="sweep-noise-btn", style={"marginTop": "20px", "width": "100%", "padding": "10px", "backgroundColor": "#0969da", "color": "white", "border": "none", "borderRadius": "6px", "cursor": "pointer", "fontWeight": "bold"}),
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
                                colors={"border": "#d0d7de", "primary": "#0969da", "background": "#f6f8fa"},
                                children=[
                                    dcc.Tab(label="Signals", value="tab-signals", style={"fontWeight": "bold"}),
                                    dcc.Tab(label="T-SNE 3D", value="tab-tsne", style={"fontWeight": "bold"}),
                                    dcc.Tab(label="PCA 3D", value="tab-pca", style={"fontWeight": "bold"}),
                                    dcc.Tab(label="FFT Spectrum", value="tab-fft", style={"fontWeight": "bold"})
                                ]
                            ),
                            html.Div(id="tab-content", style={"marginTop": "20px", "padding": "10px", "backgroundColor": "#ffffff", "border": "1px solid #d0d7de", "borderRadius": "8px"})
                        ]
                    )
                ]
            ),
            # Per-sinusoid controls
            html.Div(id="per-sinusoid-controls", style={"display": "flex", "marginTop": "20px", "gap": "15px", "flexWrap": "wrap"})
        ]
    )

def generate_sinusoid_controls() -> list:
    panels = []
    # High-contrast colors for light background: Deep Red, Deep Blue, Deep Purple, Dark Grey, Deep Green
    border_colors = ["#d73a49", "#005cc5", "#6f42c1", "#24292f", "#22863a"]
    # Even darker versions for text/headers to ensure visibility
    text_colors = ["#8b0000", "#00008b", "#4b0082", "#000000", "#006400"]
    for i in range(5):
        panels.append(
            html.Div(
                style={"border": f"3px solid {border_colors[i]}", "padding": "15px", "borderRadius": "8px", "flex": "1", "minWidth": "200px", "backgroundColor": "#ffffff", "boxShadow": "2px 2px 5px rgba(0,0,0,0.1)"},
                children=[
                    html.H4(f"Sinusoid {i+1}", style={"color": text_colors[i], "marginTop": "0", "borderBottom": f"2px solid {border_colors[i]}", "fontWeight": "bold", "textAlign": "center"}),
                    dcc.Checklist([{"label": "MIX", "value": "mix"}], ["mix"] if i < 3 else [], id=f"mix-check-{i}", labelStyle={"fontWeight": "bold", "color": "#000000"}),
                    dcc.Checklist([{"label": "BPF", "value": "bpf"}], [], id=f"bpf-check-{i}", labelStyle={"fontWeight": "bold", "color": "#000000"}),
                    html.Label("Freq (Hz)", style={"fontWeight": "bold", "display": "block", "marginTop": "5px", "color": "#000000"}),
                    dcc.Slider(0.1, 100, 1.0, value=(i+1)*2, id=f"f-slider-{i}"),
                    html.Label("Phase (rad)", style={"fontWeight": "bold", "display": "block", "marginTop": "5px", "color": "#000000"}),
                    dcc.Slider(-6.28, 6.28, 0.1, value=0.0, id=f"phi-slider-{i}"),
                    html.Label("Amp", style={"fontWeight": "bold", "display": "block", "marginTop": "5px", "color": "#000000"}),
                    dcc.Slider(0, 2.0, 0.1, value=1.0, id=f"a-slider-{i}")
                ]
            )
        )
    return panels
