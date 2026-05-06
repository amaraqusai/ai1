"""
Plotly generation for UI.
"""
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def create_signals_plot(signals, combined, mode, colors):
    fig = go.Figure()
    m = "lines" if mode == "LINE" else "markers"
    t = signals[0]["t"] if signals else np.array([])
    for i, s in enumerate(signals):
        fig.add_trace(go.Scatter(x=s["t"], y=s["y"], mode=m, name=f"Sin {i+1}", line=dict(color=colors[i])))
    
    fig_comb = go.Figure()
    if combined is not None and len(t) > 0:
        fig_comb.add_trace(go.Scatter(x=t, y=combined, mode=m, name="Mixed Clean", line=dict(color="white")))

    fig.update_layout(template="plotly_dark", margin=dict(t=30, b=30, l=30, r=30), height=300)
    fig_comb.update_layout(template="plotly_dark", margin=dict(t=30, b=30, l=30, r=30), height=300)
    return fig, fig_comb

def create_pca_plot(features, labels, colors_map):
    if len(features) < 3:
        return go.Figure().update_layout(template="plotly_dark")
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(features)
    var = pca.explained_variance_ratio_ * 100
    
    fig = go.Figure(data=[go.Scatter3d(
        x=pcs[:, 0], y=pcs[:, 1], z=pcs[:, 2],
        mode='markers',
        marker=dict(size=4, color=[colors_map[l] for l in labels])
    )])
    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis_title=f"PC1 ({var[0]:.1f}%)",
            yaxis_title=f"PC2 ({var[1]:.1f}%)",
            zaxis_title=f"PC3 ({var[2]:.1f}%)"
        )
    )
    return fig

def create_tsne_plot(features, labels, colors_map):
    if len(features) < 3:
        return go.Figure().update_layout(template="plotly_dark")
    # Quick TSNE for visual
    tsne = TSNE(n_components=3, perplexity=min(30, len(features)-1), n_iter=250, init="random")
    emb = tsne.fit_transform(features)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
        mode='markers',
        marker=dict(size=4, color=[colors_map[l] for l in labels])
    )])
    fig.update_layout(template="plotly_dark")
    return fig

def create_fft_plot(combined_signal, fs, log_scale=False):
    if combined_signal is None or len(combined_signal) == 0:
        return go.Figure().update_layout(template="plotly_dark")
    
    spectrum = np.abs(np.fft.rfft(combined_signal))
    freqs = np.fft.rfftfreq(len(combined_signal), 1/fs)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=spectrum, mode="lines", line=dict(color="#58a6ff")))
    if log_scale:
        fig.update_xaxes(type="log")
    fig.update_layout(template="plotly_dark", title="Magnitude Spectrum")
    return fig
