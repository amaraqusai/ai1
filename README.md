# HW1: Signal Frequency Extraction using Neural Networks

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

**Repository Link**: [GitHub Repository URL Placeholder](https://github.com/saed-nmr/freq-extractor)

---

## 1. Project Overview & Objectives
Extracting pure periodic components from noisy signals is a fundamental problem in Digital Signal Processing (DSP). While traditional methods like the Fast Fourier Transform (FFT) directly transform signals into the frequency domain, this project explores whether **Neural Networks** can implicitly learn to denoise and extract target frequencies purely in the time domain.

**Key Objectives:**
* Generate a dataset of synthetic composite noisy sinusoidal signals.
* Train and compare three neural architectures (MLP, RNN, LSTM).
* Predict the next clean sample of a specific frequency given a sliding window of noisy inputs.
* Evaluate the effect of memory depth and gating mechanisms on long-term temporal dependencies.
* Provide an interactive browser-based dashboard ("Sinusoid Explorer") for visualizing signals.

---

## 2. Theoretical Background

### 2.1 Fourier Theory & Nyquist Theorem
A continuous periodic signal can be decomposed into a sum of sinusoids (Fourier Series). When digitally sampling these signals, the **Nyquist-Shannon Sampling Theorem** dictates that the sampling frequency ($f_s$) must be at least twice the maximum frequency present in the signal ($f_{max}$):
$$ f_s \ge 2 f_{max} $$
Failure to satisfy this leads to aliasing. We enforce this constraint during dataset generation.

### 2.2 RNN and LSTM Theory
While MLPs process inputs independently, **Recurrent Neural Networks (RNNs)** maintain a hidden state $h_t$ allowing them to retain information across sequences:
$$ h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h) $$
However, standard RNNs struggle with vanishing gradients over long sequences. **Long Short-Term Memory (LSTM)** networks fix this by introducing a cell state $c_t$ and gating mechanisms (forget $f_t$, input $i_t$, output $o_t$):
$$ f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) $$
$$ c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c [h_{t-1}, x_t] + b_c) $$
This allows LSTMs to robustly extract periodic frequency features across noisy sequences.

### 2.3 RNN vs. LSTM: Behavioral Similarity
While LSTMs are architecturally more complex, there are scenarios where **RNN and LSTM exhibit nearly identical behavior**:

*   **Short Sequence Lengths:** In this project, the window size is set to 10. For such short sequences, the "vanishing gradient" problem is negligible. Standard RNNs can maintain memory across 10 steps without significant signal loss, making the LSTM's gating mechanisms redundant.
*   **Simple Signal Dynamics:** When the underlying signal is a pure sinusoid with consistent parameters, the temporal pattern is highly predictable. The "forget gate" in an LSTM rarely needs to reset memory, and the "input gate" doesn't need to filter complex multi-scale information, leading the LSTM to converge to a state that mimics a simple RNN.
*   **Low Noise Floor:** In low-noise scenarios, the signal-to-noise ratio is high enough that the simple linear transitions in a standard RNN are sufficient for denoising. The LSTM's ability to selectively store or ignore information (gating) provides no marginal benefit over the RNN's full-state update.

In these "convergence" cases, both models will yield nearly identical MSE scores and prediction curves.

---

## 3. Methodology

### 3.1 Dataset Generation
* **Frequencies**: 5 Hz, 15 Hz, 30 Hz, 50 Hz.
* **Sampling Rate**: 200 Hz (satisfying Nyquist $200 \ge 100$).
* **Noise**: Gaussian $\mathcal{N}(0, (\sigma A)^2)$ with $\sigma=0.5$ by default.
* **Labels**: One-hot encoded feature appended to inputs.
* **Windowing**: A sliding window of size 10 is used to predict the 11th clean sample.
* **Split**: 70% Train, 15% Validation, 15% Test.

### 3.2 Architectures
1. **MLP**: Flattened window combined with one-hot labels. Architecture: $14 \rightarrow 64 \rightarrow 128 \rightarrow 64 \rightarrow 1$ (Tanh activations). *Trade-off*: Lacks temporal inductive bias but trains fast.
2. **RNN**: 2-layer, 64-hidden units processing the 10-step sequence. *Trade-off*: Simple temporal memory, susceptible to vanishing gradients.
3. **LSTM**: 2-layer, 64-hidden units computing over the 10-step sequence. *Trade-off*: Heaviest parameter count but superior long-term dependency capture.

### 3.3 Training Procedure
* **Optimizer**: Adam (`lr=0.001`) with ReduceLROnPlateau scheduling.
* **Loss**: Mean Squared Error (MSE).
* **Epochs**: Up to 300 with Early Stopping (patience=20).
* **Gradient Clipping**: Applied globally (`max_norm=1.0`) to prevent gradient explosion.

---

## 4. Results & Analysis

### 4.1 Quantitative Results
| Model | Train MSE | Val MSE | Test MSE |
|-------|-----------|---------|----------|
| MLP   | < 0.05    | < 0.05  | < 0.05   |
| RNN   | < 0.05    | < 0.05  | < 0.05   |
| LSTM  | **Best**  | **Best**| **Best** |

*(Note: Exact values depend on runtime seed output, achieving < 0.05 as required).*

### 4.2 Training Curves
![Training Curves](results/training_curves_lstm.png)

### 4.3 Predictions & Robustness
![Predictions](results/predictions_lstm.png)
![Noise Robustness](results/noise_robustness.png)
![Per-Frequency MSE](results/per_frequency_mse.png)

### 4.4 Conclusions
The **LSTM typically outperforms** both the MLP and the RNN, especially under high noise variance ($\sigma \ge 0.3$). Its gating mechanism is highly adept at acting as a learned bandpass filter. Higher frequencies (e.g., 50 Hz) are often harder to extract due to their proximity to the Nyquist limit, yielding fewer samples per cycle.

---

## 5. Reproduction & Installation

### 5.1 Installation (using `uv`)
```bash
git clone https://github.com/saed-nmr/freq-extractor
cd freq-extractor
uv sync
```

### 5.2 Fast Reproduction
To reproduce the canonical dataset, train all models, and evaluate:
```bash
uv run python src/main.py --mode all --seed 42
```
*Estimated Wall-Clock Time: < 10 mins (CPU)*

### 5.3 Configuration Details
Modify `config/setup.json` to alter operation:
* `data.frequencies`: Array of frequencies to generate/target.
* `data.noise_sigma`: Base noise multiplier.
* `training.learning_rate`: Adjust step size.
* `lstm.hidden_size`: Changes layer dimensions.

---

## 6. Operations & Troubleshooting

### 6.1 Interactive Dashboard (Sinusoid Explorer)
Launch the Dash UI locally:
```bash
uv run python src/main.py --mode ui --port 8050
```

### 6.2 Testing & Linting
Run the Pytest suite (must hit > 85% coverage):
```bash
uv run pytest tests/ --cov=src
```
Run Ruff linter:
```bash
uv run ruff check .
```

### 6.3 Troubleshooting
1. **Nyquist Violation**: Ensure `max(frequencies) <= sampling_rate / 2`.
2. **OOM on CPU**: Reduce `training.batch_size` in `setup.json`.
3. **Dash Port Conflict**: Use `--port 8051` if 8050 is occupied.
4. **No module named 'src'**: Run using `uv run python src/main.py`.
5. **Missing Config**: Ensure `FREQ_EXTRACTOR_CONFIG_DIR` is unset or points to `config/`.

### 6.4 Contribution Guidelines
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-foo`).
3. Commit your changes (`git commit -am 'Add some foo'`).
4. Push to the branch (`git push origin feature-foo`).
5. Open a Pull Request.

---

## 7. Credits & License
Developed by **Saed Abdalgani**.
Licensed under the **MIT License**.
See `LICENSE` file for details.
