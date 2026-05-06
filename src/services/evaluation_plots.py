"""
Plotting functions for the evaluation service.
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def save_plot(filename: str):
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{filename}", dpi=150, bbox_inches="tight")
    plt.close()

def plot_training_curves(train_losses: list, val_losses: list, model_type: str):
    plt.figure()
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses, label="Val MSE")
    plt.title(f"{model_type.upper()} Training Curves")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    save_plot(f"training_curves_{model_type}.png")

def plot_predictions(clean: np.ndarray, noisy: np.ndarray, pred: np.ndarray, model_type: str):
    plt.figure(figsize=(10, 4))
    # reshape if necessary
    clean = clean.flatten()
    noisy = noisy.flatten()
    pred = pred.flatten()
    plt.plot(noisy[-100:], label="Noisy Input", alpha=0.5)
    plt.plot(clean[-100:], label="Clean Target", linewidth=2)
    plt.plot(pred[-100:], label="Prediction", linestyle="--")
    plt.title(f"{model_type.upper()} Predictions vs Target (last 100 samples)")
    plt.legend()
    save_plot(f"predictions_{model_type}.png")

def plot_noise_robustness(sigmas: list, mses: dict):
    plt.figure()
    for mtype, vals in mses.items():
        plt.plot(sigmas, vals, marker='o', label=mtype.upper())
    plt.title("Noise Robustness")
    plt.xlabel("Noise Sigma")
    plt.ylabel("Test MSE")
    plt.legend()
    save_plot("noise_robustness.png")

def plot_per_frequency_mse(freqs: list, mses: dict):
    x = np.arange(len(freqs))
    width = 0.25
    plt.figure()
    for idx, (mtype, vals) in enumerate(mses.items()):
        offset = (idx - 1) * width
        plt.bar(x + offset, vals, width, label=mtype.upper())
    plt.xticks(x, [f"{f} Hz" for f in freqs])
    plt.title("MSE by Frequency")
    plt.ylabel("Test MSE")
    plt.legend()
    save_plot("per_frequency_mse.png")

def plot_signal_examples(clean: np.ndarray, noisy: np.ndarray, target: np.ndarray, freq: float):
    plt.figure()
    plt.plot(noisy[:100], label="Noisy", alpha=0.5)
    plt.plot(clean[:100], label="Clean")
    plt.plot(target[:100], label="Target", linestyle="--")
    plt.title(f"Signal Example: {freq} Hz")
    plt.legend()
    save_plot(f"signal_example_{freq}Hz.png")
