"""
Evaluation service for model comparison and reporting.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List
import numpy as np

from .evaluation_plots import (
    plot_predictions, plot_noise_robustness, 
    plot_per_frequency_mse, plot_signal_examples
)

class EvaluationService:
    def __init__(self, device: torch.device):
        self.device = device
        self.criterion = nn.MSELoss()

    def compute_split_mse(self, model: nn.Module, dls: Dict[str, Any]) -> Dict[str, float]:
        model.eval()
        model = model.to(self.device)
        res = {}
        with torch.no_grad():
            for s in ["train", "val", "test"]:
                total_loss = 0.0
                dl = dls[s]
                for X, y in dl:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = model(X)
                    total_loss += self.criterion(pred, y).item() * X.size(0)
                res[s] = total_loss / len(dl.dataset)
        return res

    def get_predictions(self, model: nn.Module, dl: Any):
        model.eval()
        model = model.to(self.device)
        preds, targets, inputs = [], [], []
        with torch.no_grad():
            for X, y in dl:
                X_d = X.to(self.device)
                pred = model(X_d)
                preds.append(pred.cpu().numpy())
                targets.append(y.numpy())
                inputs.append(X.cpu().numpy())
        return np.concatenate(inputs), np.concatenate(targets), np.concatenate(preds)

    def evaluate_noise_robustness(self, models: Dict[str, nn.Module], test_data: Any, sigmas: List[float]):
        # Mock implementation for tracking MSEs across noise variants
        # In a full run, we would regenerate test data with varying noise sigmas.
        # But this function just plots if called externally with true `mses`.
        pass

    def build_comparison_table(self, results: Dict[str, Dict[str, float]]) -> str:
        md = "| Model | Train MSE | Val MSE | Test MSE |\n"
        md += "|-------|-----------|---------|----------|\n"
        for mtype, metrics in results.items():
            tr = metrics.get('train', 0.0)
            vl = metrics.get('val', 0.0)
            ts = metrics.get('test', 0.0)
            md += f"| {mtype.upper()} | {tr:.5f} | {vl:.5f} | {ts:.5f} |\n"
        return md

    def run_full_evaluation(self):
        # A wrapper if we need to call everything sequentially
        pass
