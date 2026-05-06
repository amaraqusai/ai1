"""
Training helpers: EarlyStopping and CheckpointManager.
"""
import torch
import copy
from typing import Dict, Any, Optional
import os
from ..shared.gatekeeper import get_gatekeeper
from ..shared.version import CODE_VERSION

class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class CheckpointManager:
    @staticmethod
    def save(filepath: str, state_dict: dict, optimizer_state: dict, epoch: int, val_mse: float, model_type: str, history: dict = None):
        gk = get_gatekeeper("checkpoint")
        payload = {
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer_state,
            "epoch": epoch,
            "val_mse": val_mse,
            "config_version": CODE_VERSION,
            "model_type": model_type,
            "history": history or {"train_loss": [], "val_loss": []}
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        gk.execute(torch.save, payload, filepath)

    @staticmethod
    def load(filepath: str) -> Dict[str, Any]:
        gk = get_gatekeeper("checkpoint")
        payload = gk.execute(torch.load, filepath, map_location="cpu", weights_only=True)
        if "config_version" in payload and payload["config_version"] != CODE_VERSION:
            raise ValueError(f"Checkpoint config_version {payload['config_version']} does not match current code version {CODE_VERSION}.")
        return payload
