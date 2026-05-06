"""
Training Service coordinates the full training loop.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import random
import numpy as np
from typing import Dict, Any

from .training_helpers import EarlyStopping, CheckpointManager

logger = logging.getLogger("freq_extractor.training")

def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TrainingService:
    def __init__(self, model: nn.Module, model_type: str, config: Dict[str, Any], device: torch.device):
        self.model = model.to(device)
        self.model_type = model_type
        self.config = config["training"]
        self.device = device
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=10)
        self.early_stopper = EarlyStopping(patience=self.config["patience"])

    def train_one_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X, y in dataloader:
            try:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(X)
                loss = self.criterion(pred, y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item() * X.size(0)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and self.device.type != 'cpu':
                    logger.warning("CUDA OOM detected! Falling back to CPU for sequence processing.")
                    self.device = torch.device('cpu')
                    self.model = self.model.to(self.device)
                    
                    # Re-run current batch
                    X, y = X.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()
                    pred = self.model(X)
                    loss = self.criterion(pred, y)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    total_loss += loss.item() * X.size(0)
                else:
                    raise e
        return total_loss / len(dataloader.dataset)

    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                loss = self.criterion(pred, y)
                total_loss += loss.item() * X.size(0)
        return total_loss / len(dataloader.dataset)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, config_full: Dict[str, Any]) -> None:
        _set_seeds(config_full["data"]["base_seed"])
        
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(1, self.config["max_epochs"] + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            self.scheduler.step(val_loss)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            logger.info(f"Epoch {epoch:03d} | Train MSE: {train_loss:.5f} | Val MSE: {val_loss:.5f}")
            
            self.early_stopper(val_loss, self.model)
            if self.early_stopper.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break
                
        if self.early_stopper.best_state:
            self.model.load_state_dict(self.early_stopper.best_state)
            
        ckpt_path = f"results/checkpoints/best_{self.model_type}.pt"
        CheckpointManager.save(
            filepath=ckpt_path,
            state_dict=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            epoch=epoch,
            val_mse=self.early_stopper.best_loss,
            model_type=self.model_type,
            history=history
        )
