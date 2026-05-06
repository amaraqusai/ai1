"""
Model factory to instantiate the desired network architecture.
"""
from typing import Dict, Any
import torch.nn as nn

from .mlp_model import MLPModel
from .rnn_model import RNNModel
from .lstm_model import LSTMModel

class ModelFactory:
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> nn.Module:
        model_type = model_type.lower()
        num_freqs = len(config.get("data", {}).get("frequencies", [5, 15, 30, 50]))
        window_size = config.get("data", {}).get("window_size", 10)
        
        if model_type == "mlp":
            hidden_sizes = config.get("mlp", {}).get("hidden_sizes", [64, 128, 64])
            return MLPModel(input_dim=window_size + num_freqs, hidden_sizes=hidden_sizes, output_dim=1)
            
        elif model_type == "rnn":
            rnn_cfg = config.get("rnn", {})
            return RNNModel(
                input_dim=1 + num_freqs, 
                hidden_size=rnn_cfg.get("hidden_size", 64),
                num_layers=rnn_cfg.get("num_layers", 2),
                output_dim=1
            )
            
        elif model_type == "lstm":
            lstm_cfg = config.get("lstm", {})
            return LSTMModel(
                input_dim=1 + num_freqs, 
                hidden_size=lstm_cfg.get("hidden_size", 64),
                num_layers=lstm_cfg.get("num_layers", 2),
                output_dim=1
            )
            
        else:
            raise ValueError(f"Unknown model architecture: {model_type}")
