"""
Tests for Training Service.
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.services.training_service import TrainingService
from src.services.mlp_model import MLPModel

@pytest.fixture
def dummy_train_loader():
    X = torch.randn(64, 14)
    y = torch.randn(64, 1)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=16)

@pytest.fixture
def dummy_val_loader():
    X = torch.randn(32, 14)
    y = torch.randn(32, 1)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=16)

@pytest.fixture
def mock_config():
    return {
        "training": {
            "learning_rate": 0.01,
            "max_epochs": 2,
            "patience": 1,
            "batch_size": 16
        },
        "data": {"base_seed": 42}
    }

def test_tr_t1_smoke(dummy_train_loader, dummy_val_loader, mock_config):
    m = MLPModel()
    t = TrainingService(m, "mlp", mock_config, torch.device("cpu"))
    
    initial_loss = t.evaluate(dummy_val_loader)
    t.train(dummy_train_loader, dummy_val_loader, mock_config)
    final_loss = t.evaluate(dummy_val_loader)
    
    assert final_loss < initial_loss or final_loss < 2.0

def test_tr_t2_early_stopping(dummy_train_loader, dummy_val_loader, mock_config):
    # Setting max_epochs high to test early stopping
    mock_config["training"]["max_epochs"] = 50
    mock_config["training"]["patience"] = 2
    
    m = MLPModel()
    t = TrainingService(m, "mlp", mock_config, torch.device("cpu"))
    
    # Overwrite train_one_epoch to simulate a stagnating model
    t.train_one_epoch = lambda dl: 1.0
    t.evaluate = lambda dl: 1.0 # Constant val loss
    
    t.train(dummy_train_loader, dummy_val_loader, mock_config)
    
    # Should stop after patience + 1
    assert t.early_stopper.early_stop == True

def test_tr_t5_scheduler_halving(dummy_train_loader, dummy_val_loader, mock_config):
    m = MLPModel()
    t = TrainingService(m, "mlp", mock_config, torch.device("cpu"))
    initial_lr = t.optimizer.param_groups[0]['lr']
    
    # Trigger scheduler manually (patience is 10)
    for _ in range(12):
        t.scheduler.step(1.0)
        
    final_lr = t.optimizer.param_groups[0]['lr']
    assert final_lr == initial_lr * 0.5

def test_tr_t6_mse_zero(mock_config):
    X = torch.randn(16, 14)
    y = torch.randn(16, 1)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=16)
    
    m = nn.Identity() # just passes x
    m.eval = lambda: None
    m.train = lambda: None
    t = TrainingService(MLPModel(), "mlp", mock_config, torch.device("cpu"))
    
    # Evaluate with y vs y
    # evaluate uses pred=m(X). To make pred=y, we just mock model artificially
    class MockIdentityModel(nn.Module):
        def forward(self, x): return x[:, :1]
    
    # Inject y into X
    X[:, 0] = y[:, 0]
    t.model = MockIdentityModel()
    val = t.evaluate(dl)
    assert val == 0.0

def test_tr_empty_loader_raises(mock_config):
    m = MLPModel()
    t = TrainingService(m, "mlp", mock_config, torch.device("cpu"))
    
    ds = TensorDataset(torch.empty(0, 14), torch.empty(0, 1))
    dl = DataLoader(ds, batch_size=16)
    
    with pytest.raises(ZeroDivisionError):
        t.train_one_epoch(dl)
