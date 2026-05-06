"""
Tests for Neural Network Models.
"""
import pytest
import torch
import torch.nn as nn
from src.services.mlp_model import MLPModel
from src.services.rnn_model import RNNModel
from src.services.lstm_model import LSTMModel
from src.services.model_factory import ModelFactory
import os

class TestMLP:
    def test_mlp_t1_shape(self):
        m = MLPModel()
        x = torch.randn(32, 14)
        out = m(x)
        assert out.shape == (32, 1)
        assert not torch.isnan(out).any()

    def test_mlp_t2_batch1(self):
        m = MLPModel()
        x = torch.randn(1, 14)
        out = m(x)
        assert out.shape == (1, 1)

    def test_mlp_t3_params(self):
        m = MLPModel()
        p = m.count_parameters()
        # 14*64+64 + 64*128+128 + 128*64+64 + 64*1+1 = 960 + 8320 + 8256 + 65 = 17601
        assert 16000 <= p <= 19000

    def test_mlp_t4_gradients(self):
        m = MLPModel()
        x = torch.randn(8, 14)
        out = m(x)
        loss = out.sum()
        loss.backward()
        for param in m.parameters():
            assert param.grad is not None
            assert torch.sum(torch.abs(param.grad)) > 0

    def test_mlp_t5_tanh(self):
        m = MLPModel()
        has_tanh = any(isinstance(module, nn.Tanh) for module in m.network)
        assert has_tanh

    def test_mlp_t6_wrong_dim(self):
        m = MLPModel()
        x = torch.randn(8, 13)
        with pytest.raises(RuntimeError):
            m(x)

    def test_mlp_t7_serialization(self, tmp_path):
        m = MLPModel()
        x = torch.randn(5, 14)
        out1 = m(x)
        
        path = tmp_path / "mod.pt"
        torch.save(m.state_dict(), path)
        
        m2 = MLPModel()
        m2.load_state_dict(torch.load(path))
        out2 = m2(x)
        
        assert torch.allclose(out1, out2)

class TestRNN:
    def test_rnn_t1_shape(self):
        m = RNNModel()
        # (B, T, features)
        x = torch.randn(32, 10, 5)
        out = m(x)
        assert out.shape == (32, 1)

    def test_rnn_t2_hidden_size(self):
        m = RNNModel(hidden_size=64)
        assert m.rnn.hidden_size == 64

    def test_rnn_t3_num_layers(self):
        m = RNNModel(num_layers=2)
        assert m.rnn.num_layers == 2

    def test_rnn_t4_variable_seq(self):
        m = RNNModel()
        x1 = torch.randn(4, 5, 5)
        x2 = torch.randn(4, 20, 5)
        assert m(x1).shape == (4, 1)
        assert m(x2).shape == (4, 1)

    def test_rnn_t5_orthogonal(self):
        m = RNNModel()
        w = m.rnn.weight_hh_l0.detach()
        u, s, v = torch.svd(w)
        # s should be close to 1
        assert torch.allclose(s, torch.ones_like(s), atol=1e-5)

    def test_rnn_t6_gradients(self):
        m = RNNModel()
        x = torch.randn(8, 10, 5)
        out = m(x)
        out.sum().backward()
        for name, param in m.named_parameters():
            if 'weight' in name or 'bias' in name:
                assert param.grad is not None
                assert torch.sum(torch.abs(param.grad)) > 0

    def test_rnn_t7_dropout(self):
        m = RNNModel(num_layers=2)
        m.train() # Dropout active
        x = torch.randn(8, 10, 5)
        out1 = m(x)
        out2 = m(x)
        assert not torch.allclose(out1, out2)
        
        m.eval() # Dropout disabled
        out3 = m(x)
        out4 = m(x)
        assert torch.allclose(out3, out4)

 # Just a bound to ensure it trains

class TestModelFactory:
    def test_factory_returns_correct(self):
        config = {}
        m1 = ModelFactory.create_model("mlp", config)
        assert isinstance(m1, MLPModel)
        
        m2 = ModelFactory.create_model("rnn", config)
        assert isinstance(m2, RNNModel)
        
        m3 = ModelFactory.create_model("lstm", config)
        assert isinstance(m3, LSTMModel)

    def test_factory_raises(self):
        with pytest.raises(ValueError):
            ModelFactory.create_model("transformer", {})
