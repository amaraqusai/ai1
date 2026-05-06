"""
Tests for LSTM Models.
"""
import pytest
import torch
import torch.nn as nn
from src.services.mlp_model import MLPModel
from src.services.rnn_model import RNNModel
from src.services.lstm_model import LSTMModel

class TestLSTM:
    def test_lstm_t1_shape(self):
        m = LSTMModel()
        x = torch.randn(32, 10, 5)
        out = m(x)
        assert out.shape == (32, 1)

    def test_lstm_t2_init_state(self):
        m = LSTMModel(num_layers=2, hidden_size=64)
        c0, h0 = m.init_hidden(32)
        assert c0.shape == (2, 32, 64)
        assert h0.shape == (2, 32, 64)

    def test_lstm_t3_param_count(self):
        m1 = RNNModel(hidden_size=64, num_layers=2)
        m2 = LSTMModel(hidden_size=64, num_layers=2)
        assert m2.count_parameters() >= 3.5 * m1.count_parameters()

    def test_lstm_t4_forget_bias(self):
        m = LSTMModel()
        b = m.lstm.bias_hh_l0.detach()
        assert b.shape[0] == 4 * 64
        # We initialized forget bias to 1.0 (some block)
        assert torch.any(b == 1.0)

    def test_lstm_t5_gradients(self):
        m = LSTMModel()
        x = torch.randn(8, 10, 5)
        m(x).sum().backward()
        for param in m.parameters():
            assert param.grad is not None

    def test_lstm_t6_eval_dropout(self):
        m = LSTMModel(num_layers=2)
        m.eval()
        x = torch.randn(8, 10, 5)
        out1 = m(x)
        out2 = m(x)
        assert torch.allclose(out1, out2)

    def test_lstm_t7_outperforms_mlp_smoke(self):
        m_lstm = LSTMModel()
        m_mlp = MLPModel()
        
        opt_l = torch.optim.Adam(m_lstm.parameters(), lr=0.01)
        opt_m = torch.optim.Adam(m_mlp.parameters(), lr=0.01)
        
        X_seq = torch.randn(16, 10, 14)
        X_flat = X_seq[:, -1, :]
        y = torch.randn(16, 1)
        
        for _ in range(5):
            opt_l.zero_grad()
            loss_l = nn.MSELoss()(m_lstm(X_seq), y)
            loss_l.backward()
            opt_l.step()
            
            opt_m.zero_grad()
            loss_m = nn.MSELoss()(m_mlp(X_flat), y)
            loss_m.backward()
            opt_m.step()
            
        assert loss_l.item() < 2.0
