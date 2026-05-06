"""
End-to-end integration tests for SDK and pipeline.
"""
import pytest
import os
import subprocess
from src.sdk.sdk import SDK
from src.shared.config import get_setup

def test_6i_1_run_all_end_to_end(tmp_path, sample_setup_config, monkeypatch):
    import json
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    
    with open(cfg_dir / "setup.json", "w") as f:
        sample_setup_config["training"]["max_epochs"] = 1
        sample_setup_config["data"]["duration_seconds"] = 1
        json.dump(sample_setup_config, f)
        
    with open(cfg_dir / "rate_limits.json", "w") as f:
        json.dump({"services": {}}, f)
        
    with open(cfg_dir / "logging_config.json", "w") as f:
        json.dump({"version": 1, "loggers": {"": {"handlers": []}}}, f)
        
    monkeypatch.setenv("FREQ_EXTRACTOR_CONFIG_DIR", str(cfg_dir))
    monkeypatch.chdir(str(tmp_path))
    
    import src.shared.config
    src.shared.config.get_setup.cache_clear()
    src.shared.config.get_rate_limits.cache_clear()
    src.shared.config.get_logging_config.cache_clear()
    
    # Needs to clear out gatekeepers
    import src.shared.gatekeeper as gk
    gk._gatekeepers.clear()

    sdk = SDK()
    sdk.run_all()
    
    # Check results
    assert (tmp_path / "data" / "train.npz").exists()
    assert (tmp_path / "results" / "checkpoints" / "best_mlp.pt").exists()
    assert (tmp_path / "results" / "checkpoints" / "best_rnn.pt").exists()
    assert (tmp_path / "results" / "checkpoints" / "best_lstm.pt").exists()

def test_6i_4_cli_invocation():
    # Use python -m to test the entrypoint
    # We won't run full training as it takes too long, we will test invalid args
    env = os.environ.copy()
    res = subprocess.run(
        ["python", "src/main.py", "--mode", "evaluate", "--model", "invalid_model"],
        capture_output=True, text=True, env=env
    )
    assert res.returncode != 0
    assert "invalid choice" in res.stderr
