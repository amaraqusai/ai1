"""
Tests for shared infrastructure.
"""
import pytest
from src.shared.version import parse_version, validate_config_version
from src.shared.gatekeeper import Gatekeeper, get_gatekeeper, GatekeeperError

def test_parse_version():
    assert parse_version("1.00") == 1.0
    assert parse_version("invalid") == 0.0

def test_validate_config_version():
    validate_config_version("1.00")
    validate_config_version("1.50")
    with pytest.raises(ValueError):
        validate_config_version("0.99")
    with pytest.raises(ValueError):
        validate_config_version("2.00")

def dummy_func(x, fail_times=0):
    dummy_func.calls += 1
    if dummy_func.calls <= fail_times:
        raise OSError("failed")
    return x

def test_gatekeeper_success():
    gk = Gatekeeper("test")
    dummy_func.calls = 0
    res = gk.execute(dummy_func, 42)
    assert res == 42
    assert gk.total_calls == 1

def test_gatekeeper_retries():
    gk = Gatekeeper("test", max_retries=3, base_delay=0.01)
    dummy_func.calls = 0
    res = gk.execute(dummy_func, 100, fail_times=2)
    assert res == 100
    assert gk.total_calls == 1
    assert gk.total_retries == 2

def test_gatekeeper_raises():
    gk = Gatekeeper("test", max_retries=1, base_delay=0.01)
    dummy_func.calls = 0
    with pytest.raises(GatekeeperError):
        gk.execute(dummy_func, 42, fail_times=3)

def test_gatekeeper_singleton():
    gk1 = get_gatekeeper("db")
    gk2 = get_gatekeeper("db")
    assert id(gk1) == id(gk2)

def test_gatekeeper_get_status():
    gk = Gatekeeper("test")
    stats = gk.get_status()
    assert "total_calls" in stats
    assert "errors" in stats
    assert "retries" in stats

def test_shared_config_cache(tmp_path, monkeypatch):
    import json
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "setup.json"
    with open(cfg_file, "w") as f:
        json.dump({"project_name": "test", "version": "1.00"}, f)
        
    monkeypatch.setenv("FREQ_EXTRACTOR_CONFIG_DIR", str(cfg_dir))
    
    from src.shared.config import get_setup
    get_setup.cache_clear()
    
    c1 = get_setup()
    c2 = get_setup()
    assert id(c1) == id(c2)
    
    # modify file
    with open(cfg_file, "w") as f:
        json.dump({"project_name": "mod", "version": "1.00"}, f)
        
    c3 = get_setup()
    assert c3["project_name"] == "test"
    
    get_setup.cache_clear()
    c4 = get_setup()
    assert c4["project_name"] == "mod"

def test_shared_config_not_found(monkeypatch):
    monkeypatch.setenv("FREQ_EXTRACTOR_CONFIG_DIR", "/nonexistent")
    from src.shared.config import get_setup
    get_setup.cache_clear()
    with pytest.raises(FileNotFoundError):
        get_setup()

def test_gatekeeper_rate_limiter(monkeypatch):
    import time
    from src.shared.gatekeeper import SlidingWindowRateLimiter
    
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=1)
    
    calls = []
    def mock_sleep(s):
        calls.append(s)
        
    monkeypatch.setattr(time, "sleep", mock_sleep)
    
    limiter.wait_if_needed()
    limiter.wait_if_needed()
    assert len(calls) == 0
    
    limiter.wait_if_needed()
    assert len(calls) == 1
    assert calls[0] > 0


