"""
Rate limiting and IO gatekeeper implementation.
"""

import time
import logging
from typing import Callable, Any, Dict
from collections import deque

logger = logging.getLogger("freq_extractor.gatekeeper")

class ApiGatekeeper:
    """
    Controls execution rate and retries for I/O operations.
    Implements a sliding window rate limiter and exponential backoff retry.
    """
    
    def __init__(self, name: str, calls_per_minute: int, max_retries: int, timeout_seconds: int):
        self.name = name
        self.limit = calls_per_minute
        self.window = 60.0
        self.max_retries = max_retries
        self.timeout = timeout_seconds
        self.timestamps: deque = deque()

    def _wait_if_needed(self) -> None:
        """Blocks until the operation is allowed by the rate limit window."""
        now = time.time()
        # Remove timestamps older than the window
        while self.timestamps and self.timestamps[0] <= now - self.window:
            self.timestamps.popleft()
            
        if len(self.timestamps) >= self.limit:
            wait_time = (self.timestamps[0] + self.window) - now
            if wait_time > 0:
                logger.debug(f"{self.name} rate limited. Waiting {wait_time:.2f}s")
                time.sleep(wait_time)
            # Recursively re-check after waiting
            self._wait_if_needed()
        else:
            self.timestamps.append(time.time())

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Executes the callable with retries, backoff, and rate limiting."""
        attempt = 0
        while attempt <= self.max_retries:
            self._wait_if_needed()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                attempt += 1
                if attempt > self.max_retries:
                    logger.error(f"{self.name} operation failed after {attempt} attempts: {e}")
                    raise
                
                backoff = (2 ** attempt) * 0.1
                logger.warning(f"{self.name} failed (attempt {attempt}). Retrying in {backoff}s. Error: {e}")
                time.sleep(backoff)
                
    def get_queue_status(self) -> Dict[str, Any]:
        """Returns the current status of the gatekeeper queue."""
        now = time.time()
        valid_stamps = [t for t in self.timestamps if t > now - self.window]
        return {
            "name": self.name,
            "used": len(valid_stamps),
            "limit": self.limit,
            "window": self.window
        }

# Singleton registry
_gatekeepers: Dict[str, ApiGatekeeper] = {}

def get_gatekeeper(service_name: str) -> ApiGatekeeper:
    """Factory function for retrieving or instantiating Gatekeepers."""
    from .config import get_rate_limits
    
    if service_name not in _gatekeepers:
        limits = get_rate_limits().get("services", {})
        config = limits.get(service_name, limits.get("default", {
            "calls_per_minute": 60,
            "max_retries": 3,
            "timeout_seconds": 10
        }))
        
        _gatekeepers[service_name] = ApiGatekeeper(
            name=service_name,
            calls_per_minute=config.get("calls_per_minute", 60),
            max_retries=config.get("max_retries", 3),
            timeout_seconds=config.get("timeout_seconds", 10)
        )
        
    return _gatekeepers[service_name]
