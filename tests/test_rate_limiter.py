import pytest
import time
from src.utils.rate_limiter import RateLimiter

def test_allow_request_within_limit():
    # Setup: Allow 5 calls every 10 seconds
    limiter = RateLimiter(max_calls=5, period_seconds=10)
    
    # Action & Assert: Consuming 5 tokens should be allowed
    for _ in range(5):
        assert limiter.allow_request() is True

def test_block_request_exceeding_limit():
    # Setup: Allow 2 calls every 60 seconds
    limiter = RateLimiter(max_calls=2, period_seconds=60)
    
    # Action: Consume 2 tokens
    limiter.allow_request()
    limiter.allow_request()
    
    # Assert: The 3rd call should be blocked
    assert limiter.allow_request() is False

def test_reset_limit_after_window():
    # Setup: Allow 1 call every 1 second
    limiter = RateLimiter(max_calls=1, period_seconds=1)
    
    # Action: Consume the token
    assert limiter.allow_request() is True
    assert limiter.allow_request() is False # Blocked
    
    # Wait for the window to pass
    time.sleep(1.1)
    
    # Assert: Should be allowed again
    assert limiter.allow_request() is True

def test_wait_for_token_logic():
    # Setup: Allow 1 call every 1 second
    limiter = RateLimiter(max_calls=1, period_seconds=1)
    
    # Consume 1
    limiter.allow_request()
    
    start_time = time.time()
    # This method should block (sleep) until a token is available
    limiter.wait_for_token() 
    end_time = time.time()
    
    # Assert: It should have waited at least a split second roughly matching the window
    # (We use 0.9 to account for slight execution delays)
    assert (end_time - start_time) >= 0.9