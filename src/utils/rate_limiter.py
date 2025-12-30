import time
from collections import deque


class RateLimiter:
    """
    A rate limiter that enforces a maximum number of calls within a time period.
    Uses a sliding window algorithm to track request timestamps.
    """
    
    def __init__(self, max_calls, period_seconds):
        """
        Initialize the rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed within the period
            period_seconds: Time period in seconds
        """
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.request_timestamps = deque()
    
    def _clean_old_requests(self, current_time):
        """
        Remove request timestamps that are outside the current time window.
        
        Args:
            current_time: Current timestamp to use as reference
        """
        cutoff_time = current_time - self.period_seconds
        while self.request_timestamps and self.request_timestamps[0] <= cutoff_time:
            self.request_timestamps.popleft()
    
    def allow_request(self):
        """
        Check if a request should be allowed.
        
        Returns:
            True if the request is allowed, False if it exceeds the rate limit
        """
        current_time = time.time()
        self._clean_old_requests(current_time)
        
        if len(self.request_timestamps) < self.max_calls:
            self.request_timestamps.append(current_time)
            return True
        else:
            return False
    
    def wait_for_token(self):
        """
        Block until a token becomes available.
        Sleeps until the oldest request in the window expires.
        """
        current_time = time.time()
        self._clean_old_requests(current_time)
        
        # If we're under the limit, no need to wait
        if len(self.request_timestamps) < self.max_calls:
            return
        
        # Calculate how long to wait until the oldest request expires
        oldest_request_time = self.request_timestamps[0]
        wait_time = (oldest_request_time + self.period_seconds) - current_time
        
        # Add a small buffer to ensure the window has passed
        if wait_time > 0:
            time.sleep(wait_time)
        
        # Now we can add the request
        current_time = time.time()
        self._clean_old_requests(current_time)
        self.request_timestamps.append(current_time)

