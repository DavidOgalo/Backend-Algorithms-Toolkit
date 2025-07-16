"""
API Rate Limiting with Token Bucket Algorithm (Production-Ready)
---------------------------------------------------------------
Implements a flexible, extensible, and robust token bucket rate limiting system for backend APIs and services.

Features:
- Token bucket algorithm for burst tolerance and smooth rate limiting
- Per-client and per-endpoint rate limiting
- Configurable bucket size, refill rate, and time window
- Extensible for custom endpoints and dynamic rules
- Accurate retry-after calculation for throttled requests
- Comprehensive docstrings and error handling

Use Cases:
- API request throttling (REST, GraphQL, RPC)
- Auth, upload, and custom endpoint rate limiting
- Distributed systems and microservices
- Preventing abuse, DoS, and enforcing SLAs
"""
from datetime import datetime
from typing import Dict, Any 
from collections import defaultdict
from abc import ABC, abstractmethod

class RateLimitExceeded(Exception):
    """
    Raised when the rate limit is exceeded for a client or endpoint.
    Includes a recommended retry-after time in seconds.
    """
    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded, retry after {retry_after} seconds")

class RateLimiter(ABC):
    """
    Abstract base class for rate limiters. Defines the contract for extensible rate limiting strategies.
    """
    @abstractmethod
    def is_allowed(self, key: str) -> bool:
        """Check if a request is allowed for the given key."""
        pass

    @abstractmethod
    def get_remaining_requests(self, key: str) -> int:
        """Get the number of remaining requests for the given key."""
        pass

#Concrete Implementation

class TokenBucketRateLimiter(RateLimiter):
    """
    Token bucket rate limiter implementation.

    Supports:
    - Smooth, gradual token refilling (not bulk resets)
    - Burst tolerance and strict rate enforcement
    - Per-client and per-endpoint buckets
    - Accurate retry-after calculation
    """
    def __init__(self, max_requests: int, time_window: int):
        """
        Initialize a token bucket rate limiter.
        :param max_requests: Maximum tokens (requests) allowed in the time window
        :param time_window: Time window in seconds for rate limiting
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.buckets: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'tokens': max_requests,
            'last_refill': datetime.now()
        })

    def _refill_bucket(self, key: str):
        """
        Gradually refill tokens in the bucket based on elapsed time.
        Allows for smooth refilling and burst tolerance.
        """
        bucket = self.buckets[key]
        now = datetime.now()
        time_passed = (now - bucket['last_refill']).total_seconds()
        rate_per_second = self.max_requests / self.time_window
        tokens_to_add = int(time_passed * rate_per_second)
        if tokens_to_add > 0:
            bucket['tokens'] = min(self.max_requests, bucket['tokens'] + tokens_to_add)
            bucket['last_refill'] = now

    def is_allowed(self, key: str) -> bool:
        """
        Check if a request is allowed for the given key.
        Refills the bucket first. If any tokens are available, deduct 1 and allow the request.
        """
        self._refill_bucket(key)
        bucket = self.buckets[key]
        if bucket['tokens'] > 0:
            bucket['tokens'] -= 1
            return True
        return False

    def get_remaining_requests(self, key: str) -> int:
        """
        Get the number of remaining requests (tokens) for the given key.
        Refills the bucket before reporting.
        """
        self._refill_bucket(key)
        bucket = self.buckets[key]
        return bucket['tokens']

    def get_retry_after_seconds(self, key: str) -> int:
        """
        Calculate seconds until the next token is available for the given key.
        Returns 0 if a token is available now.
        """
        bucket = self.buckets[key]
        now = datetime.now()
        time_passed = (now - bucket['last_refill']).total_seconds()
        rate_per_second = self.max_requests / self.time_window
        if bucket['tokens'] >= 1:
            return 0
        # Calculate time until next token becomes available
        time_until_next_token = (1 - (time_passed * rate_per_second)) / rate_per_second
        return max(1, int(time_until_next_token))

        

class APIRateLimitManager:
    """
    Manages rate limiting for different API endpoints and clients.

    Features:
    - Per-endpoint and per-client rate limiting
    - Extensible for custom endpoints and dynamic rules
    - Centralized management for backend APIs
    """
    def __init__(self):
        # Stores different rate limit rules per endpoint type
        self.limiters = {
            'default': TokenBucketRateLimiter(100, 3600),   # 100 requests/hour
            'auth': TokenBucketRateLimiter(5, 300),         # 5 requests/5min
            'upload': TokenBucketRateLimiter(10, 3600)      # 10 uploads/hour
        }

    def check_rate_limit(self, client_id: str, endpoint_type: str = 'default') -> bool:
        """
        Check if a client can make a request to the specified endpoint.
        Raises RateLimitExceeded if not allowed.
        """
        limiter = self.limiters.get(endpoint_type, self.limiters['default'])
        key = f"{client_id}:{endpoint_type}"
        if not limiter.is_allowed(key):
            retry_after = limiter.get_retry_after_seconds(key)
            raise RateLimitExceeded(retry_after=retry_after)
        return True

    def get_rate_limit_info(self, client_id: str, endpoint_type: str = 'default') -> Dict[str, int]:
        """
        Get rate limit information for a client and endpoint.
        Returns limit, remaining, and reset time.
        """
        limiter = self.limiters.get(endpoint_type, self.limiters['default'])
        key = f"{client_id}:{endpoint_type}"
        return {
            'limit': limiter.max_requests,
            'remaining': limiter.get_remaining_requests(key),
            'reset_time': limiter.time_window
        }


# Usage Example
if __name__ == "__main__":
    rate_limiter = APIRateLimitManager()
    client_id = "user123"
    try:
        # Check auth endpoint
        rate_limiter.check_rate_limit(client_id, 'auth')
        print("Auth request allowed")

        # Get rate limit info
        info = rate_limiter.get_rate_limit_info(client_id, 'auth')
        print(f"Rate limit info: {info}")

    except RateLimitExceeded as e:
        print(f"Rate Limit Error: {e}")