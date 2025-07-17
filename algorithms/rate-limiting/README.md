# Rate Limiting Algorithms

Production-grade rate limiting algorithms for backend services, APIs, and distributed systems. This module is designed to be extensible, supporting multiple rate limiting strategies for different backend scenarios.

---

## Features

- Extensible architecture for multiple rate limiting algorithms
- Per-client and per-endpoint rate limiting
- Configurable bucket size, refill rate, and time window (for token bucket)
- Accurate retry-after calculation for throttled requests
- Centralized management for backend APIs
- Comprehensive error handling and docstrings

---

## Algorithms Included

- **Token Bucket:** Burst-tolerant, time-based rate limiting (first implementation)
- **APIRateLimitManager:** Centralized management for multiple endpoints and clients
- _More algorithms coming soon (e.g., Leaky Bucket, Fixed Window, Sliding Window, Dynamic Rate Limiting)_

---

## Use Cases

- API request throttling (REST, GraphQL, RPC)
- Auth, upload, and custom endpoint rate limiting
- Distributed systems and microservices
- Preventing abuse, DoS, and enforcing SLAs

---

## Usage

See [`token_bucket.py`](./token_bucket.py) for the first implementation and advanced usage examples. Future implementations will follow a similar extensible interface.

### Example: API Rate Limiting (Token Bucket)

```python
from token_bucket import APIRateLimitManager, RateLimitExceeded

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
```

---

For more details, see the code and docstrings in [`token_bucket.py`](./token_bucket.py). Future rate limiting algorithms will be documented here as they are added.
