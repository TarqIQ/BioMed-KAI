from functools import wraps
from typing import Dict, Optional
import time
import asyncio
from collections import defaultdict, deque
from fastapi import HTTPException, Request
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Thread-safe rate limiter using sliding window algorithm"""
    
    def __init__(self):
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, key: str, max_requests: int, period: int) -> bool:
        """
        Check if request is allowed based on rate limiting rules
        
        Args:
            key: Unique identifier (e.g., user_id, IP address)
            max_requests: Maximum number of requests allowed
            period: Time period in seconds
            
        Returns:
            True if request is allowed, False otherwise
        """
        async with self.lock:
            now = time.time()
            window_start = now - period
            
            # Clean old requests outside the window
            while self.requests[key] and self.requests[key][0] <= window_start:
                self.requests[key].popleft()
            
            # Check if we're under the limit
            if len(self.requests[key]) < max_requests:
                self.requests[key].append(now)
                return True
            
            return False
    
    async def get_reset_time(self, key: str, period: int) -> Optional[datetime]:
        """Get the time when the rate limit will reset"""
        async with self.lock:
            if not self.requests[key]:
                return None
            
            oldest_request = self.requests[key][0]
            reset_time = datetime.fromtimestamp(oldest_request + period)
            return reset_time

# Global rate limiter instance
_rate_limiter = RateLimiter()

def rate_limit(requests: int, period: int, per: str = "user"):
    """
    Rate limiting decorator for FastAPI endpoints
    
    Args:
        requests: Maximum number of requests allowed
        period: Time period in seconds
        per: Rate limiting scope ("user", "ip", or custom key)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request object and user info from kwargs
            request: Optional[Request] = None
            current_user: Optional[Dict] = None
            
            # Look for Request object in args (typically first argument for FastAPI)
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            # Look for current_user in kwargs (from Depends)
            current_user = kwargs.get('current_user')
            
            # Determine rate limiting key
            if per == "user" and current_user:
                key = f"user:{current_user.get('user_id', 'anonymous')}"
            elif per == "ip" and request:
                # Get real IP, considering proxies
                forwarded_for = request.headers.get("X-Forwarded-For")
                if forwarded_for:
                    client_ip = forwarded_for.split(",")[0].strip()
                else:
                    client_ip = request.client.host if request.client else "unknown"
                key = f"ip:{client_ip}"
            elif per == "user" and not current_user:
                # Fallback to IP if user not available
                if request:
                    forwarded_for = request.headers.get("X-Forwarded-For")
                    if forwarded_for:
                        client_ip = forwarded_for.split(",")[0].strip()
                    else:
                        client_ip = request.client.host if request.client else "unknown"
                    key = f"ip:{client_ip}"
                else:
                    key = "anonymous"
            else:
                key = f"{per}:default"
            
            # Check rate limit
            if not await _rate_limiter.is_allowed(key, requests, period):
                reset_time = await _rate_limiter.get_reset_time(key, period)
                
                logger.warning(f"Rate limit exceeded for key: {key}")
                
                # Add rate limit headers
                headers = {
                    "X-RateLimit-Limit": str(requests),
                    "X-RateLimit-Period": str(period),
                    "X-RateLimit-Remaining": "0"
                }
                
                if reset_time:
                    headers["X-RateLimit-Reset"] = reset_time.isoformat()
                
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Maximum {requests} requests per {period} seconds.",
                    headers=headers
                )
            
            # Execute the original function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def rate_limit_key(key_func):
    """
    Rate limiting decorator with custom key function
    
    Args:
        key_func: Function that returns a rate limiting key based on request context
    """
    def decorator(requests: int, period: int):
        def inner_decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate custom key
                key = key_func(*args, **kwargs)
                
                # Check rate limit
                if not await _rate_limiter.is_allowed(key, requests, period):
                    reset_time = await _rate_limiter.get_reset_time(key, period)
                    
                    logger.warning(f"Rate limit exceeded for custom key: {key}")
                    
                    headers = {
                        "X-RateLimit-Limit": str(requests),
                        "X-RateLimit-Period": str(period),
                        "X-RateLimit-Remaining": "0"
                    }
                    
                    if reset_time:
                        headers["X-RateLimit-Reset"] = reset_time.isoformat()
                    
                    raise HTTPException(
                        status_code=429,
                        detail=f"Rate limit exceeded. Maximum {requests} requests per {period} seconds.",
                        headers=headers
                    )
                
                return await func(*args, **kwargs)
            
            return wrapper
        return inner_decorator
    return decorator

async def cleanup_old_entries():
    """Periodic cleanup of old rate limiting entries"""
    while True:
        try:
            async with _rate_limiter.lock:
                current_time = time.time()
                keys_to_remove = []
                
                for key, timestamps in _rate_limiter.requests.items():
                    # Remove entries older than 1 hour
                    cutoff_time = current_time - 3600
                    while timestamps and timestamps[0] <= cutoff_time:
                        timestamps.popleft()
                    
                    # Remove empty entries
                    if not timestamps:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del _rate_limiter.requests[key]
            
            # Sleep for 5 minutes before next cleanup
            await asyncio.sleep(300)
            
        except Exception as e:
            logger.error(f"Error during rate limiter cleanup: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying