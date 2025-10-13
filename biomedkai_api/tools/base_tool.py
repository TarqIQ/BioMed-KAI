from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential


class BaseTool(ABC):
    """Base class for all tools"""
    
    def __init__(self, 
                 name: str,
                 description: str,
                 config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.config = config or {}
        self.logger = structlog.get_logger(name=f"tool.{name}")
        self.cache = {}
        self.rate_limiter = self._create_rate_limiter()
        
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass
    
    @abstractmethod
    def validate_params(self, **kwargs) -> bool:
        """Validate input parameters"""
        pass
    
    def _create_rate_limiter(self):
        """Create rate limiter for API calls"""
        max_calls = self.config.get("rate_limit", 100)
        period = self.config.get("rate_period", 60)
        
        class RateLimiter:
            def __init__(self, max_calls, period):
                self.max_calls = max_calls
                self.period = period
                self.calls = []
                
            async def acquire(self):
                now = datetime.utcnow()
                # Remove old calls
                self.calls = [call for call in self.calls 
                             if (now - call).total_seconds() < self.period]
                
                if len(self.calls) >= self.max_calls:
                    sleep_time = self.period - (now - self.calls[0]).total_seconds()
                    await asyncio.sleep(max(0, sleep_time))
                    await self.acquire()
                else:
                    self.calls.append(now)
        
        return RateLimiter(max_calls, period)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def execute_with_retry(self, **kwargs) -> Dict[str, Any]:
        """Execute tool with retry logic"""
        await self.rate_limiter.acquire()
        
        self.logger.info(f"Executing {self.name}", params=kwargs)
        
        try:
            # Validate parameters
            if not self.validate_params(**kwargs):
                raise ValueError(f"Invalid parameters for {self.name}")
            
            # Check cache
            cache_key = self._get_cache_key(**kwargs)
            if cache_key in self.cache:
                self.logger.info(f"Cache hit for {self.name}")
                return self.cache[cache_key]
            
            # Execute tool
            result = await self.execute(**kwargs)
            
            # Cache result
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Tool {self.name} failed", error=str(e))
            raise
    
    def _get_cache_key(self, **kwargs) -> str:
        """Generate cache key from parameters"""
        import hashlib
        import json
        
        key_data = json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear the tool's cache"""
        self.cache = {}
        self.logger.info(f"Cache cleared for {self.name}")