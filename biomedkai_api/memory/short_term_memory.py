import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import uuid

import redis.asyncio as redis
import asyncpg
from neo4j import AsyncGraphDatabase
import numpy as np
from sentence_transformers import SentenceTransformer

from ..memory.base_memory import BaseMemory, MemoryEntry
from ..config.settings import settings

class ShortTermMemory(BaseMemory):
    """
    Redis-based short-term memory for recent interactions
    """
    
    def __init__(self, redis_client: redis.Redis):
        super().__init__()
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour
        
    async def add(self, 
                  key: str,
                  content: Any,
                  metadata: Optional[Dict[str, Any]] = None,
                  embedding: Optional[List[float]] = None,
                  ttl: Optional[int] = None) -> str:
        """Add entry to short-term memory"""
        entry = MemoryEntry(content, metadata, embedding)
        
        # Store in Redis
        await self.redis.setex(
            key,
            ttl or self.default_ttl,
            json.dumps(entry.to_dict())
        )
        
        # Add to recent keys list
        await self.redis.lpush("recent_keys", key)
        await self.redis.ltrim("recent_keys", 0, 999)  # Keep last 1000
        
        return entry.id
        
    async def get(self, key: str) -> Optional[MemoryEntry]:
        """Get entry from short-term memory"""
        data = await self.redis.get(key)
        if data:
            return MemoryEntry.from_dict(json.loads(data))
        return None
        
    async def search(self, 
                     query: str,
                     limit: int = 10,
                     filters: Optional[Dict[str, Any]] = None) -> List[MemoryEntry]:
        """Search in short-term memory"""
        # Get recent keys
        recent_keys = await self.redis.lrange("recent_keys", 0, limit * 2)
        
        entries = []
        for key in recent_keys:
            entry = await self.get(key.decode())
            if entry and self._matches_filters(entry, filters):
                entries.append(entry)
                
        # Simple text matching for now
        if query:
            entries = [e for e in entries if query.lower() in str(e.content).lower()]
            
        return entries[:limit]
        
    async def update(self, 
                     key: str,
                     content: Optional[Any] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update entry in short-term memory"""
        entry = await self.get(key)
        if not entry:
            return False
            
        if content is not None:
            entry.content = content
        if metadata is not None:
            entry.metadata.update(metadata)
            
        # Update in Redis
        ttl = await self.redis.ttl(key)
        await self.redis.setex(
            key,
            ttl if ttl > 0 else self.default_ttl,
            json.dumps(entry.to_dict())
        )
        
        return True
        
    async def delete(self, key: str) -> bool:
        """Delete from short-term memory"""
        result = await self.redis.delete(key)
        return result > 0
        
    async def clear(self) -> int:
        """Clear all short-term memory"""
        keys = await self.redis.keys("*")
        if keys:
            return await self.redis.delete(*keys)
        return 0
        
    async def size(self) -> int:
        """Get number of entries"""
        return await self.redis.dbsize()
        
    async def get_recent_interactions(self, patient_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent interactions for a patient"""
        pattern = f"patient:{patient_id}:*"
        keys = await self.redis.keys(pattern)
        
        interactions = []
        for key in keys[:limit]:
            entry = await self.get(key.decode())
            if entry:
                interactions.append(entry.to_dict())
                
        return sorted(interactions, key=lambda x: x["timestamp"], reverse=True)
        
    def _matches_filters(self, entry: MemoryEntry, filters: Optional[Dict[str, Any]]) -> bool:
        """Check if entry matches filters"""
        if not filters:
            return True
            
        for key, value in filters.items():
            if key not in entry.metadata or entry.metadata[key] != value:
                return False
                
        return True

    async def set(self, 
                  key: str,
                  content: Any,
                  metadata: Optional[Dict[str, Any]] = None,
                  embedding: Optional[List[float]] = None,
                  ttl: Optional[int] = None) -> str:
        """Set entry in short-term memory (alias for add)"""
        return await self.add(key, content, metadata, embedding, ttl)