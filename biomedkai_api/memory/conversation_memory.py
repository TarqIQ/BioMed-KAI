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

class ConversationMemory:
    """
    Specialized memory for conversation history
    """
    
    def __init__(self, 
                 postgres_pool: asyncpg.Pool,
                 redis_client: redis.Redis,
                 embedding_model: SentenceTransformer):
        self.postgres_pool = postgres_pool
        self.redis = redis_client
        self.embedding_model = embedding_model
        
    async def add_message(self, 
                         session_id: str,
                         patient_id: Optional[str],
                         message: Dict[str, Any]):
        """Add message to conversation history"""
        # Generate embedding for message content
        content = message.get("content", "")
        embedding = self.embedding_model.encode(content).tolist() if content else None
        
        # Store in PostgreSQL
        async with self.postgres_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO conversation_memory 
                (thread_id, patient_id, message_role, message_content, metadata, embedding)
                VALUES ($1, $2, $3, $4, $5, $6::vector)
                """,
                session_id, patient_id, message.get("role", "user"),
                content, json.dumps(message.get("metadata", {})), embedding
            )
            
        # Cache recent messages in Redis
        cache_key = f"conversation:{session_id}"
        await self.redis.lpush(cache_key, json.dumps(message))
        await self.redis.ltrim(cache_key, 0, 99)  # Keep last 100 messages
        await self.redis.expire(cache_key, 3600)  # 1 hour TTL
        
    async def get_history(self, 
                         session_id: str,
                         limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history"""
        # Try cache first
        cache_key = f"conversation:{session_id}"
        cached = await self.redis.lrange(cache_key, 0, limit - 1)
        
        if cached:
            return [json.loads(msg) for msg in cached]
            
        # Fallback to database
        async with self.postgres_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT * FROM conversation_memory
                WHERE thread_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                session_id, limit
            )
            
        messages = []
        for row in reversed(rows):  # Reverse to get chronological order
            messages.append({
                "role": row["message_role"],
                "content": row["message_content"],
                "metadata": json.loads(row["metadata"]),
                "timestamp": row["created_at"].isoformat()
            })
            
        return messages