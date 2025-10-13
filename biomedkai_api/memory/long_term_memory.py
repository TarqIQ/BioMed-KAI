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

class LongTermMemory(BaseMemory):
    """
    PostgreSQL-based long-term memory
    """
    
    def __init__(self, postgres_pool: asyncpg.Pool):
        super().__init__()
        self.pool = postgres_pool
        
    async def add(self, 
                  key: str,
                  content: Any,
                  metadata: Optional[Dict[str, Any]] = None,
                  embedding: Optional[List[float]] = None) -> str:
        """Add entry to long-term memory"""
        entry = MemoryEntry(content, metadata, embedding)
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory_entries (id, key, content, metadata, embedding, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (key) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    updated_at = CURRENT_TIMESTAMP
                """,
                entry.id, key, json.dumps(entry.content), 
                json.dumps(entry.metadata), entry.embedding, entry.timestamp
            )
            
        return entry.id
        
    async def get(self, key: str) -> Optional[MemoryEntry]:
        """Get entry from long-term memory"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM memory_entries WHERE key = $1",
                key
            )
            
            if row:
                return self._row_to_entry(row)
                
        return None
        
    async def search(self, 
                     query: str,
                     limit: int = 10,
                     filters: Optional[Dict[str, Any]] = None) -> List[MemoryEntry]:
        """Search in long-term memory"""
        where_clauses = ["1=1"]
        params = []
        param_count = 0
        
        # Add text search
        if query:
            param_count += 1
            params.append(query)
            where_clauses.append(f"content::text ILIKE ${param_count}")
            
        # Add filters
        if filters:
            for key, value in filters.items():
                param_count += 1
                params.append(value)
                where_clauses.append(f"metadata->>${key} = ${param_count}")
                
        param_count += 1
        params.append(limit)
        
        query_sql = f"""
        SELECT * FROM memory_entries 
        WHERE {' AND '.join(where_clauses)}
        ORDER BY created_at DESC
        LIMIT ${param_count}
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query_sql, *params)
            
        return [self._row_to_entry(row) for row in rows]
        
    async def update(self, 
                     key: str,
                     content: Optional[Any] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update entry in long-term memory"""
        updates = []
        params = []
        param_count = 0
        
        if content is not None:
            param_count += 1
            params.append(json.dumps(content))
            updates.append(f"content = ${param_count}")
            
        if metadata is not None:
            param_count += 1
            params.append(json.dumps(metadata))
            updates.append(f"metadata = ${param_count}")
            
        if not updates:
            return False
            
        param_count += 1
        params.append(key)
        
        query_sql = f"""
        UPDATE memory_entries 
        SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP
        WHERE key = ${param_count}
        """
        
        async with self.pool.acquire() as conn:
            result = await conn.execute(query_sql, *params)
            
        return result.split()[-1] != "0"
        
    async def delete(self, key: str) -> bool:
        """Delete from long-term memory"""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM memory_entries WHERE key = $1",
                key
            )
            
        return result.split()[-1] != "0"
        
    async def clear(self) -> int:
        """Clear all long-term memory"""
        async with self.pool.acquire() as conn:
            result = await conn.execute("DELETE FROM memory_entries")
            
        return int(result.split()[-1])
        
    async def size(self) -> int:
        """Get number of entries"""
        async with self.pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM memory_entries")
            
        return count
        
    async def get_patient_data(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get patient data from long-term memory"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM patient_context 
                WHERE patient_id = $1 
                ORDER BY updated_at DESC 
                LIMIT 1
                """,
                patient_id
            )
            
            if row:
                return json.loads(row["context_data"])
                
        return None
        
    async def store_session(self, session_id: str, state: Dict[str, Any]):
        """Store session state"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO session_states (session_id, state_data, created_at)
                VALUES ($1, $2, CURRENT_TIMESTAMP)
                ON CONFLICT (session_id) DO UPDATE SET
                    state_data = EXCLUDED.state_data,
                    updated_at = CURRENT_TIMESTAMP
                """,
                session_id, json.dumps(state)
            )
            
    def _row_to_entry(self, row: asyncpg.Record) -> MemoryEntry:
        """Convert database row to MemoryEntry"""
        entry = MemoryEntry(
            content=json.loads(row["content"]),
            metadata=json.loads(row["metadata"]),
            embedding=row["embedding"]
        )
        entry.id = row["id"]
        entry.timestamp = row["created_at"]
        return entry