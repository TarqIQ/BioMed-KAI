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

class VectorMemory:
    """
    Vector similarity search using pgvector
    """
    
    def __init__(self, postgres_pool: asyncpg.Pool, embedding_model: SentenceTransformer):
        self.pool = postgres_pool
        self.embedding_model = embedding_model
        
    async def add_document(self, 
                          document_id: str,
                          text: str,
                          metadata: Dict[str, Any]) -> str:
        """Add document with vector embedding"""
        # Generate embedding
        embedding = self.embedding_model.encode(text).tolist()
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO vector_documents (id, text, metadata, embedding)
                VALUES ($1, $2, $3, $4::vector)
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    updated_at = CURRENT_TIMESTAMP
                """,
                document_id, text, json.dumps(metadata), embedding
            )
            
        return document_id
        
    async def search_similar(self, 
                           embedding: List[float],
                           filters: Optional[Dict[str, Any]] = None,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        where_clauses = []
        params = [embedding, limit]
        param_count = 2
        
        if filters:
            for key, value in filters.items():
                param_count += 1
                params.append(value)
                where_clauses.append(f"metadata->>{key} = ${param_count}")
                
        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        query_sql = f"""
        SELECT id, text, metadata, 
               1 - (embedding <=> $1::vector) as similarity
        FROM vector_documents
        {where_clause}
        ORDER BY embedding <=> $1::vector
        LIMIT $2
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query_sql, *params)
            
        return [
            {
                "id": row["id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]),
                "similarity": float(row["similarity"])
            }
            for row in rows
        ]