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
from ..memory.short_term_memory import ShortTermMemory
from ..memory.long_term_memory import LongTermMemory
from ..memory.vector_memory import VectorMemory
from ..memory.conversation_memory import ConversationMemory
from ..config.settings import settings


class HybridMemorySystem:
    """
    Manages all memory subsystems for the medical AI
    """
    
    def __init__(self, 
                 redis_url: str,
                 postgres_url: str,
                 neo4j_url: str,
                 neo4j_auth: tuple):
        self.redis_url = redis_url
        self.postgres_url = postgres_url
        self.neo4j_url = neo4j_url
        self.neo4j_auth = neo4j_auth
        
        # Memory subsystems
        self.short_term = None
        self.long_term = None
        self.vector_memory = None
        self.conversation_memory = None
        
        # Embedding model
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        
    async def initialize(self):
        """Initialize all memory subsystems"""
        # Initialize Redis for short-term memory
        self.redis_client = await redis.from_url(self.redis_url)
        self.short_term = ShortTermMemory(self.redis_client)
        
        # Initialize PostgreSQL for long-term memory
        # self.postgres_pool = await asyncpg.create_pool(self.postgres_url)
        # self.long_term = LongTermMemory(self.postgres_pool)
        
        # Initialize vector memory
        # self.vector_memory = VectorMemory(self.postgres_pool, self.embedding_model)
        
        # Initialize conversation memory
        self.conversation_memory = ConversationMemory(
            postgres_pool=None,
            redis_client=self.redis_client,
            embedding_model=self.embedding_model
        )
        
        # Initialize Neo4j for patient context
        self.neo4j_driver = AsyncGraphDatabase.driver(
            self.neo4j_url,
            auth=self.neo4j_auth
        )
        
    async def close(self):
        """Close all connections"""
        if self.redis_client:
            await self.redis_client.close()
        # if self.postgres_pool:
        #     await self.postgres_pool.close()
        if self.neo4j_driver:
            await self.neo4j_driver.close()
            
    async def get_patient_context(self, patient_id: str) -> Dict[str, Any]:
        """Get comprehensive patient context"""
        context = {}
        
        # Get from long-term memory
        patient_data = await self.long_term.get_patient_data(patient_id)
        if patient_data:
            context.update(patient_data)
            
        # Get recent interactions from short-term
        recent = await self.short_term.get_recent_interactions(patient_id)
        context["recent_interactions"] = recent
        
        # Get from Neo4j knowledge graph
        async with self.neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (p:Patient {id: $patient_id})
                OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Condition)
                OPTIONAL MATCH (p)-[:TAKES_MEDICATION]->(m:Medication)
                OPTIONAL MATCH (p)-[:HAS_ALLERGY]->(a:Allergy)
                RETURN p, collect(DISTINCT c) as conditions, 
                       collect(DISTINCT m) as medications,
                       collect(DISTINCT a) as allergies
                """,
                patient_id=patient_id
            )
            
            record = await result.single()
            if record:
                context["conditions"] = [dict(c) for c in record["conditions"]]
                context["medications"] = [dict(m) for m in record["medications"]]
                context["allergies"] = [dict(a) for a in record["allergies"]]
                
        return context
        
    async def add_conversation_entry(self, 
                                   session_id: str,
                                   patient_id: Optional[str],
                                   message: Dict[str, Any]):
        """Add a conversation entry to memory"""
        await self.conversation_memory.add_message(
            session_id=session_id,
            patient_id=patient_id,
            message=message
        )
        
    async def get_conversation_history(self, 
                                     session_id: str,
                                     limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        return await self.conversation_memory.get_history(session_id, limit)
        
    async def search_similar_cases(self, 
                                 symptoms: List[str],
                                 conditions: List[str],
                                 limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar medical cases"""
        # Create query embedding
        query_text = " ".join(symptoms + conditions)
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        # Search in vector memory
        similar_cases = await self.vector_memory.search_similar(
            embedding=query_embedding,
            filters={"type": "medical_case"},
            limit=limit
        )
        
        return similar_cases
        
    async def store_session_state(self, session_id: str, state: Dict[str, Any]):
        """Store session state for recovery"""
        await self.short_term.set(
            f"session:{session_id}",
            state,
            ttl=86400  # 24 hours
        )
        
        # Also store in long-term for persistence
        await self.long_term.store_session(session_id, state)