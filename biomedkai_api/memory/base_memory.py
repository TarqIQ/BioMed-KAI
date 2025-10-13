from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid


class MemoryEntry:
    """Represents a single memory entry"""
    
    def __init__(self, 
                 content: Any,
                 metadata: Optional[Dict[str, Any]] = None,
                 embedding: Optional[List[float]] = None):
        self.id = str(uuid.uuid4())
        self.content = content
        self.metadata = metadata or {}
        self.embedding = embedding
        self.timestamp = datetime.utcnow()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        entry = cls(
            content=data["content"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding")
        )
        entry.id = data["id"]
        entry.timestamp = datetime.fromisoformat(data["timestamp"])
        return entry


class BaseMemory(ABC):
    """Abstract base class for memory implementations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
    @abstractmethod
    async def add(self, 
                  key: str,
                  content: Any,
                  metadata: Optional[Dict[str, Any]] = None,
                  embedding: Optional[List[float]] = None) -> str:
        """Add a memory entry"""
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by key"""
        pass
    
    @abstractmethod
    async def search(self, 
                     query: str,
                     limit: int = 10,
                     filters: Optional[Dict[str, Any]] = None) -> List[MemoryEntry]:
        """Search memory entries"""
        pass
    
    @abstractmethod
    async def update(self, 
                     key: str,
                     content: Optional[Any] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update a memory entry"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a memory entry"""
        pass
    
    @abstractmethod
    async def clear(self) -> int:
        """Clear all memory entries"""
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """Get the number of entries in memory"""
        pass