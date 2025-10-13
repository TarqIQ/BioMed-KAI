from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
import uuid
from pydantic import BaseModel, Field


class MessageType(Enum):
    QUERY = "query"
    RESPONSE = "response"
    HANDOFF = "handoff"
    VALIDATION = "validation"
    ERROR = "error"
    TOOL_CALL = "tool_call"
    CONTEXT_REQUEST = "context_request"
    EMERGENCY = "emergency"
    HUMAN_REVIEW = "human_review"


class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class AgentRole(Enum):
    SUPERVISOR = "supervisor"
    DIAGNOSTIC = "diagnostic"
    TREATMENT = "treatment"
    DRUG_INTERACTION = "drug_interaction"
    RESEARCH = "research"
    VALIDATION = "validation"
    WEB_SEARCH = "web_search"
    GENERAL = "general"
    PREVENTIVE = "preventive"


@dataclass
class AgentMessage:
    """Standard message format for inter-agent communication"""
    sender: str
    recipient: str
    message_type: MessageType
    content: Dict[str, Any]
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: Priority = Priority.MEDIUM
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "message_type": self.message_type.value,
            "content": self.content,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "parent_id": self.parent_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        return cls(
            sender=data["sender"],
            recipient=data["recipient"],
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            context=data["context"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            priority=Priority(data["priority"]),
            correlation_id=data["correlation_id"],
            parent_id=data.get("parent_id"),
            metadata=data.get("metadata", {})
        )