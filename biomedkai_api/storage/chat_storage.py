"""
Chat storage service for persisting conversations to PostgreSQL
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
import json
import structlog

from ..config.settings import settings

logger = structlog.get_logger()


class ChatStorage:
    """Service for storing and retrieving chat conversations"""
    
    def __init__(self):
        """Initialize database connection"""
        database_url = f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.logger = structlog.get_logger(name="chat_storage")
    
    def save_chat(self, chat_id: str, user_id: Optional[str] = None, title: Optional[str] = None, metadata: Optional[Dict] = None) -> bool:
        """Save or update a chat session"""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO chats (id, user_id, title, metadata, updated_at)
                        VALUES (:id, :user_id, :title, :metadata, CURRENT_TIMESTAMP)
                        ON CONFLICT (id) 
                        DO UPDATE SET 
                            title = COALESCE(EXCLUDED.title, chats.title),
                            metadata = COALESCE(EXCLUDED.metadata, chats.metadata),
                            updated_at = CURRENT_TIMESTAMP
                    """),
                    {
                        "id": chat_id,
                        "user_id": user_id,
                        "title": title or self._generate_title_from_messages(chat_id),
                        "metadata": json.dumps(metadata) if metadata else None
                    }
                )
                conn.commit()
                return True
        except Exception as e:
            self.logger.error("Error saving chat", error=str(e), chat_id=chat_id)
            return False
    
    def save_message(self, chat_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> Optional[int]:
        """Save a message to the database"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("""
                        INSERT INTO messages (chat_id, role, content, metadata)
                        VALUES (:chat_id, :role, :content, :metadata)
                        RETURNING id
                    """),
                    {
                        "chat_id": chat_id,
                        "role": role,
                        "content": content,
                        "metadata": json.dumps(metadata) if metadata else None
                    }
                )
                conn.commit()
                message_id = result.fetchone()[0]
                return message_id
        except Exception as e:
            self.logger.error("Error saving message", error=str(e), chat_id=chat_id)
            return None
    
    def save_chat_metadata(self, chat_id: str, message_id: Optional[int], kg_data: Optional[Dict] = None, 
                          references: Optional[List[Dict]] = None, agent_selection: Optional[Dict] = None) -> bool:
        """Save metadata (KG data, references, agent selection) for a chat"""
        try:
            with self.engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO chat_metadata (chat_id, message_id, kg_data, references_data, agent_selection)
                        VALUES (:chat_id, :message_id, :kg_data, :references_data, :agent_selection)
                    """),
                    {
                        "chat_id": chat_id,
                        "message_id": message_id,
                        "kg_data": json.dumps(kg_data) if kg_data else None,
                        "references_data": json.dumps(references) if references else None,
                        "agent_selection": json.dumps(agent_selection) if agent_selection else None
                    }
                )
                conn.commit()
                return True
        except Exception as e:
            self.logger.error("Error saving chat metadata", error=str(e), chat_id=chat_id)
            return False
    
    def get_chat(self, chat_id: str) -> Optional[Dict]:
        """Retrieve a chat with all messages"""
        try:
            with self.engine.connect() as conn:
                # Get chat info
                chat_result = conn.execute(
                    text("SELECT id, user_id, title, created_at, updated_at, metadata FROM chats WHERE id = :id"),
                    {"id": chat_id}
                )
                chat_row = chat_result.fetchone()
                
                if not chat_row:
                    return None
                
                # Get messages
                messages_result = conn.execute(
                    text("SELECT id, role, content, metadata, created_at FROM messages WHERE chat_id = :chat_id ORDER BY created_at"),
                    {"chat_id": chat_id}
                )
                
                messages = []
                for msg in messages_result:
                    messages.append({
                        "id": msg[0],
                        "role": msg[1],
                        "content": msg[2],
                        "metadata": json.loads(msg[3]) if msg[3] else None,
                        "created_at": msg[4].isoformat() if msg[4] else None
                    })
                
                return {
                    "id": chat_row[0],
                    "user_id": chat_row[1],
                    "title": chat_row[2],
                    "created_at": chat_row[3].isoformat() if chat_row[3] else None,
                    "updated_at": chat_row[4].isoformat() if chat_row[4] else None,
                    "metadata": json.loads(chat_row[5]) if chat_row[5] else None,
                    "messages": messages
                }
        except Exception as e:
            self.logger.error("Error retrieving chat", error=str(e), chat_id=chat_id)
            return None
    
    def get_chat_metadata(self, chat_id: str, message_id: Optional[int] = None) -> Optional[Dict]:
        """Retrieve metadata for a chat or specific message"""
        try:
            with self.engine.connect() as conn:
                if message_id:
                    result = conn.execute(
                        text("""
                            SELECT kg_data, references_data, agent_selection 
                            FROM chat_metadata 
                            WHERE chat_id = :chat_id AND message_id = :message_id
                            ORDER BY created_at DESC LIMIT 1
                        """),
                        {"chat_id": chat_id, "message_id": message_id}
                    )
                else:
                    result = conn.execute(
                        text("""
                            SELECT kg_data, references_data, agent_selection 
                            FROM chat_metadata 
                            WHERE chat_id = :chat_id
                            ORDER BY created_at DESC LIMIT 1
                        """),
                        {"chat_id": chat_id}
                    )
                
                row = result.fetchone()
                if not row:
                    return None
                
                return {
                    "kg_data": json.loads(row[0]) if row[0] else None,
                    "references": json.loads(row[1]) if row[1] else None,
                    "agent_selection": json.loads(row[2]) if row[2] else None
                }
        except Exception as e:
            self.logger.error("Error retrieving chat metadata", error=str(e), chat_id=chat_id)
            return None
    
    def list_chats(self, user_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Dict]:
        """List chats for a user"""
        try:
            with self.engine.connect() as conn:
                if user_id:
                    result = conn.execute(
                        text("""
                            SELECT id, title, created_at, updated_at 
                            FROM chats 
                            WHERE user_id = :user_id 
                            ORDER BY updated_at DESC 
                            LIMIT :limit OFFSET :offset
                        """),
                        {"user_id": user_id, "limit": limit, "offset": offset}
                    )
                else:
                    result = conn.execute(
                        text("""
                            SELECT id, title, created_at, updated_at 
                            FROM chats 
                            ORDER BY updated_at DESC 
                            LIMIT :limit OFFSET :offset
                        """),
                        {"limit": limit, "offset": offset}
                    )
                
                chats = []
                for row in result:
                    chats.append({
                        "id": row[0],
                        "title": row[1],
                        "created_at": row[2].isoformat() if row[2] else None,
                        "updated_at": row[3].isoformat() if row[3] else None
                    })
                
                return chats
        except Exception as e:
            self.logger.error("Error listing chats", error=str(e))
            return []
    
    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat and all associated messages"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("DELETE FROM chats WHERE id = :id"), {"id": chat_id})
                conn.commit()
                return True
        except Exception as e:
            self.logger.error("Error deleting chat", error=str(e), chat_id=chat_id)
            return False
    
    def _generate_title_from_messages(self, chat_id: str) -> str:
        """Generate a title from the first user message"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT content FROM messages 
                        WHERE chat_id = :chat_id AND role = 'user' 
                        ORDER BY created_at LIMIT 1
                    """),
                    {"chat_id": chat_id}
                )
                row = result.fetchone()
                if row:
                    title = row[0][:50]  # First 50 characters
                    return title + "..." if len(row[0]) > 50 else title
                return "New Chat"
        except Exception:
            return "New Chat"
    
    def close(self):
        """Close database connection"""
        self.engine.dispose()

