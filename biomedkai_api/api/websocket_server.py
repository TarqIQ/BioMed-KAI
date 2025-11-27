from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any, Optional
import json
import asyncio
import uuid
from datetime import datetime
import structlog

from ..core.orchestrator import MedicalAgentOrchestrator
from ..api.middleware.authentication import verify_websocket_token


websocket_router = APIRouter()
logger = structlog.get_logger()

# Active WebSocket connections
active_connections: Dict[str, WebSocket] = {}


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        
    async def connect(self, websocket: WebSocket, session_id: str, user_id: Optional[str] = None):
        """Accept new connection"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        
        if user_id:
            self.user_sessions[user_id] = session_id
            
        logger.info("WebSocket connected", session_id=session_id, user_id=user_id)
        
    def disconnect(self, session_id: str):
        """Remove connection"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            
        # Remove from user sessions
        user_id = None
        for uid, sid in self.user_sessions.items():
            if sid == session_id:
                user_id = uid
                break
                
        if user_id:
            del self.user_sessions[user_id]
            
        logger.info("WebSocket disconnected", session_id=session_id, user_id=user_id)
        
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send message to specific connection"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            await websocket.send_json(message)
            
    async def broadcast(self, message: Dict[str, Any], exclude: Optional[str] = None):
        """Broadcast message to all connections"""
        for session_id, websocket in self.active_connections.items():
            if session_id != exclude:
                await websocket.send_json(message)


# Global connection manager
manager = ConnectionManager()

@websocket_router.websocket("/chat")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = None
):
    """
    WebSocket endpoint for medical AI chat
    """
    # Verify authentication if token provided
    session_id = str(uuid.uuid4())  # Generate a unique session ID
    user_id = None
    if token:
        try:
            user_data = await verify_websocket_token(token)
            user_id = user_data.get("user_id")
        except Exception as e:
            await websocket.close(code=4001, reason="Authentication failed")
            return
            
    # Connect
    await manager.connect(websocket, session_id, user_id)
    
    try:
        # Send initial connection message
        await manager.send_message(session_id, {
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Get orchestrator instance
        from ..main import orchestrator
        
        # Handle messages
        while True:
            # Receive message
            data = await websocket.receive_json()
            print(f"Received data: {data}")  # Debugging line
            message_type = data.get("type")
            action = data.get("action", "chat")
            print(f"Message type: {message_type}, Action: {action}")  # Debugging line
            if message_type == "chat_message" or action == "chat":
                await handle_chat_message(session_id, data, orchestrator)
                
            elif message_type == "get_history":
                await handle_get_history(session_id, data, orchestrator)
                
            elif message_type == "ping":
                await manager.send_message(session_id, {"type": "pong"})
                
            else:
                await manager.send_message(session_id, {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
                
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        
    except Exception as e:
        logger.error("WebSocket error", session_id=session_id, error=str(e))
        await manager.send_message(session_id, {
            "type": "error",
            "message": "Internal server error"
        })
        manager.disconnect(session_id)


@websocket_router.websocket("/chat/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    token: Optional[str] = None
):
    """
    WebSocket endpoint for medical AI chat
    """
    # Verify authentication if token provided
    user_id = None
    if token:
        try:
            user_data = await verify_websocket_token(token)
            user_id = user_data.get("user_id")
        except Exception as e:
            await websocket.close(code=4001, reason="Authentication failed")
            return
            
    # Connect
    await manager.connect(websocket, session_id, user_id)
    
    try:
        # Send initial connection message
        await manager.send_message(session_id, {
            "type": "connection_established",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Get orchestrator instance
        from .main import orchestrator
        
        # Handle messages
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            message_type = data.get("type")
            
            if message_type == "chat_message":
                await handle_chat_message(session_id, data, orchestrator)
                
            elif message_type == "get_history":
                await handle_get_history(session_id, data, orchestrator)
                
            elif message_type == "ping":
                await manager.send_message(session_id, {"type": "pong"})
                
            else:
                await manager.send_message(session_id, {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
                
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        
    except Exception as e:
        logger.error("WebSocket error", session_id=session_id, error=str(e))
        await manager.send_message(session_id, {
            "type": "error",
            "message": "Internal server error"
        })
        manager.disconnect(session_id)


async def handle_chat_message(
    session_id: str,
    data: Dict[str, Any],
    orchestrator: MedicalAgentOrchestrator
):
    """Handle incoming chat message with proper streaming"""
    
    query = data.get("prompt", "")
    patient_id = data.get("patient_id")
    context = data.get("context", {})
    
    if not query:
        logger.warning("Received empty message", session_id=session_id)
        await manager.send_message(session_id, {
            "type": "error",
            "message": "Empty message"
        })
        return
        
    # Send acknowledgment
    await manager.send_message(session_id, {
        "type": "message_received",
        "timestamp": datetime.utcnow().isoformat()
    })
    
    # Send typing indicator
    await manager.send_message(session_id, {
        "type": "agent_typing",
        "agent": "processing"
    })

    logger.info("Processing chat message", session_id=session_id, query=query[:100], patient_id=patient_id)
    
    try:
        # Generate unique message ID for this conversation
        message_id = str(uuid.uuid4())
        
        # Start streaming response
        full_response = ""
        chunk_count = 0
        
        # Extract user_id from data if available
        user_id = data.get("user_id") or data.get("userId")
        
        async for chunk in orchestrator.process_query(
            query=query, 
            patient_id=patient_id, 
            context=context,
            chat_id=session_id,
            user_id=user_id
        ):
            chunk_count += 1
            full_response += chunk

            # print(f"Sending chunk {chunk_count}: {chunk}")
            
            # Send each chunk as it's generated
            await manager.send_message(session_id, {
                "type": "stream_delta",
                "message_id": message_id,
                "chunk": chunk,
                "delta": chunk,  # Assuming chunk is the delta
                "content": full_response,
                "chunk_number": chunk_count,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Small delay to prevent overwhelming the websocket
            await asyncio.sleep(0.01)
        
        # Send completion message
        await manager.send_message(session_id, {
            "type": "stream_end",
            "message_id": message_id,
            "full_response": full_response,
            "total_chunks": chunk_count,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info("Message processing completed", 
                   session_id=session_id, 
                   message_id=message_id,
                   chunks_sent=chunk_count)
        
    except Exception as e:
        logger.error("Error processing message", session_id=session_id, error=str(e))
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"Failed to process message: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        })


async def handle_get_history(
    session_id: str,
    data: Dict[str, Any],
    orchestrator: MedicalAgentOrchestrator
):
    """Handle request for conversation history"""
    
    limit = data.get("limit", 10)
    
    try:
        # Get history from memory system
        from .main import memory_system
        
        history = await memory_system.get_conversation_history(session_id, limit)
        
        await manager.send_message(session_id, {
            "type": "history",
            "messages": history,
            "count": len(history),
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error("Error fetching history", session_id=session_id, error=str(e))
        await manager.send_message(session_id, {
            "type": "error",
            "message": "Failed to fetch history"
        })