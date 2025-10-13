from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional, List
import json
import uuid
from datetime import datetime
from pydantic import BaseModel, Field

from ..api.schemas.request_models import (
    ChatRequest, PatientContextRequest, SearchRequest,
    AnalysisRequest, FeedbackRequest
)
from ..api.schemas.response_models import (
    ChatResponse, PatientContextResponse, SearchResponse,
    AnalysisResponse, SessionSummaryResponse
)
from ..api.middleware.authentication import get_current_user
from ..api.middleware.rate_limiting import rate_limit


api_router = APIRouter()


class HealthCheck(BaseModel):
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]


@api_router.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    from .main import orchestrator, memory_system
    
    services = {
        "orchestrator": "healthy" if orchestrator else "unhealthy",
        "memory": "healthy" if memory_system else "unhealthy",
        "database": "healthy",  # Check actual DB connection
        "model": "healthy"  # Check model status
    }
    
    return HealthCheck(
        status="healthy" if all(v == "healthy" for v in services.values()) else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        services=services
    )


@api_router.post("/chat", response_model=ChatResponse)
@rate_limit(requests=100, period=60)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """
    Process a medical query through the AI system
    """
    from .main import orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service unavailable")
        
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    # Process query
    response_chunks = []
    agent_metadata = {}
    
    async for chunk in orchestrator.process_query(
        query=request.message,
        patient_id=request.patient_id,
        context=request.context
    ):
        response_chunks.append(chunk)
        
    # Combine chunks
    full_response = "".join(response_chunks)
    
    # Get confidence and metadata from orchestrator state
    # This would be implemented based on your state management
    confidence = 0.85  # Placeholder
    
    # Background task to update analytics
    background_tasks.add_task(
        update_usage_analytics,
        user_id=current_user.get("user_id"),
        session_id=session_id,
        query_length=len(request.message)
    )
    
    return ChatResponse(
        session_id=session_id,
        message=full_response,
        confidence=confidence,
        agent_metadata=agent_metadata,
        timestamp=datetime.utcnow().isoformat()
    )


@api_router.post("/chat/stream")
@rate_limit(requests=50, period=60)
async def chat_stream(
    request: ChatRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Stream responses from the medical AI system
    """
    from .main import orchestrator
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service unavailable")
        
    async def generate():
        """Generate streaming response"""
        session_id = request.session_id or str(uuid.uuid4())
        
        # Send initial metadata
        yield json.dumps({
            "type": "session_start",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat()
        }) + "\n"
        
        # Stream response chunks
        async for chunk in orchestrator.process_query(
            query=request.message,
            patient_id=request.patient_id,
            context=request.context
        ):
            yield json.dumps({
                "type": "response_chunk",
                "chunk": chunk,
                "timestamp": datetime.utcnow().isoformat()
            }) + "\n"
            
        # Send completion
        yield json.dumps({
            "type": "session_complete",
            "timestamp": datetime.utcnow().isoformat()
        }) + "\n"
        
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )


@api_router.get("/patient/{patient_id}/context", response_model=PatientContextResponse)
async def get_patient_context(
    patient_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get patient context and medical history
    """
    from .main import memory_system
    
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system unavailable")
        
    # Check authorization
    if not await check_patient_access(current_user, patient_id):
        raise HTTPException(status_code=403, detail="Access denied")
        
    context = await memory_system.get_patient_context(patient_id)
    
    return PatientContextResponse(
        patient_id=patient_id,
        context=context,
        last_updated=datetime.utcnow().isoformat()
    )


@api_router.put("/patient/{patient_id}/context")
async def update_patient_context(
    patient_id: str,
    request: PatientContextRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Update patient context
    """
    from .main import memory_system
    
    if not memory_system:
        raise HTTPException(status_code=503, detail="Memory system unavailable")
        
    # Check authorization
    if not await check_patient_access(current_user, patient_id):
        raise HTTPException(status_code=403, detail="Access denied")
        
    # Update context
    await memory_system.long_term.add(
        key=f"patient:{patient_id}:context",
        content=request.context,
        metadata={"updated_by": current_user.get("user_id")}
    )
    
    return {"status": "success", "patient_id": patient_id}


@api_router.post("/search", response_model=SearchResponse)
async def search_medical_info(
    request: SearchRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Search medical information across multiple sources
    """
    from .main import tool_registry
    
    search_results = {}
    
    # Search across requested sources
    for source in request.sources:
        if source in tool_registry:
            tool = tool_registry[source]
            results = await tool.execute_with_retry(
                query=request.query,
                filters=request.filters,
                limit=request.limit
            )
            search_results[source] = results
            
    return SearchResponse(
        query=request.query,
        results=search_results,
        total_results=sum(len(r.get("results", [])) for r in search_results.values()),
        timestamp=datetime.utcnow().isoformat()
    )


@api_router.post("/analyze", response_model=AnalysisResponse)
async def analyze_medical_data(
    request: AnalysisRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Analyze medical data (symptoms, lab results, etc.)
    """
    from .main import orchestrator
    
    # Route to specific analysis agent based on type
    analysis_query = f"Analyze the following {request.analysis_type}: {json.dumps(request.data)}"
    
    results = []
    confidence_scores = {}
    
    async for chunk in orchestrator.process_query(
        query=analysis_query,
        patient_id=request.patient_id,
        context={"analysis_type": request.analysis_type}
    ):
        results.append(chunk)
        
    return AnalysisResponse(
        analysis_id=str(uuid.uuid4()),
        analysis_type=request.analysis_type,
        results="".join(results),
        confidence_scores=confidence_scores,
        recommendations=[],  # Extract from results
        timestamp=datetime.utcnow().isoformat()
    )


@api_router.get("/session/{session_id}/summary", response_model=SessionSummaryResponse)
async def get_session_summary(
    session_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """
    Get summary of a consultation session
    """
    from .main import memory_system
    
    # Get session history
    history = await memory_system.get_conversation_history(session_id)
    
    # Get session state
    state = await memory_system.short_term.get(f"session:{session_id}")
    
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
        
    # Extract summary data
    summary_data = {
        "symptoms_identified": state.content.get("symptoms", []),
        "diagnoses": [d["diagnosis"] for d in state.content.get("diagnosis_history", [])],
        "treatments_recommended": [t["plan"] for t in state.content.get("treatment_plans", [])],
        "follow_up_required": state.content.get("requires_human_review", False)
    }
    
    return SessionSummaryResponse(
        session_id=session_id,
        summary=summary_data,
        message_count=len(history),
        agents_involved=state.content.get("total_agents_involved", 0),
        average_confidence=0.85,  # Calculate from state
        timestamp=datetime.utcnow().isoformat()
    )


@api_router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    current_user: Dict = Depends(get_current_user)
):
    """
    Submit feedback on AI responses
    """
    from .main import memory_system
    
    # Store feedback
    await memory_system.long_term.add(
        key=f"feedback:{request.session_id}:{request.message_id}",
        content={
            "rating": request.rating,
            "feedback": request.feedback,
            "user_id": current_user.get("user_id"),
            "timestamp": datetime.utcnow().isoformat()
        },
        metadata={"type": "feedback"}
    )
    
    return {"status": "success", "feedback_id": str(uuid.uuid4())}


# Helper functions
async def check_patient_access(user: Dict, patient_id: str) -> bool:
    """Check if user has access to patient data"""
    # Implement your authorization logic
    # For now, return True
    return True


async def update_usage_analytics(user_id: str, session_id: str, query_length: int):
    """Update usage analytics in background"""
    # Implement analytics tracking
    pass