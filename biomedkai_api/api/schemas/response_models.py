from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime

class ChatResponse(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., description="AI response message")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the response")
    agent_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata from involved agents")
    timestamp: str = Field(..., description="Response timestamp in ISO format")


class PatientContextResponse(BaseModel):
    patient_id: str = Field(..., description="Patient identifier")
    context: Dict[str, Any] = Field(..., description="Patient context and medical history")
    last_updated: str = Field(..., description="Last update timestamp in ISO format")


class SearchResponse(BaseModel):
    query: str = Field(..., description="Original search query")
    results: Dict[str, Any] = Field(..., description="Search results by source")
    total_results: int = Field(..., description="Total number of results found")
    timestamp: str = Field(..., description="Search timestamp in ISO format")


class AnalysisResponse(BaseModel):
    analysis_id: str = Field(..., description="Unique analysis identifier")
    analysis_type: str = Field(..., description="Type of analysis performed")
    results: str = Field(..., description="Analysis results")
    confidence_scores: Dict[str, float] = Field(default_factory=dict, description="Confidence scores by category")
    recommendations: List[str] = Field(default_factory=list, description="Analysis recommendations")
    timestamp: str = Field(..., description="Analysis timestamp in ISO format")


class SessionSummaryResponse(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    summary: Dict[str, Any] = Field(..., description="Session summary data")
    message_count: int = Field(..., description="Number of messages in session")
    agents_involved: int = Field(..., description="Number of agents involved")
    average_confidence: float = Field(..., ge=0.0, le=1.0, description="Average confidence score")
    timestamp: str = Field(..., description="Summary timestamp in ISO format")
