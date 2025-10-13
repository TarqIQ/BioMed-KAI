from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

#     ChatRequest, PatientContextRequest, SearchRequest,
#     AnalysisRequest, FeedbackRequest

class ChatRequest(BaseModel):
    message: str = Field(..., description="The chat message from the user")
    patient_id: Optional[str] = Field(None, description="Optional patient ID for context")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the query")

class PatientContextRequest(BaseModel):
    context: Dict[str, Any] = Field(..., description="Patient context data to update")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    sources: List[str] = Field(..., description="List of sources to search")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    limit: Optional[int] = Field(10, description="Maximum number of results per source")

class AnalysisRequest(BaseModel):
    analysis_type: str = Field(..., description="Type of analysis to perform")
    data: Dict[str, Any] = Field(..., description="Data to analyze")
    patient_id: Optional[str] = Field(None, description="Associated patient ID")

class FeedbackRequest(BaseModel):
    session_id: str = Field(..., description="Session ID for the feedback")
    message_id: str = Field(..., description="Specific message ID being rated")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    feedback: Optional[str] = Field(None, description="Optional text feedback")