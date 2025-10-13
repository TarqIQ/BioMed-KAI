from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
import json
from enum import Enum


class AgentState(str, Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING_FOR_TOOL = "waiting_for_tool"
    WAITING_FOR_HUMAN = "waiting_for_human"
    ERROR = "error"
    COMPLETED = "completed"


class MedicalAssistantState(TypedDict):
    """Global state for the medical assistant workflow"""
    # Conversation state
    messages: List[Dict[str, Any]]
    current_agent: str
    agent_states: Dict[str, str]
    
    # Medical context
    patient_context: Dict[str, Any]
    medical_entities: List[Dict[str, Any]]
    symptoms: List[str]
    conditions: List[str]
    medications: List[str]
    allergies: List[str]
    
    # Analysis results
    diagnosis_history: List[Dict[str, Any]]
    treatment_plans: List[Dict[str, Any]]
    drug_interactions: List[Dict[str, Any]]
    research_findings: List[Dict[str, Any]]
    
    # Recommendations and validation
    recommendations: List[str]
    validation_results: Dict[str, Any]
    confidence_scores: Dict[str, float]
    
    # Control flow
    requires_human_review: bool
    emergency_flag: bool
    tool_results: Dict[str, Any]
    error_log: List[Dict[str, Any]]
    
    # Metadata
    session_id: str
    patient_id: Optional[str]
    timestamp: str
    total_agents_involved: int


class StateManager:
    """Manages the global state of the medical assistant"""
    
    def __init__(self):
        self.state_history: List[MedicalAssistantState] = []
        self.checkpoints: Dict[str, MedicalAssistantState] = {}
        
    def create_initial_state(self, 
                           session_id: str,
                           patient_id: Optional[str] = None) -> MedicalAssistantState:
        """Create initial state for a new session"""
        return {
            "messages": [],
            "current_agent": "supervisor",
            "agent_states": {},
            "patient_context": {},
            "medical_entities": [],
            "symptoms": [],
            "conditions": [],
            "medications": [],
            "allergies": [],
            "diagnosis_history": [],
            "treatment_plans": [],
            "drug_interactions": [],
            "research_findings": [],
            "recommendations": [],
            "validation_results": {},
            "confidence_scores": {},
            "requires_human_review": False,
            "emergency_flag": False,
            "tool_results": {},
            "error_log": [],
            "session_id": session_id,
            "patient_id": patient_id,
            "timestamp": datetime.utcnow().isoformat(),
            "total_agents_involved": 1
        }
    
    def update_state(self, 
                    current_state: MedicalAssistantState,
                    updates: Dict[str, Any]) -> MedicalAssistantState:
        """Update state with new values"""
        new_state = current_state.copy()
        
        for key, value in updates.items():
            if key in new_state:
                if isinstance(new_state[key], list) and isinstance(value, list):
                    # Append to lists
                    new_state[key].extend(value)
                elif isinstance(new_state[key], dict) and isinstance(value, dict):
                    # Merge dictionaries
                    new_state[key].update(value)
                else:
                    # Replace value
                    new_state[key] = value
        
        # Update timestamp
        new_state["timestamp"] = datetime.utcnow().isoformat()
        
        # Store in history
        self.state_history.append(new_state)
        
        return new_state
    
    def create_checkpoint(self, 
                         state: MedicalAssistantState,
                         checkpoint_name: str):
        """Create a named checkpoint of the current state"""
        self.checkpoints[checkpoint_name] = state.copy()
    
    def restore_checkpoint(self, checkpoint_name: str) -> Optional[MedicalAssistantState]:
        """Restore state from a checkpoint"""
        return self.checkpoints.get(checkpoint_name)
    
    def get_state_summary(self, state: MedicalAssistantState) -> Dict[str, Any]:
        """Get a summary of the current state"""
        return {
            "session_id": state["session_id"],
            "patient_id": state["patient_id"],
            "current_agent": state["current_agent"],
            "message_count": len(state["messages"]),
            "agents_involved": state["total_agents_involved"],
            "has_diagnosis": len(state["diagnosis_history"]) > 0,
            "has_treatment_plan": len(state["treatment_plans"]) > 0,
            "requires_human_review": state["requires_human_review"],
            "emergency_flag": state["emergency_flag"],
            "average_confidence": sum(state["confidence_scores"].values()) / len(state["confidence_scores"]) if state["confidence_scores"] else 0,
            "timestamp": state["timestamp"]
        }
