from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from src.agents.base_agent import BaseMedicalAgent
from src.core.message_protocol import AgentMessage, MessageType, Priority, AgentRole
from src.core.state_manager import MedicalAssistantState


class SupervisorAgent(BaseMedicalAgent):
    """
    Central orchestrator that routes queries to appropriate specialist agents
    """
    
    def __init__(self, model: Any, tools: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(
            name="supervisor",
            role=AgentRole.SUPERVISOR.value,
            model=model,
            tools=tools,
            config=config
        )
        
        self.routing_prompt = """
You are a medical supervisor agent responsible for routing queries to the appropriate specialist.

Available agents:
- diagnostic: For symptom analysis, differential diagnosis, and medical condition assessment
- treatment: For treatment planning, therapy recommendations, and care protocols
- drug_interaction: For medication checks, drug interactions, and pharmaceutical safety
- research: For medical literature, clinical studies, and evidence-based information
- web_search: For real-time information, latest medical news, and general searches
- general: For urgent cases requiring immediate assessment
- validation: For verifying and validating medical recommendations

Query: {query}
Medical Entities Detected: {entities}
Patient Context: {context}
Current Symptoms: {symptoms}

Based on the query and context, determine:
1. Primary agent to handle this query
2. Whether this is an emergency requiring immediate triage
3. Confidence level in routing decision (0-1)
4. Any secondary agents that might be needed

Respond in JSON format:
{{
    "primary_agent": "agent_name",
    "is_emergency": boolean,
    "confidence": float,
    "secondary_agents": ["agent1", "agent2"],
    "reasoning": "explanation of routing decision"
}}
"""
        
    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Supervisor can always process"""
        return True
        
    async def process(self, 
                     state: MedicalAssistantState,
                     message: Optional[AgentMessage] = None) -> Dict[str, Any]:
        """Process incoming query and route to appropriate agent"""
        
        # Extract query from latest message
        latest_message = state["messages"][-1] if state["messages"] else {}
        query = latest_message.get("content", "")
        
        # Detect medical entities using tools
        entities = await self._detect_entities(query)
        
        # Check for emergency keywords
        is_emergency = self._check_emergency(query, entities)
        
        # Determine routing
        routing_decision = await self._determine_routing(
            query=query,
            entities=entities,
            context=state.get("patient_context", {}),
            symptoms=state.get("symptoms", [])
        )
        
        # Update state
        updates = {
            "medical_entities": entities,
            "emergency_flag": is_emergency or routing_decision.get("is_emergency", False),
            "current_agent": routing_decision["primary_agent"],
            "agent_states": {
                self.name: "completed",
                routing_decision["primary_agent"]: "pending"
            }
        }
        
        # Log routing decision
        await self.log_decision(
            decision=f"Route to {routing_decision['primary_agent']}",
            reasoning=routing_decision.get("reasoning", ""),
            confidence=routing_decision.get("confidence", 0.8),
            state=state
        )
        
        # Send handoff message to next agent
        await self.send_message(
            recipient=routing_decision["primary_agent"],
            message_type=MessageType.HANDOFF,
            content={
                "query": query,
                "entities": entities,
                "context": state.get("patient_context", {}),
                "priority": Priority.EMERGENCY if is_emergency else Priority.MEDIUM
            },
            priority=Priority.EMERGENCY if is_emergency else Priority.MEDIUM
        )
        
        return updates
        
    async def _detect_entities(self, query: str) -> List[Dict[str, Any]]:
        """Detect medical entities in the query"""
        if "entity_extractor" in self.tools:
            try:
                result = await self.call_tool("entity_extractor", {"text": query})
                return result.get("entities", [])
            except Exception as e:
                self.logger.warning(f"Entity extraction failed: {e}, using fallback")
                # Fallback: simple keyword detection
                return self._fallback_entity_detection(query)
        return []
    
    def _fallback_entity_detection(self, query: str) -> List[Dict[str, Any]]:
        """Simple fallback entity detection"""
        entities = []
        query_lower = query.lower()
        
        # Common medical terms
        symptom_keywords = ["fever", "pain", "cough", "headache", "nausea", "fatigue"]
        condition_keywords = ["diabetes", "hypertension", "cancer", "infection", "disease"]
        
        for symptom in symptom_keywords:
            if symptom in query_lower:
                entities.append({
                    "text": symptom,
                    "label": "SYMPTOM",
                    "score": 0.7
                })
                
        for condition in condition_keywords:
            if condition in query_lower:
                entities.append({
                    "text": condition,
                    "label": "CONDITION",
                    "score": 0.7
                })
                
        return entities
        
    def _check_emergency(self, query: str, entities: List[Dict[str, Any]]) -> bool:
        """Check if query contains emergency indicators"""
        emergency_keywords = [
            "emergency", "urgent", "critical", "severe", "acute",
            "chest pain", "difficulty breathing", "unconscious", "bleeding",
            "stroke", "heart attack", "anaphylaxis", "seizure"
        ]
        
        query_lower = query.lower()
        
        # Check keywords
        if any(keyword in query_lower for keyword in emergency_keywords):
            return True
            
        # Check entities
        for entity in entities:
            if entity.get("type") == "EMERGENCY" or \
               any(keyword in entity.get("text", "").lower() for keyword in emergency_keywords):
                return True
                
        return False
        
    async def _determine_routing(self, 
                                query: str,
                                entities: List[Dict[str, Any]],
                                context: Dict[str, Any],
                                symptoms: List[str]) -> Dict[str, Any]:
        """Use LLM to determine routing"""
        
        # Format entities for prompt
        entities_str = json.dumps([{
            "text": e.get("text", ""),
            "type": e.get("label", ""),
            "confidence": e.get("score", 1.0)
        } for e in entities], indent=2)
        
        prompt = self.routing_prompt.format(
            query=query,
            entities=entities_str,
            context=json.dumps(context, indent=2),
            symptoms=", ".join(symptoms) if symptoms else "None reported"
        )
        
        response = ""
        async for chunk in self.generate_llm_response(prompt):
            response += chunk
            
        try:
            # Parse JSON response
            routing = json.loads(response)
            
            # Validate routing
            valid_agents = ["diagnostic", "treatment", "drug_interaction", 
                          "research", "web_search", "general", "validation"]
            
            if routing.get("primary_agent") not in valid_agents:
                routing["primary_agent"] = "diagnostic"  # Default
                
            return routing
            
        except json.JSONDecodeError:
            # Fallback routing logic
            if symptoms or any(e.get("type") == "SYMPTOM" for e in entities):
                return {
                    "primary_agent": "diagnostic",
                    "is_emergency": False,
                    "confidence": 0.7,
                    "secondary_agents": [],
                    "reasoning": "Symptoms detected, routing to diagnostic agent"
                }
            else:
                return {
                    "primary_agent": "research",
                    "is_emergency": False,
                    "confidence": 0.6,
                    "secondary_agents": [],
                    "reasoning": "General query, routing to research agent"
                }
    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Validate supervisor input"""
        messages = state.get("messages", [])
        return len(messages) > 0
    
    async def process(self, state: MedicalAssistantState) -> Dict[str, Any]:
        """Process supervisor request"""
        return await self.process_with_streaming(state)
    
    def _get_system_prompt(self) -> str:
        return """You are a Medical Supervisor AI Assistant. Your role is to:

1. **Analyze incoming medical queries** and determine the appropriate medical domain
2. **Extract key medical entities** from patient queries (symptoms, conditions, medications)
3. **Assess urgency level** and flag emergency situations
4. **Coordinate care** by directing queries to appropriate specialists
5. **Provide initial assessment** and general medical guidance

**Key Responsibilities:**
- Triage medical queries by urgency and specialty
- Extract and categorize medical information
- Provide comprehensive initial assessments
- Guide patients toward appropriate care pathways
- Ensure safety and accuracy in all recommendations

**Safety Guidelines:**
- Always recommend professional medical consultation for serious symptoms
- Never provide definitive diagnoses
- Flag potential emergency situations immediately
- Acknowledge limitations and uncertainties
- Emphasize evidence-based information"""
    
    def _get_relevant_entity_types(self) -> List[str]:
        return ["Disease", "Symptom", "Drug", "Condition", "Procedure"]
    
    def _determine_next_agent(self, state: MedicalAssistantState, response: str) -> str:
        """Determine which agent should handle the query next"""
        response_lower = response.lower()
        
        # Emergency detection
        emergency_indicators = ["emergency", "urgent", "immediate", "critical", "severe"]
        if any(indicator in response_lower for indicator in emergency_indicators):
            return "general"
        
        # Diagnostic indicators
        diagnostic_indicators = ["diagnosis", "condition", "disease", "symptoms"]
        if any(indicator in response_lower for indicator in diagnostic_indicators):
            return "diagnostic"
        
        # Treatment indicators
        treatment_indicators = ["treatment", "therapy", "medication", "management"]
        if any(indicator in response_lower for indicator in treatment_indicators):
            return "treatment"
        
        # Drug interaction indicators
        drug_indicators = ["interaction", "side effect", "contraindication"]
        if any(indicator in response_lower for indicator in drug_indicators):
            return "drug_interaction"
        
        # Research indicators
        research_indicators = ["research", "study", "clinical trial", "latest evidence"]
        if any(indicator in response_lower for indicator in research_indicators):
            return "research"
        
        return "end"
    
    async def _get_agent_specific_updates(self, response: str, context: Dict[str, Any], state: MedicalAssistantState) -> Dict[str, Any]:
        """Extract medical entities and update state"""
        updates = {}
        
        # Extract entities from knowledge graph context
        kg_context = context.get("knowledge_graph", {})
        entities = kg_context.get("entities", [])
        
        # Update symptoms, conditions, medications based on entities
        symptoms = []
        conditions = []
        medications = []
        
        for entity in entities:
            labels = entity.get("labels", [])
            name = entity.get("name", "")
            
            if "Symptom" in labels:
                symptoms.append(name)
            elif "Disease" in labels or "Condition" in labels:
                conditions.append(name)
            elif "Drug" in labels or "Medication" in labels:
                medications.append(name)
        
        if symptoms:
            updates["symptoms"] = list(set(state.get("symptoms", []) + symptoms))
        if conditions:
            updates["conditions"] = list(set(state.get("conditions", []) + conditions))
        if medications:
            updates["medications"] = list(set(state.get("medications", []) + medications))
        
        # Check for emergency flags
        emergency_keywords = ["emergency", "urgent", "critical", "severe pain", "chest pain", "difficulty breathing"]
        if any(keyword in response.lower() for keyword in emergency_keywords):
            updates["emergency_flag"] = True
        
        return updates
            
# class SupervisorAgent(BaseMedicalAgent):
#     """Supervisor agent that coordinates the workflow"""
    
#     def __init__(self, model: Any, tools: Dict[str, Any], config: Dict[str, Any]):
#         super().__init__("supervisor", model, tools, config)
    
#     async def validate_input(self, state: MedicalAssistantState) -> bool:
#         """Validate supervisor input"""
#         messages = state.get("messages", [])
#         return len(messages) > 0
    
#     async def process(self, state: MedicalAssistantState) -> Dict[str, Any]:
#         """Process supervisor request"""
#         return await self.process_with_streaming(state)
    
#     def _get_system_prompt(self) -> str:
#         return """You are a Medical Supervisor AI Assistant. Your role is to:

# 1. **Analyze incoming medical queries** and determine the appropriate medical domain
# 2. **Extract key medical entities** from patient queries (symptoms, conditions, medications)
# 3. **Assess urgency level** and flag emergency situations
# 4. **Coordinate care** by directing queries to appropriate specialists
# 5. **Provide initial assessment** and general medical guidance

# **Key Responsibilities:**
# - Triage medical queries by urgency and specialty
# - Extract and categorize medical information
# - Provide comprehensive initial assessments
# - Guide patients toward appropriate care pathways
# - Ensure safety and accuracy in all recommendations

# **Safety Guidelines:**
# - Always recommend professional medical consultation for serious symptoms
# - Never provide definitive diagnoses
# - Flag potential emergency situations immediately
# - Acknowledge limitations and uncertainties
# - Emphasize evidence-based information"""
    
#     def _get_relevant_entity_types(self) -> List[str]:
#         return ["Disease", "Symptom", "Drug", "Condition", "Procedure"]
    
#     def _determine_next_agent(self, state: MedicalAssistantState, response: str) -> str:
#         """Determine which agent should handle the query next"""
#         response_lower = response.lower()
        
#         # Emergency detection
#         emergency_indicators = ["emergency", "urgent", "immediate", "critical", "severe"]
#         if any(indicator in response_lower for indicator in emergency_indicators):
#             return "general"
        
#         # Diagnostic indicators
#         diagnostic_indicators = ["diagnosis", "condition", "disease", "symptoms"]
#         if any(indicator in response_lower for indicator in diagnostic_indicators):
#             return "diagnostic"
        
#         # Treatment indicators
#         treatment_indicators = ["treatment", "therapy", "medication", "management"]
#         if any(indicator in response_lower for indicator in treatment_indicators):
#             return "treatment"
        
#         # Drug interaction indicators
#         drug_indicators = ["interaction", "side effect", "contraindication"]
#         if any(indicator in response_lower for indicator in drug_indicators):
#             return "drug_interaction"
        
#         # Research indicators
#         research_indicators = ["research", "study", "clinical trial", "latest evidence"]
#         if any(indicator in response_lower for indicator in research_indicators):
#             return "research"
        
#         return "end"
    
#     async def _get_agent_specific_updates(self, response: str, context: Dict[str, Any], state: MedicalAssistantState) -> Dict[str, Any]:
#         """Extract medical entities and update state"""
#         updates = {}
        
#         # Extract entities from knowledge graph context
#         kg_context = context.get("knowledge_graph", {})
#         entities = kg_context.get("entities", [])
        
#         # Update symptoms, conditions, medications based on entities
#         symptoms = []
#         conditions = []
#         medications = []
        
#         for entity in entities:
#             labels = entity.get("labels", [])
#             name = entity.get("name", "")
            
#             if "Symptom" in labels:
#                 symptoms.append(name)
#             elif "Disease" in labels or "Condition" in labels:
#                 conditions.append(name)
#             elif "Drug" in labels or "Medication" in labels:
#                 medications.append(name)
        
#         if symptoms:
#             updates["symptoms"] = list(set(state.get("symptoms", []) + symptoms))
#         if conditions:
#             updates["conditions"] = list(set(state.get("conditions", []) + conditions))
#         if medications:
#             updates["medications"] = list(set(state.get("medications", []) + medications))
        
#         # Check for emergency flags
#         emergency_keywords = ["emergency", "urgent", "critical", "severe pain", "chest pain", "difficulty breathing"]
#         if any(keyword in response.lower() for keyword in emergency_keywords):
#             updates["emergency_flag"] = True
        
#         return updates
