from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncGenerator
import structlog
from datetime import datetime
import uuid

from ..core.state_manager import MedicalAssistantState


class BaseMedicalAgent(ABC):
    """Base class for all medical agents with streaming support"""
    
    def __init__(self, 
                 name: str,
                 model: Any,
                 role: str,
                 tools: Dict[str, Any],
                 config: Dict[str, Any]):
        self.name = name
        self.model = model
        self.tools = tools
        self.config = config
        self.logger = structlog.get_logger(name=f"agent.{name}")
        
    @abstractmethod
    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Validate input before processing"""
        pass
        
    @abstractmethod
    async def process(self, state: MedicalAssistantState) -> Dict[str, Any]:
        """Process the request and return state updates"""
        pass
    
    async def process_with_streaming(self, state: MedicalAssistantState) -> Dict[str, Any]:
        """Process with streaming support - the main workflow you want"""
        
        try:
            # Step 1: Get the user query from state
            messages = state.get("messages", [])
            if not messages:
                return {"error": "No messages in state"}
            
            user_message = messages[-1]
            query = user_message.get("content", "")
            
            if not query:
                return {"error": "No query found in latest message"}
            
            # Step 2: Get context using tools (especially knowledge graph)
            context = await self._get_context(query, state)
            
            # Step 3: Create enhanced prompt with context
            enhanced_prompt = self._create_enhanced_prompt(query, context, state)
            
            # Step 4: Generate streaming response from LLM
            full_response = ""
            async for chunk in self._stream_llm_response(enhanced_prompt, state):
                full_response += chunk
                # Note: In the orchestrator, we'll handle the actual streaming to websocket
            
            # Step 5: Process the response and update state
            response_message = {
                "role": "assistant",
                "content": full_response,
                "metadata": {
                    "agent": self.name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "context_used": bool(context),
                    "tools_used": list(self.tools.keys())
                }
            }
            
            # Step 6: Return state updates
            updates = {
                "messages": state.get("messages", []) + [response_message],
                "current_agent": self._determine_next_agent(state, full_response),
                "confidence_scores": {
                    **state.get("confidence_scores", {}),
                    self.name: self._calculate_confidence(full_response)
                }
            }
            
            # Add agent-specific updates
            agent_updates = await self._get_agent_specific_updates(full_response, context, state)
            updates.update(agent_updates)
            
            return updates
            
        except Exception as e:
            self.logger.error(f"Error in {self.name} processing", error=str(e))
            return {
                "error_log": state.get("error_log", []) + [{
                    "agent": self.name,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }],
                "current_agent": "supervisor"
            }
    
    async def _get_context(self, query: str, state: MedicalAssistantState) -> Dict[str, Any]:
        """Get context using available tools"""
        context = {
            "knowledge_graph": {},
            "patient_context": state.get("patient_context", {}),
            "previous_findings": {}
        }
        
        try:
            # Use knowledge graph search if available
            kg_tool = self.tools.get("knowledge_graph_search")
            if kg_tool:
                kg_results = await kg_tool.execute(
                    query=query,
                    entity_types=self._get_relevant_entity_types(),
                    limit=5,
                    include_relationships=True
                )
                context["knowledge_graph"] = kg_results
                
            # Use other tools specific to this agent
            for tool_name, tool in self.tools.items():
                if tool_name != "knowledge_graph_search":
                    try:
                        tool_result = await self._use_tool(tool_name, tool, query, state)
                        if tool_result:
                            context[tool_name] = tool_result
                    except Exception as e:
                        self.logger.warning(f"Tool {tool_name} failed", error=str(e))
                        
        except Exception as e:
            self.logger.error("Error getting context", error=str(e))
            
        return context
    
    async def _use_tool(self, tool_name: str, tool: Any, query: str, state: MedicalAssistantState) -> Optional[Dict[str, Any]]:
        """Use a specific tool - override in subclasses for tool-specific logic"""
        return None
    
    def _get_relevant_entity_types(self) -> List[str]:
        """Get entity types relevant to this agent - override in subclasses"""
        return ["Disease", "Drug", "Symptom", "Gene", "Protein", "Metabolite", "Pathway"]
    
    def _create_enhanced_prompt(self, query: str, context: Dict[str, Any], state: MedicalAssistantState) -> str:
        """Create enhanced prompt with context - override in subclasses for agent-specific prompts"""
        
        # Base system prompt for this agent
        system_prompt = self._get_system_prompt()
        
        # Add context information
        context_str = ""
        kg_context = context.get("knowledge_graph", {})
        
        if kg_context.get("context"):
            context_str += f"\n**Medical Knowledge Base Context:**\n{kg_context['context']}\n"
        
        if kg_context.get("entities"):
            context_str += f"\n**Relevant Medical Entities:** {len(kg_context['entities'])} found\n"
            
        if kg_context.get("relationships"):
            context_str += f"**Medical Relationships:** {len(kg_context['relationships'])} connections\n"
        
        # Add patient context
        patient_context = context.get("patient_context", {})
        if patient_context:
            context_str += f"\n**Patient Context:**\n"
            for key, value in patient_context.items():
                context_str += f"- {key}: {value}\n"
        
        # Add previous findings from state
        if state.get("symptoms"):
            context_str += f"\n**Known Symptoms:** {', '.join(state['symptoms'])}\n"
            
        if state.get("conditions"):
            context_str += f"**Known Conditions:** {', '.join(state['conditions'])}\n"
            
        if state.get("medications"):
            context_str += f"**Current Medications:** {', '.join(state['medications'])}\n"
        
        # Create final prompt
        enhanced_prompt = f"""{system_prompt}

{context_str}

**User Query:** {query}

**Instructions:**
1. Analyze the query in context of the provided medical information
2. Provide accurate, evidence-based information relevant to your specialization
3. Use the medical knowledge base context when applicable
4. Acknowledge any limitations or uncertainties
5. Recommend professional medical consultation when appropriate

Please provide your specialized response:"""

        return enhanced_prompt
    
    async def _stream_llm_response(self, prompt: str, state: MedicalAssistantState) -> AsyncGenerator[str, None]:
        """Stream response from the LLM"""
        try:
            chat_history = self._build_chat_history(state)
            
            async for chunk in self.model.generate_with_context(
                prompt, 
                chat_history, 
                use_rag=False  # We already have context
            ):
                yield chunk
                
        except Exception as e:
            self.logger.error("Error streaming from LLM", error=str(e))
            yield f"Error generating response: {str(e)}"
    
    def _build_chat_history(self, state: MedicalAssistantState) -> List[Dict[str, Any]]:
        """Build chat history from state"""
        messages = state.get("messages", [])
        # Convert to format expected by LLM
        chat_history = []
        for msg in messages[:-1]:  # Exclude the current message
            chat_history.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        return chat_history
    
    def _determine_next_agent(self, state: MedicalAssistantState, response: str) -> str:
        """Determine next agent based on response - override in subclasses"""
        return "end"
    
    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score for the response - override in subclasses"""
        # Simple heuristic - can be improved
        uncertainty_indicators = ["uncertain", "unclear", "might", "possibly", "could be"]
        uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in response.lower())
        
        # Base confidence starts at 0.8, reduced by uncertainty indicators
        confidence = max(0.3, 0.8 - (uncertainty_count * 0.1))
        return confidence
    
    async def _get_agent_specific_updates(self, response: str, context: Dict[str, Any], state: MedicalAssistantState) -> Dict[str, Any]:
        """Get agent-specific state updates - override in subclasses"""
        return {}
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent - must be implemented by subclasses"""
        pass




# class DiagnosticAgent(BaseMedicalAgent):
#     """Diagnostic agent specializing in medical diagnosis"""
    
#     def __init__(self, model: Any, tools: Dict[str, Any], config: Dict[str, Any]):
#         super().__init__("diagnostic", model, tools, config)
    
#     async def validate_input(self, state: MedicalAssistantState) -> bool:
#         """Validate diagnostic input"""
#         # Check if there are symptoms or relevant medical information
#         symptoms = state.get("symptoms", [])
#         messages = state.get("messages", [])
#         return len(symptoms) > 0 or len(messages) > 0
    
#     async def process(self, state: MedicalAssistantState) -> Dict[str, Any]:
#         """Process diagnostic request"""
#         return await self.process_with_streaming(state)
    
#     def _get_system_prompt(self) -> str:
#         return """You are a Medical Diagnostic AI Assistant. Your expertise includes:

# 1. **Symptom Analysis** - Systematic evaluation of presenting symptoms
# 2. **Differential Diagnosis** - Consider multiple possible conditions
# 3. **Risk Assessment** - Evaluate likelihood and severity of conditions
# 4. **Clinical Reasoning** - Apply evidence-based diagnostic approaches
# 5. **Lab Interpretation** - Analyze laboratory and diagnostic test results

# **Diagnostic Approach:**
# - Gather comprehensive symptom information
# - Apply systematic diagnostic frameworks
# - Consider epidemiological factors
# - Evaluate red flags and urgent conditions
# - Recommend appropriate diagnostic tests
# - Provide probability assessments for differential diagnoses

# **Safety Protocols:**
# - Never provide definitive diagnoses without professional evaluation
# - Always recommend appropriate medical consultation
# - Flag serious or emergency conditions immediately
# - Acknowledge diagnostic uncertainty
# - Emphasize the importance of clinical correlation"""
    
#     def _get_relevant_entity_types(self) -> List[str]:
#         return ["Disease", "Symptom", "Condition", "Biomarker", "Test", "Procedure"]
    
#     async def _use_tool(self, tool_name: str, tool: Any, query: str, state: MedicalAssistantState) -> Optional[Dict[str, Any]]:
#         """Use diagnostic-specific tools"""
#         try:
#             if tool_name == "symptom_extractor":
#                 return await tool.execute(text=query)
#             elif tool_name == "lab_interpreter":
#                 # Extract any lab values from query or state
#                 return await tool.execute(query=query)
#             elif tool_name == "pubmed_search":
#                 # Search for relevant diagnostic literature
#                 return await tool.execute(query=f"diagnosis {query}")
#         except Exception as e:
#             self.logger.warning(f"Tool {tool_name} failed in diagnostic agent", error=str(e))
#         return None
    
#     def _determine_next_agent(self, state: MedicalAssistantState, response: str) -> str:
#         """After diagnosis, typically go to treatment"""
#         if "treatment" in response.lower() or "therapy" in response.lower():
#             return "treatment"
#         return "validation"
    
#     async def _get_agent_specific_updates(self, response: str, context: Dict[str, Any], state: MedicalAssistantState) -> Dict[str, Any]:
#         """Update diagnostic findings"""
#         updates = {}
        
#         # Create diagnostic entry
#         diagnostic_entry = {
#             "timestamp": datetime.utcnow().isoformat(),
#             "agent": self.name,
#             "diagnosis": {
#                 "primary_diagnosis": self._extract_primary_diagnosis(response),
#                 "differential_diagnoses": self._extract_differential_diagnoses(response),
#                 "confidence": self._calculate_confidence(response)
#             },
#             "reasoning": response,
#             "context_used": context
#         }
        
#         updates["diagnosis_history"] = state.get("diagnosis_history", []) + [diagnostic_entry]
        
#         return updates
    
#     def _extract_primary_diagnosis(self, response: str) -> Dict[str, Any]:
#         """Extract primary diagnosis from response"""
#         # Simple extraction - can be improved with NLP
#         lines = response.split('\n')
#         for line in lines:
#             if 'primary' in line.lower() or 'most likely' in line.lower():
#                 return {
#                     "condition": line.strip(),
#                     "confidence": self._calculate_confidence(line)
#                 }
#         return {"condition": "Unknown", "confidence": 0.5}
    
#     def _extract_differential_diagnoses(self, response: str) -> List[Dict[str, Any]]:
#         """Extract differential diagnoses from response"""
#         # Simple extraction - can be improved
#         differentials = []
#         lines = response.split('\n')
        
#         in_differential_section = False
#         for line in lines:
#             if 'differential' in line.lower() or 'other possibilities' in line.lower():
#                 in_differential_section = True
#                 continue
            
#             if in_differential_section and line.strip().startswith('-'):
#                 condition = line.strip().lstrip('- ')
#                 differentials.append({
#                     "condition": condition,
#                     "confidence": self._calculate_confidence(line)
#                 })
        
#         return differentials


# class TreatmentAgent(BaseMedicalAgent):
#     """Treatment agent specializing in medical treatment recommendations"""
    
#     def __init__(self, model: Any, tools: Dict[str, Any], config: Dict[str, Any]):
#         super().__init__("treatment", model, tools, config)
    
#     async def validate_input(self, state: MedicalAssistantState) -> bool:
#         """Validate treatment input"""
#         # Check if there's a diagnosis or condition to treat
#         diagnosis_history = state.get("diagnosis_history", [])
#         conditions = state.get("conditions", [])
#         return len(diagnosis_history) > 0 or len(conditions) > 0
    
#     async def process(self, state: MedicalAssistantState) -> Dict[str, Any]:
#         """Process treatment request"""
#         return await self.process_with_streaming(state)
    
#     def _get_system_prompt(self) -> str:
#         return """You are a Medical Treatment AI Assistant specializing in evidence-based treatment recommendations. Your expertise includes:

# 1. **Treatment Planning** - Develop comprehensive treatment strategies
# 2. **Medication Management** - Recommend appropriate pharmacological interventions
# 3. **Non-pharmacological Therapies** - Suggest lifestyle, behavioral, and alternative treatments
# 4. **Clinical Guidelines** - Apply current medical guidelines and protocols
# 5. **Risk-Benefit Analysis** - Evaluate treatment options considering patient factors

# **Treatment Approach:**
# - Apply evidence-based treatment guidelines
# - Consider patient-specific factors (age, comorbidities, allergies)
# - Recommend both pharmacological and non-pharmacological interventions
# - Provide clear rationale for treatment choices
# - Consider contraindications and potential interactions
# - Suggest monitoring parameters and follow-up

# **Safety Considerations:**
# - Always emphasize professional medical supervision
# - Highlight potential side effects and contraindications
# - Recommend appropriate monitoring and follow-up
# - Consider drug interactions and allergies
# - Provide emergency contact guidance when appropriate"""
    
#     def _get_relevant_entity_types(self) -> List[str]:
#         return ["Drug", "Treatment", "Procedure", "Therapy", "Medication", "Protocol"]
    
#     async def _use_tool(self, tool_name: str, tool: Any, query: str, state: MedicalAssistantState) -> Optional[Dict[str, Any]]:
#         """Use treatment-specific tools"""
#         try:
#             if tool_name == "guideline_checker":
#                 # Get the condition from diagnosis
#                 conditions = state.get("conditions", [])
#                 if conditions:
#                     return await tool.execute(condition=conditions[0])
#             elif tool_name == "drug_database":
#                 return await tool.execute(query=query)
#             elif tool_name == "pubmed_search":
#                 return await tool.execute(query=f"treatment {query}")
#         except Exception as e:
#             self.logger.warning(f"Tool {tool_name} failed in treatment agent", error=str(e))
#         return None
    
#     def _determine_next_agent(self, state: MedicalAssistantState, response: str) -> str:
#         """After treatment, check for drug interactions"""
#         return "drug_interaction"
    
#     async def _get_agent_specific_updates(self, response: str, context: Dict[str, Any], state: MedicalAssistantState) -> Dict[str, Any]:
#         """Update treatment plans"""
#         updates = {}
        
#         treatment_plan = {
#             "timestamp": datetime.utcnow().isoformat(),
#             "agent": self.name,
#             "plan": {
#                 "medications": self._extract_medications(response),
#                 "non_pharmacological": self._extract_non_pharmacological(response),
#                 "monitoring": self._extract_monitoring(response),
#                 "follow_up": self._extract_follow_up(response)
#             },
#             "rationale": response,
#             "context_used": context
#         }
        
#         updates["treatment_plans"] = state.get("treatment_plans", []) + [treatment_plan]
        
#         return updates
    
#     def _extract_medications(self, response: str) -> List[str]:
#         """Extract medication recommendations from response"""
#         medications = []
#         lines = response.split('\n')
        
#         for line in lines:
#             if any(word in line.lower() for word in ['medication', 'drug', 'prescribe', 'mg', 'tablet']):
#                 medications.append(line.strip())
        
#         return medications
    
#     def _extract_non_pharmacological(self, response: str) -> List[str]:
#         """Extract non-pharmacological recommendations"""
#         non_pharm = []
#         lines = response.split('\n')
        
#         keywords = ['lifestyle', 'exercise', 'diet', 'therapy', 'counseling', 'physiotherapy']
#         for line in lines:
#             if any(keyword in line.lower() for keyword in keywords):
#                 non_pharm.append(line.strip())
        
#         return non_pharm
    
#     def _extract_monitoring(self, response: str) -> List[str]:
#         """Extract monitoring recommendations"""
#         monitoring = []
#         lines = response.split('\n')
        
#         keywords = ['monitor', 'follow-up', 'check', 'test', 'lab', 'blood work']
#         for line in lines:
#             if any(keyword in line.lower() for keyword in keywords):
#                 monitoring.append(line.strip())
        
#         return monitoring
    
#     def _extract_follow_up(self, response: str) -> List[str]:
#         """Extract follow-up recommendations"""
#         follow_up = []
#         lines = response.split('\n')
        
#         keywords = ['follow-up', 'appointment', 'visit', 'weeks', 'months', 'return']
#         for line in lines:
#             if any(keyword in line.lower() for keyword in keywords):
#                 follow_up.append(line.strip())
        
#         return follow_up