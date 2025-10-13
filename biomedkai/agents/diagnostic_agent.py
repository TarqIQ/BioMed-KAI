from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from src.agents.base_agent import BaseMedicalAgent
from src.core.message_protocol import AgentMessage, MessageType, Priority, AgentRole
from src.core.state_manager import MedicalAssistantState


class DiagnosticAgent(BaseMedicalAgent):
    """
    Specialist agent for medical diagnosis and symptom analysis
    """
    
    def __init__(self, model: Any, tools: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(
            name="diagnostic",
            role=AgentRole.DIAGNOSTIC.value,
            model=model,
            tools=tools,
            config=config
        )
        
        self.diagnostic_prompt = """
You are a medical diagnostic reasoning assistant. Your role is to analyze presented symptoms and provide the most likely differential diagnoses based on clinical presentation, considering epidemiology, pathophysiology, and evidence-based medicine.

**Your Approach:**
- Analyze symptoms systematically using clinical reasoning
- Consider patient demographics (age, gender, occupation, medical history)
- Apply diagnostic probability and clinical decision-making principles
- Think through red flags and urgent conditions first
- Consider common conditions before rare ones (common things being common)

**Response Format:**
For each of the top 5 most likely diagnoses, provide:
1. **Diagnosis Name**
2. **Probability Assessment** (High/Moderate/Low likelihood)
3. **Supporting Evidence** (2-3 key symptoms/factors that support this diagnosis)
4. **Key Distinguishing Features** (what makes this diagnosis likely given the presentation)

**Clinical Reasoning Process:**
- First identify the primary symptom complex
- Consider anatomical location and system involvement
- Factor in timeline (acute vs chronic)
- Account for patient's medical history and medications
- Apply epidemiological factors (age, gender, occupation)

**Critical Reminder:**
This is for educational/informational purposes only. Always emphasize the need for proper medical evaluation, physical examination, and appropriate diagnostic testing by qualified healthcare professionals.

**Patient Presentation:** {input_query}

Provide your top 5 differential diagnoses with clinical reasoning.

"""

        self.symptom_analysis_prompt = """
Analyze the following symptoms for medical significance:

Symptoms: {symptoms}
Duration: {duration}
Severity: {severity}
Associated Factors: {factors}

Determine:
1. Symptom patterns and clusters
2. Possible organ systems involved
3. Acute vs chronic presentation
4. Severity assessment
5. Need for urgent evaluation
"""
        
    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Validate diagnostic agent has necessary input"""
        # Need at least symptoms or a query
        return bool(state.get("symptoms") or 
                   state.get("medical_entities") or
                   (state.get("messages") and len(state["messages"]) > 0))
        
    async def process(self, 
                     state: MedicalAssistantState,
                     message: Optional[AgentMessage] = None) -> Dict[str, Any]:
        """Process diagnostic request and generate diagnosis"""
        
        # Extract symptoms
        symptoms = await self._extract_symptoms(state, message)
        
        # Analyze symptoms
        symptom_analysis = await self._analyze_symptoms(symptoms)
        
        # Fetch medical context
        medical_context = await self._fetch_medical_context(symptoms, state)
        
        # Generate diagnosis
        diagnosis = await self._generate_diagnosis(
            symptoms=symptoms,
            symptom_analysis=symptom_analysis,
            medical_context=medical_context,
            patient_context=state.get("patient_context", {}),
            medications=state.get("medications", []),
            allergies=state.get("allergies", [])
        )
        
        # Calculate confidence
        confidence = self._calculate_diagnostic_confidence(diagnosis)
        
        # Determine if human review needed
        needs_review = self.should_request_human_review(confidence, state)
        
        # Prepare updates
        updates = {
            "symptoms": symptoms,
            "diagnosis_history": state.get("diagnosis_history", []) + [{
                "timestamp": datetime.utcnow().isoformat(),
                "diagnosis": diagnosis,
                "confidence": confidence,
                "symptoms": symptoms,
                "analysis": symptom_analysis,
                "agent": self.name
            }],
            "confidence_scores": {
                **state.get("confidence_scores", {}),
                "diagnostic": confidence
            },
            "requires_human_review": needs_review,
            "current_agent": "treatment" if not needs_review else "validation"
        }
        
        # Add diagnosis to messages
        updates["messages"] = state.get("messages", []) + [{
            "role": "assistant",
            "content": self._format_diagnosis_message(diagnosis),
            "metadata": {
                "agent": self.name,
                "confidence": confidence,
                "diagnosis_data": diagnosis
            }
        }]
        
        # Log diagnostic decision
        await self.log_decision(
            decision="diagnosis_generated",
            reasoning=diagnosis.get("clinical_reasoning", ""),
            confidence=confidence,
            state=state
        )
        
        return updates
        
    async def _extract_symptoms(self, 
                               state: MedicalAssistantState,
                               message: Optional[AgentMessage]) -> List[str]:
        """Extract symptoms from state and message"""
        symptoms = state.get("symptoms", []).copy()
        
        # Add symptoms from message
        if message and message.content.get("symptoms"):
            symptoms.extend(message.content["symptoms"])
            
        # Extract from medical entities
        for entity in state.get("medical_entities", []):
            if entity.get("label", "").upper() in ["SYMPTOM", "SIGN", "COMPLAINT"]:
                symptom_text = entity.get("text", "")
                if symptom_text and symptom_text not in symptoms:
                    symptoms.append(symptom_text)
                    
        # Extract from latest message using NLP
        if state.get("messages"):
            latest_msg = state["messages"][-1].get("content", "")
            if "symptom_extractor" in self.tools:
                extracted = await self.call_tool("symptom_extractor", {"text": latest_msg})
                symptoms.extend(extracted.get("symptoms", []))
                
        return list(set(symptoms))  # Remove duplicates
        
    async def _analyze_symptoms(self, symptoms: List[str]) -> Dict[str, Any]:
        """Analyze symptoms for patterns and severity"""
        if not symptoms:
            return {"severity": "unknown", "patterns": [], "urgent": False}
            
        prompt = self.symptom_analysis_prompt.format(
            symptoms=", ".join(symptoms),
            duration="Not specified",
            severity="Not specified",
            factors="Not specified"
        )
        
        response = ""
        async for chunk in self.generate_llm_response(prompt):
            response += chunk
            
        try:
            return json.loads(response)
        except:
            return {
                "severity": "moderate",
                "patterns": symptoms,
                "urgent": any(s in " ".join(symptoms).lower() 
                            for s in ["severe", "acute", "sudden"])
            }
            
    async def _fetch_medical_context(self, 
                                    symptoms: List[str],
                                    state: MedicalAssistantState) -> str:
        """Fetch relevant medical context from knowledge base"""
        context_parts = []
        
        # Fetch from Neo4j knowledge graph
        if "knowledge_graph_search" in self.tools:
            for symptom in symptoms[:5]:  # Limit to top 5 symptoms
                result = await self.call_tool("knowledge_graph_search", {
                    "query": symptom,
                    "entity_types": ["Disease", "Symptom", "Condition"],
                    "limit": 3
                })
                if result.get("context"):
                    context_parts.append(f"Context for '{symptom}':\n{result['context']}")
                    
        # Fetch from medical literature
        if "pubmed_search" in self.tools and self.config.get("use_literature", True):
            query = " ".join(symptoms[:3]) + " diagnosis differential"
            result = await self.call_tool("pubmed_search", {
                "query": query,
                "max_results": 3
            })
            if result.get("articles"):
                context_parts.append("Relevant Literature:\n" + result["summary"])
                
        return "\n\n".join(context_parts)
        
    async def _generate_diagnosis(self, **kwargs) -> Dict[str, Any]:
        """Generate comprehensive diagnosis using LLM"""
        
        # Prepare prompt
        prompt = self.diagnostic_prompt.format(
            symptoms=", ".join(kwargs.get("symptoms", [])),
            medical_history=json.dumps(kwargs.get("patient_context", {}).get("medical_history", {})),
            medications=", ".join(kwargs.get("medications", [])) or "None",
            allergies=", ".join(kwargs.get("allergies", [])) or "None",
            vitals=json.dumps(kwargs.get("patient_context", {}).get("vitals", {})),
            lab_results=json.dumps(kwargs.get("patient_context", {}).get("lab_results", {})),
            medical_context=kwargs.get("medical_context", "No additional context available")
        )
        
        response = ""
        async for chunk in self.generate_llm_response(prompt):
            response += chunk
            
        try:
            diagnosis = json.loads(response)
            
            # Ensure required fields
            required_fields = ["primary_diagnosis", "differential_diagnoses", 
                             "red_flags", "recommended_tests", "clinical_reasoning"]
            for field in required_fields:
                if field not in diagnosis:
                    diagnosis[field] = {}
                    
            return diagnosis
            
        except json.JSONDecodeError:
            # Fallback structure
            return {
                "primary_diagnosis": {
                    "condition": "Unable to parse diagnosis",
                    "confidence": 0.3,
                    "evidence": ["Diagnosis generation failed"]
                },
                "differential_diagnoses": [],
                "red_flags": [],
                "recommended_tests": [],
                "clinical_reasoning": response
            }
            
    def _calculate_diagnostic_confidence(self, diagnosis: Dict[str, Any]) -> float:
        """Calculate overall diagnostic confidence"""
        factors = {}
        
        # Primary diagnosis confidence
        primary_conf = diagnosis.get("primary_diagnosis", {}).get("confidence", 0.5)
        factors["primary"] = primary_conf
        
        # Evidence quality
        evidence = diagnosis.get("primary_diagnosis", {}).get("evidence", [])
        factors["evidence"] = min(1.0, len(evidence) * 0.2)
        
        # Differential count (more differentials = less certainty)
        diff_count = len(diagnosis.get("differential_diagnoses", []))
        factors["differential"] = 1.0 - min(0.5, diff_count * 0.1)
        
        # Red flags presence
        red_flags = diagnosis.get("red_flags", [])
        factors["safety"] = 0.5 if red_flags else 1.0
        
        return self.calculate_confidence(factors)
        
    def _format_diagnosis_message(self, diagnosis: Dict[str, Any]) -> str:
        """Format diagnosis for display"""
        primary = diagnosis.get("primary_diagnosis", {})
        
        message = f"## Diagnostic Assessment\n\n"
        message += f"**Primary Diagnosis:** {primary.get('condition', 'Unknown')}\n"
        message += f"- Confidence: {primary.get('confidence', 0):.1%}\n"
        
        if primary.get('icd_code'):
            message += f"- ICD-10 Code: {primary['icd_code']}\n"
            
        message += f"\n### Supporting Evidence:\n"
        for evidence in primary.get('evidence', []):
            message += f"- {evidence}\n"
            
        if diagnosis.get('differential_diagnoses'):
            message += f"\n### Differential Diagnoses:\n"
            for diff in diagnosis['differential_diagnoses']:
                message += f"- {diff.get('condition')} ({diff.get('probability', 0):.1%})\n"
                
        if diagnosis.get('red_flags'):
            message += f"\n### ⚠️ Red Flags:\n"
            for flag in diagnosis['red_flags']:
                message += f"- **{flag}**\n"
                
        if diagnosis.get('recommended_tests'):
            message += f"\n### Recommended Tests:\n"
            for test in diagnosis['recommended_tests']:
                priority = test.get('priority', 'routine')
                message += f"- {test.get('name')} [{priority.upper()}]\n"
                
        return message

    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Validate drug interaction input"""
        medications = state.get("medications", [])
        treatment_plans = state.get("treatment_plans", [])
        return len(medications) > 0 or len(treatment_plans) > 0
    
    async def process(self, state: MedicalAssistantState) -> Dict[str, Any]:
        """Process drug interaction request"""
        return await self.process_with_streaming(state)
    
    def _get_system_prompt(self) -> str:
        return self.diagnostic_prompt
        return """You are a Medical Diagnostic AI Assistant. Your expertise includes:

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
    
    def _get_relevant_entity_types(self) -> List[str]:
        return ["Drug", "Medication", "Chemical", "Interaction", "Side_Effect"]
    
    async def _use_tool(self, tool_name: str, tool: Any, query: str, state: MedicalAssistantState) -> Optional[Dict[str, Any]]:
        """Use drug interaction specific tools"""
        try:
            if tool_name == "drug_interaction_checker":
                medications = state.get("medications", [])
                if len(medications) >= 2:
                    return await tool.execute(drugs=medications)
            elif tool_name == "allergy_checker":
                medications = state.get("medications", [])
                patient_context = state.get("patient_context", {})
                allergies = patient_context.get("allergies", [])
                if medications and allergies:
                    return await tool.execute(drugs=medications, allergies=allergies)
            elif tool_name == "drug_database":
                return await tool.execute(query=query)
        except Exception as e:
            self.logger.warning(f"Tool {tool_name} failed in drug interaction agent", error=str(e))
        return None
    
    def _determine_next_agent(self, state: MedicalAssistantState, response: str) -> str:
        """After drug interaction check, go to validation"""
        return "validation"
    
    async def _get_agent_specific_updates(self, response: str, context: Dict[str, Any], state: MedicalAssistantState) -> Dict[str, Any]:
        """Update drug interaction findings"""
        updates = {}
        
        interaction_analysis = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.name,
            "interactions": self._extract_interactions(response),
            "warnings": self._extract_warnings(response),
            "recommendations": self._extract_safety_recommendations(response),
            "analysis": response,
            "context_used": context
        }
        
        updates["drug_interactions"] = state.get("drug_interactions", []) + [interaction_analysis]
        
        return updates
    
    def _extract_interactions(self, response: str) -> List[Dict[str, Any]]:
        """Extract drug interactions from response"""
        interactions = []
        lines = response.split('\n')
        
        current_interaction = None
        for line in lines:
            if 'interaction' in line.lower():
                if current_interaction:
                    interactions.append(current_interaction)
                current_interaction = {
                    "description": line.strip(),
                    "severity": self._assess_severity(line),
                    "mechanism": "",
                    "management": ""
                }
            elif current_interaction and 'severity' in line.lower():
                current_interaction["severity"] = self._extract_severity(line)
            elif current_interaction and 'management' in line.lower():
                current_interaction["management"] = line.strip()
        
        if current_interaction:
            interactions.append(current_interaction)
        
        return interactions
    
    def _extract_warnings(self, response: str) -> List[str]:
        """Extract safety warnings"""
        warnings = []
        lines = response.split('\n')
        
        warning_keywords = ['warning', 'caution', 'contraindicated', 'avoid', 'dangerous']
        for line in lines:
            if any(keyword in line.lower() for keyword in warning_keywords):
                warnings.append(line.strip())
        
        return warnings
    
    def _extract_safety_recommendations(self, response: str) -> List[str]:
        """Extract safety recommendations"""
        recommendations = []
        lines = response.split('\n')
        
        rec_keywords = ['recommend', 'suggest', 'monitor', 'adjust', 'consider']
        for line in lines:
            if any(keyword in line.lower() for keyword in rec_keywords):
                recommendations.append(line.strip())
        
        return recommendations
    
    def _assess_severity(self, text: str) -> str:
        """Assess interaction severity"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['severe', 'major', 'dangerous', 'contraindicated']):
            return "Major"
        elif any(word in text_lower for word in ['moderate', 'significant']):
            return "Moderate"
        elif any(word in text_lower for word in ['minor', 'mild']):
            return "Minor"
        return "Unknown"
    
    def _extract_severity(self, text: str) -> str:
        """Extract severity level from text"""
        severity_map = {
            'major': 'Major',
            'severe': 'Major',
            'moderate': 'Moderate',
            'minor': 'Minor',
            'mild': 'Minor'
        }
        
        text_lower = text.lower()
        for keyword, severity in severity_map.items():
            if keyword in text_lower:
                return severity
        
        return "Unknown"