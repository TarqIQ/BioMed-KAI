from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from ..agents.base_agent import BaseMedicalAgent
from ..core.message_protocol import AgentMessage, MessageType, Priority, AgentRole
from ..core.state_manager import MedicalAssistantState


class TreatmentAgent(BaseMedicalAgent):
    """
    Specialist agent for treatment planning and therapy recommendations
    """
    
    def __init__(self, model: Any, tools: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(
            name="treatment",
            role=AgentRole.TREATMENT.value,
            model=model,
            tools=tools,
            config=config
        )
        
        self.treatment_prompt = """
You are an expert medical treatment planner. Based on the diagnosis and patient information, create a comprehensive treatment plan.

Diagnosis Information:
{diagnosis}

Patient Profile:
- Age: {age}
- Gender: {gender}
- Medical History: {medical_history}
- Current Medications: {current_medications}
- Allergies: {allergies}
- Contraindications: {contraindications}

Clinical Guidelines:
{clinical_guidelines}

Create a detailed treatment plan including:

1. PRIMARY TREATMENT APPROACH
   - First-line therapy
   - Treatment goals
   - Expected timeline
   - Success metrics

2. MEDICATIONS
   - Drug name and dosage
   - Administration route
   - Frequency and duration
   - Monitoring requirements
   - Potential side effects

3. NON-PHARMACOLOGICAL INTERVENTIONS
   - Lifestyle modifications
   - Physical therapy
   - Dietary recommendations
   - Exercise prescriptions

4. MONITORING PLAN
   - Follow-up schedule
   - Parameters to monitor
   - Warning signs
   - When to escalate care

5. PATIENT EDUCATION
   - Key points to communicate
   - Self-care instructions
   - When to seek help
   - Resources and support

6. ALTERNATIVE OPTIONS
   - Second-line treatments
   - If primary approach fails
   - Complementary therapies

Format as structured JSON for processing.
"""
        
    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Treatment agent needs a diagnosis to work with"""
        return bool(state.get("diagnosis_history"))
        
    async def process(self, 
                     state: MedicalAssistantState,
                     message: Optional[AgentMessage] = None) -> Dict[str, Any]:
        """Generate treatment plan based on diagnosis"""
        
        # Get latest diagnosis
        diagnosis_history = state.get("diagnosis_history", [])
        if not diagnosis_history:
            return {
                "messages": state.get("messages", []) + [{
                    "role": "assistant",
                    "content": "No diagnosis available for treatment planning. Please provide diagnostic information first.",
                    "metadata": {"agent": self.name, "error": "no_diagnosis"}
                }],
                "current_agent": "supervisor"
            }
            
        latest_diagnosis = diagnosis_history[-1]
        
        # Fetch clinical guidelines
        guidelines = await self._fetch_treatment_guidelines(latest_diagnosis)
        
        # Check for contraindications
        contraindications = await self._check_contraindications(
            condition=latest_diagnosis["diagnosis"],
            patient_context=state.get("patient_context", {})
        )
        
        # Generate treatment plan
        treatment_plan = await self._generate_treatment_plan(
            diagnosis=latest_diagnosis,
            patient_context=state.get("patient_context", {}),
            current_medications=state.get("medications", []),
            allergies=state.get("allergies", []),
            contraindications=contraindications,
            guidelines=guidelines
        )
        
        # Extract medications for interaction checking
        medications = self._extract_medications(treatment_plan)
        
        # Calculate confidence
        confidence = self._calculate_treatment_confidence(treatment_plan, contraindications)
        
        # Prepare updates
        updates = {
            "treatment_plans": state.get("treatment_plans", []) + [{
                "timestamp": datetime.utcnow().isoformat(),
                "plan": treatment_plan,
                "medications": medications,
                "confidence": confidence,
                "agent": self.name
            }],
            "medications": list(set(state.get("medications", []) + medications)),
            "confidence_scores": {
                **state.get("confidence_scores", {}),
                "treatment": confidence
            },
            "current_agent": "drug_interaction" if medications else "validation"
        }
        
        # Add treatment plan to messages
        updates["messages"] = state.get("messages", []) + [{
            "role": "assistant",
            "content": self._format_treatment_message(treatment_plan),
            "metadata": {
                "agent": self.name,
                "confidence": confidence,
                "treatment_data": treatment_plan
            }
        }]
        
        # Log treatment decision
        await self.log_decision(
            decision="treatment_plan_generated",
            reasoning=f"Created treatment plan for {latest_diagnosis['diagnosis'].get('primary_diagnosis', {}).get('condition', 'Unknown')}",
            confidence=confidence,
            state=state
        )
        
        return updates
        
    async def _fetch_treatment_guidelines(self, diagnosis: Dict[str, Any]) -> str:
        """Fetch relevant clinical guidelines"""
        guidelines_parts = []
        
        primary_condition = diagnosis.get("diagnosis", {}).get("primary_diagnosis", {}).get("condition", "")
        
        # Search clinical guidelines database
        if "guideline_checker" in self.tools:
            result = await self.call_tool("guideline_checker", {
                "condition": primary_condition,
                "query_type": "treatment"
            })
            if result.get("guidelines"):
                guidelines_parts.append(result["guidelines"])
                
        # Search medical literature for treatment protocols
        if "pubmed_search" in self.tools:
            query = f"{primary_condition} treatment guidelines protocol"
            result = await self.call_tool("pubmed_search", {
                "query": query,
                "max_results": 3,
                "filters": {"article_type": ["Review", "Practice Guideline"]}
            })
            if result.get("summary"):
                guidelines_parts.append(f"Literature Review:\n{result['summary']}")
                
        return "\n\n".join(guidelines_parts) or "Standard treatment protocols apply."
        
    async def _check_contraindications(self, 
                                      condition: Dict[str, Any],
                                      patient_context: Dict[str, Any]) -> List[str]:
        """Check for treatment contraindications"""
        contraindications = []
        
        # Age-based contraindications
        age = patient_context.get("age")
        if age:
            if age < 18:
                contraindications.append("Pediatric patient - adjust dosing")
            elif age > 65:
                contraindications.append("Geriatric patient - consider reduced dosing")
                
        # Pregnancy/lactation
        if patient_context.get("pregnant"):
            contraindications.append("Pregnancy - avoid teratogenic medications")
        if patient_context.get("lactating"):
            contraindications.append("Lactation - check medication safety")
            
        # Organ function
        if patient_context.get("renal_impairment"):
            contraindications.append("Renal impairment - adjust dosing")
        if patient_context.get("hepatic_impairment"):
            contraindications.append("Hepatic impairment - use with caution")
            
        return contraindications
        
    async def _generate_treatment_plan(self, **kwargs) -> Dict[str, Any]:
        """Generate comprehensive treatment plan using LLM"""
        
        diagnosis_info = kwargs["diagnosis"]["diagnosis"]
        patient_context = kwargs["patient_context"]
        
        prompt = self.treatment_prompt.format(
            diagnosis=json.dumps(diagnosis_info, indent=2),
            age=patient_context.get("age", "Unknown"),
            gender=patient_context.get("gender", "Unknown"),
            medical_history=json.dumps(patient_context.get("medical_history", {})),
            current_medications=", ".join(kwargs.get("current_medications", [])) or "None",
            allergies=", ".join(kwargs.get("allergies", [])) or "None",
            contraindications=", ".join(kwargs.get("contraindications", [])) or "None",
            clinical_guidelines=kwargs.get("guidelines", "")
        )
        
        response = ""
        async for chunk in self.generate_llm_response(prompt):
            response += chunk
            
        try:
            treatment_plan = json.loads(response)
            
            # Ensure required fields
            required_fields = ["primary_treatment", "medications", 
                             "non_pharmacological", "monitoring_plan", 
                             "patient_education", "alternatives"]
            for field in required_fields:
                if field not in treatment_plan:
                    treatment_plan[field] = {}
                    
            return treatment_plan
            
        except json.JSONDecodeError:
            # Fallback structure
            return {
                "primary_treatment": {
                    "approach": "Standard care",
                    "goals": ["Symptom management"],
                    "timeline": "As needed"
                },
                "medications": [],
                "non_pharmacological": {},
                "monitoring_plan": {},
                "patient_education": {},
                "alternatives": {},
                "raw_response": response
            }
            
    def _extract_medications(self, treatment_plan: Dict[str, Any]) -> List[str]:
        """Extract medication names from treatment plan"""
        medications = []
        
        med_list = treatment_plan.get("medications", [])
        for med in med_list:
            if isinstance(med, dict):
                med_name = med.get("name", "")
                if med_name:
                    medications.append(med_name)
            elif isinstance(med, str):
                medications.append(med)
                
        return medications
        
    def _calculate_treatment_confidence(self, 
                                      treatment_plan: Dict[str, Any],
                                      contraindications: List[str]) -> float:
        """Calculate confidence in treatment plan"""
        factors = {}
        
        # Base confidence
        factors["base"] = 0.8
        
        # Reduce for contraindications
        factors["contraindications"] = 1.0 - (len(contraindications) * 0.1)
        
        # Evidence-based treatment
        if treatment_plan.get("clinical_guidelines"):
            factors["evidence"] = 0.9
        else:
            factors["evidence"] = 0.6
            
        # Completeness of plan
        completeness = sum(1 for field in ["primary_treatment", "medications", 
                                          "monitoring_plan", "patient_education"]
                          if treatment_plan.get(field))
        factors["completeness"] = completeness / 4
        
        return self.calculate_confidence(factors)
        
    def _format_treatment_message(self, treatment_plan: Dict[str, Any]) -> str:
        """Format treatment plan for display"""
        message = "## Treatment Plan\n\n"
        
        # Primary treatment
        primary = treatment_plan.get("primary_treatment", {})
        message += "### Primary Treatment Approach\n"
        message += f"**Approach:** {primary.get('approach', 'Not specified')}\n"
        message += f"**Goals:**\n"
        for goal in primary.get('goals', []):
            message += f"- {goal}\n"
        message += f"**Timeline:** {primary.get('timeline', 'As directed')}\n\n"
        
        # Medications
        if treatment_plan.get("medications"):
            message += "### Medications\n"
            for med in treatment_plan["medications"]:
                if isinstance(med, dict):
                    message += f"**{med.get('name', 'Unknown')}**\n"
                    message += f"- Dose: {med.get('dosage', 'As directed')}\n"
                    message += f"- Route: {med.get('route', 'Oral')}\n"
                    message += f"- Frequency: {med.get('frequency', 'As directed')}\n"
                    message += f"- Duration: {med.get('duration', 'As directed')}\n\n"
                    
        # Non-pharmacological
        non_pharm = treatment_plan.get("non_pharmacological", {})
        if non_pharm:
            message += "### Non-Pharmacological Interventions\n"
            for category, items in non_pharm.items():
                message += f"**{category.replace('_', ' ').title()}:**\n"
                if isinstance(items, list):
                    for item in items:
                        message += f"- {item}\n"
                else:
                    message += f"- {items}\n"
            message += "\n"
            
        # Monitoring
        monitoring = treatment_plan.get("monitoring_plan", {})
        if monitoring:
            message += "### Monitoring Plan\n"
            message += f"**Follow-up:** {monitoring.get('follow_up', 'As needed')}\n"
            if monitoring.get('parameters'):
                message += "**Parameters to Monitor:**\n"
                for param in monitoring['parameters']:
                    message += f"- {param}\n"
                    
        return message

    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Validate treatment input"""
        # Check if there's a diagnosis or condition to treat
        diagnosis_history = state.get("diagnosis_history", [])
        conditions = state.get("conditions", [])
        return len(diagnosis_history) > 0 or len(conditions) > 0
    
    async def process(self, state: MedicalAssistantState) -> Dict[str, Any]:
        """Process treatment request"""
        return await self.process_with_streaming(state)
    
    def _get_system_prompt(self) -> str:
        return """You are a therapeutic guidance specialist providing evidence-based treatment information. Your role is to outline appropriate therapeutic approaches for diagnosed medical conditions, following current clinical guidelines and best practices.

**Your Framework:**
- Base recommendations on current clinical practice guidelines
- Consider patient-specific factors (age, comorbidities, medications)
- Present treatment options in order of typical clinical preference
- Include both pharmacological and non-pharmacological interventions
- Consider contraindications and precautions

**Response Structure for Each Treatment Option:**
1. **Treatment Name/Approach**
2. **Mechanism/Rationale** (how/why it works)
3. **Typical Implementation** (dosing, duration, method)
4. **Expected Outcomes** (what to expect)
5. **Key Considerations** (contraindications, monitoring, side effects)

**Treatment Categories to Consider:**
- First-line therapies (standard of care)
- Adjunctive treatments (supportive care)
- Lifestyle modifications (diet, exercise, behavioral changes)
- Monitoring and follow-up requirements
- Emergency/urgent interventions if applicable

**Patient Context:** {input_query}

**Critical Disclaimer:**
All treatment information is for educational purposes. Actual treatment decisions must always be made by qualified healthcare professionals who can evaluate the complete clinical picture, perform necessary examinations, and consider individual patient factors.

Provide the 5 most appropriate evidence-based therapeutic approaches."""
    
    def _get_relevant_entity_types(self) -> List[str]:
        return ["Drug", "Treatment", "Procedure", "Therapy", "Medication", "Protocol"]
    
    async def _use_tool(self, tool_name: str, tool: Any, query: str, state: MedicalAssistantState) -> Optional[Dict[str, Any]]:
        """Use treatment-specific tools"""
        try:
            if tool_name == "guideline_checker":
                # Get the condition from diagnosis
                conditions = state.get("conditions", [])
                if conditions:
                    return await tool.execute(condition=conditions[0])
            elif tool_name == "drug_database":
                return await tool.execute(query=query)
            elif tool_name == "pubmed_search":
                return await tool.execute(query=f"treatment {query}")
        except Exception as e:
            self.logger.warning(f"Tool {tool_name} failed in treatment agent", error=str(e))
        return None
    
    def _determine_next_agent(self, state: MedicalAssistantState, response: str) -> str:
        """After treatment, check for drug interactions"""
        return "drug_interaction"
    
    async def _get_agent_specific_updates(self, response: str, context: Dict[str, Any], state: MedicalAssistantState) -> Dict[str, Any]:
        """Update treatment plans"""
        updates = {}
        
        treatment_plan = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.name,
            "plan": {
                "medications": self._extract_medications(response),
                "non_pharmacological": self._extract_non_pharmacological(response),
                "monitoring": self._extract_monitoring(response),
                "follow_up": self._extract_follow_up(response)
            },
            "rationale": response,
            "context_used": context
        }
        
        updates["treatment_plans"] = state.get("treatment_plans", []) + [treatment_plan]
        
        return updates
    
    def _extract_medications(self, response: str) -> List[str]:
        """Extract medication recommendations from response"""
        medications = []
        lines = response.split('\n')
        
        for line in lines:
            if any(word in line.lower() for word in ['medication', 'drug', 'prescribe', 'mg', 'tablet']):
                medications.append(line.strip())
        
        return medications
    
    def _extract_non_pharmacological(self, response: str) -> List[str]:
        """Extract non-pharmacological recommendations"""
        non_pharm = []
        lines = response.split('\n')
        
        keywords = ['lifestyle', 'exercise', 'diet', 'therapy', 'counseling', 'physiotherapy']
        for line in lines:
            if any(keyword in line.lower() for keyword in keywords):
                non_pharm.append(line.strip())
        
        return non_pharm
    
    def _extract_monitoring(self, response: str) -> List[str]:
        """Extract monitoring recommendations"""
        monitoring = []
        lines = response.split('\n')
        
        keywords = ['monitor', 'follow-up', 'check', 'test', 'lab', 'blood work']
        for line in lines:
            if any(keyword in line.lower() for keyword in keywords):
                monitoring.append(line.strip())
        
        return monitoring
    
    def _extract_follow_up(self, response: str) -> List[str]:
        """Extract follow-up recommendations"""
        follow_up = []
        lines = response.split('\n')
        
        keywords = ['follow-up', 'appointment', 'visit', 'weeks', 'months', 'return']
        for line in lines:
            if any(keyword in line.lower() for keyword in keywords):
                follow_up.append(line.strip())
        
        return follow_up