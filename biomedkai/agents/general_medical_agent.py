from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from src.agents.base_agent import BaseMedicalAgent
from src.core.message_protocol import AgentMessage, MessageType, Priority, AgentRole
from src.core.state_manager import MedicalAssistantState

class GeneralMedicalAgent(BaseMedicalAgent):
    """
    Specialist agent for general medical assessment
    """
    
    def __init__(self, model: Any, tools: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(
            name="general",
            role=AgentRole.GENERAL.value,
            model=model,
            tools=tools,
            config=config
        )
        
        self.triage_prompt = """
        You are a knowledgeable medical education assistant specializing in general health information. Your role is to provide accurate, evidence-based explanations about common health topics, bodily functions, and general wellness concepts.

        **Your Guidelines:**
        - Provide clear, educational explanations using accessible language
        - Include relevant medical terminology with simple definitions
        - Use analogies and examples to make complex concepts understandable
        - Focus on established medical knowledge and consensus
        - Always emphasize that this is educational information, not personalized medical advice

        **Response Structure:**
        1. Direct answer to the question
        2. Brief explanation of underlying mechanisms when relevant
        3. Additional context or related information that might be helpful
        4. When appropriate, mention when to consult healthcare professionals

        **Important Reminders:**
        - Do not provide specific medical advice or diagnose conditions
        - Encourage consultation with healthcare providers for personal health concerns
        - Present information objectively without unnecessary alarm

        **Query:** {input_query}

        Provide a comprehensive yet accessible explanation.        
        """

        self.vital_signs_analysis_prompt = """
    Analyze these vital signs for emergency indicators:

    Vital Signs:
    - Blood Pressure: {bp}
    - Heart Rate: {hr}
    - Respiratory Rate: {rr}
    - Temperature: {temp}
    - Oxygen Saturation: {o2_sat}
    - Pain Scale: {pain}

    Provide analysis in markdown format covering:
    - Critical abnormalities requiring immediate attention
    - Concerning patterns that need monitoring
    - Normal variations vs. pathological findings
    - Recommendations based on vital sign assessment
    """

        # Emergency criteria thresholds
        self.emergency_criteria = {
            "vital_signs": {
                "hr_critical": {"min": 40, "max": 150},
                "bp_critical": {"systolic_min": 70, "systolic_max": 200, "diastolic_max": 110},
                "rr_critical": {"min": 8, "max": 30},
                "temp_critical": {"min": 94, "max": 104},
                "o2_sat_critical": {"min": 85}
            },
            "red_flag_symptoms": [
                "chest pain", "difficulty breathing", "severe headache",
                "altered mental status", "severe abdominal pain",
                "uncontrolled bleeding", "stroke symptoms",
                "severe allergic reaction", "poisoning"
            ]
        }
        
    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Validate general medical has necessary input"""
        # Need at least symptoms or chief complaint
        return bool(state.get("symptoms") or 
                   state.get("chief_complaint") or
                   (state.get("messages") and len(state["messages"]) > 0))
        
    async def process(self, 
                     state: MedicalAssistantState,
                     message: Optional[AgentMessage] = None) -> Dict[str, Any]:
        """Process general medical assessment"""
        
        # Extract clinical data
        clinical_data = await self._extract_clinical_data(state, message)
        
        # Analyze vital signs
        vital_signs_analysis = await self._analyze_vital_signs(clinical_data.get("vital_signs", {}))
        
        # Check for red flags
        red_flags = await self._identify_red_flags(clinical_data)
        
        # Calculate triage score
        triage_assessment = await self._perform_triage_assessment(
            clinical_data=clinical_data,
            vital_analysis=vital_signs_analysis,
            red_flags=red_flags
        )
        
        # Determine emergency level
        emergency_level = self._determine_emergency_level(triage_assessment)
        
        # Calculate confidence
        confidence = self._calculate_triage_confidence(triage_assessment, clinical_data)
        
        # Determine next steps
        next_agent = self._determine_next_agent(emergency_level, triage_assessment)
        
        # Prepare updates
        updates = {
            "emergency_flag": emergency_level in ["critical", "emergency"],
            "triage_level": triage_assessment.get("triage_level", 3),
            "emergency_assessment": {
                "timestamp": datetime.utcnow().isoformat(),
                "level": emergency_level,
                "triage_data": triage_assessment,
                "red_flags": red_flags,
                "vital_analysis": vital_signs_analysis,
                "agent": self.name
            },
            "confidence_scores": {
                **state.get("confidence_scores", {}),
                "general": confidence
            },
            "requires_human_review": emergency_level in ["critical", "emergency"],
            "current_agent": next_agent
        }
        
        # Update symptoms with extracted data
        if clinical_data.get("symptoms"):
            updates["symptoms"] = list(set(
                state.get("symptoms", []) + clinical_data["symptoms"]
            ))
            
        # Add chief complaint
        if clinical_data.get("chief_complaint"):
            updates["chief_complaint"] = clinical_data["chief_complaint"]
            
        # Add triage message
        updates["messages"] = state.get("messages", []) + [{
            "role": "assistant",
            "content": self._format_triage_message(triage_assessment, emergency_level),
            "metadata": {
                "agent": self.name,
                "emergency_level": emergency_level,
                "triage_level": triage_assessment.get("triage_level"),
                "confidence": confidence
            }
        }]
        
        # Log triage decision
        await self.log_decision(
            decision="triage_completed",
            reasoning=triage_assessment.get("reasoning", ""),
            confidence=confidence,
            state=state
        )
        
        return updates
        
    async def _extract_clinical_data(self, 
                                   state: MedicalAssistantState,
                                   message: Optional[AgentMessage]) -> Dict[str, Any]:
        """Extract clinical data for triage assessment"""
        data = {}
        
        # Extract chief complaint
        if message and message.content.get("chief_complaint"):
            data["chief_complaint"] = message.content["chief_complaint"]
        elif state.get("messages"):
            # Extract from first user message
            first_msg = next((msg for msg in state["messages"] if msg.get("role") == "user"), None)
            if first_msg:
                data["chief_complaint"] = first_msg.get("content", "")[:200]  # Limit length
                
        # Extract symptoms
        data["symptoms"] = state.get("symptoms", []).copy()
        if message and message.content.get("symptoms"):
            data["symptoms"].extend(message.content["symptoms"])
            
        # Extract vital signs
        data["vital_signs"] = state.get("patient_context", {}).get("vitals", {})
        if message and message.content.get("vital_signs"):
            data["vital_signs"].update(message.content["vital_signs"])
            
        # Extract other clinical data
        data["pain_level"] = state.get("patient_context", {}).get("pain_level")
        data["duration"] = state.get("patient_context", {}).get("symptom_duration")
        data["age"] = state.get("patient_context", {}).get("age")
        data["medical_history"] = state.get("patient_context", {}).get("medical_history", {})
        
        return data
        
    async def _analyze_vital_signs(self, vital_signs: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze vital signs for emergency indicators"""
        if not vital_signs:
            return {"status": "unknown", "critical": False}
            
        # Use tool if available
        if "vital_signs_analyzer" in self.tools:
            result = await self.call_tool("vital_signs_analyzer", {"vitals": vital_signs})
            if result:
                return result
                
        # Manual analysis
        analysis = {
            "status": "normal",
            "critical": False,
            "abnormalities": [],
            "concerns": []
        }
        
        criteria = self.emergency_criteria["vital_signs"]
        
        # Check heart rate
        hr = vital_signs.get("heart_rate")
        if hr:
            if hr < criteria["hr_critical"]["min"] or hr > criteria["hr_critical"]["max"]:
                analysis["critical"] = True
                analysis["abnormalities"].append(f"Critical heart rate: {hr}")
                
        # Check blood pressure
        bp_sys = vital_signs.get("systolic_bp")
        bp_dia = vital_signs.get("diastolic_bp")
        if bp_sys:
            if (bp_sys < criteria["bp_critical"]["systolic_min"] or 
                bp_sys > criteria["bp_critical"]["systolic_max"]):
                analysis["critical"] = True
                analysis["abnormalities"].append(f"Critical systolic BP: {bp_sys}")
        if bp_dia and bp_dia > criteria["bp_critical"]["diastolic_max"]:
            analysis["critical"] = True
            analysis["abnormalities"].append(f"Critical diastolic BP: {bp_dia}")
            
        # Check respiratory rate
        rr = vital_signs.get("respiratory_rate")
        if rr:
            if rr < criteria["rr_critical"]["min"] or rr > criteria["rr_critical"]["max"]:
                analysis["critical"] = True
                analysis["abnormalities"].append(f"Critical respiratory rate: {rr}")
                
        # Check temperature
        temp = vital_signs.get("temperature")
        if temp:
            if temp < criteria["temp_critical"]["min"] or temp > criteria["temp_critical"]["max"]:
                analysis["critical"] = True
                analysis["abnormalities"].append(f"Critical temperature: {temp}°F")
                
        # Check oxygen saturation
        o2_sat = vital_signs.get("oxygen_saturation")
        if o2_sat and o2_sat < criteria["o2_sat_critical"]["min"]:
            analysis["critical"] = True
            analysis["abnormalities"].append(f"Critical oxygen saturation: {o2_sat}%")
            
        if analysis["critical"]:
            analysis["status"] = "critical"
        elif analysis["abnormalities"]:
            analysis["status"] = "abnormal"
            
        return analysis
        
    async def _identify_red_flags(self, clinical_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify red flag symptoms and conditions"""
        red_flags = []
        
        # Check symptoms against red flag list
        symptoms = clinical_data.get("symptoms", [])
        chief_complaint = clinical_data.get("chief_complaint", "").lower()
        
        for flag_symptom in self.emergency_criteria["red_flag_symptoms"]:
            if (any(flag_symptom.lower() in str(symptom).lower() for symptom in symptoms) or
                flag_symptom.lower() in chief_complaint):
                red_flags.append({
                    "symptom": flag_symptom,
                    "severity": "high",
                    "action": "immediate_evaluation"
                })
                
        # Use symptom analyzer tool if available
        if "symptom_analyzer" in self.tools and symptoms:
            result = await self.call_tool("symptom_analyzer", {
                "symptoms": symptoms,
                "mode": "emergency_screening"
            })
            if result and result.get("red_flags"):
                red_flags.extend(result["red_flags"])
                
        return red_flags
        
    async def _perform_triage_assessment(self, 
                                       clinical_data: Dict[str, Any],
                                       vital_analysis: Dict[str, Any],
                                       red_flags: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive triage assessment"""
        
        # Prepare assessment criteria
        criteria = self._format_assessment_criteria()
        
        # Prepare prompt
        prompt = self.triage_prompt.format(
            chief_complaint=clinical_data.get("chief_complaint", "Not provided"),
            symptoms=", ".join(clinical_data.get("symptoms", [])) or "None reported",
            vital_signs=json.dumps(clinical_data.get("vital_signs", {})),
            pain_level=clinical_data.get("pain_level", "Not assessed"),
            duration=clinical_data.get("duration", "Unknown"),
            age=clinical_data.get("age", "Not provided"),
            medical_history=json.dumps(clinical_data.get("medical_history", {})),
            assessment_criteria=criteria
        )
        
        response = ""
        async for chunk in self.generate_llm_response(prompt):
            response += chunk
            
        try:
            assessment = json.loads(response)
            
            # Ensure required fields
            if "triage_level" not in assessment:
                assessment["triage_level"] = self._calculate_triage_level(vital_analysis, red_flags)
                
            # Add analysis data
            assessment["vital_analysis"] = vital_analysis
            assessment["red_flags_detected"] = red_flags
            assessment["timestamp"] = datetime.utcnow().isoformat()
            
            return assessment
            
        except json.JSONDecodeError:
            # Fallback assessment
            return {
                "triage_level": self._calculate_triage_level(vital_analysis, red_flags),
                "immediate_actions": ["Clinical evaluation required"],
                "warning_signs": [flag["symptom"] for flag in red_flags],
                "disposition": "emergency_department" if red_flags else "urgent_care",
                "time_frame": "immediate" if red_flags else "within_2_hours",
                "reasoning": "Automated triage based on red flags and vital signs"
            }
            
    def _calculate_triage_level(self, vital_analysis: Dict[str, Any], red_flags: List[Dict[str, Any]]) -> int:
        """Calculate ESI triage level based on analysis"""
        # Level 1: Life-threatening
        if vital_analysis.get("critical") or len(red_flags) >= 2:
            return 1
            
        # Level 2: High risk
        if red_flags or vital_analysis.get("status") == "abnormal":
            return 2
            
        # Level 3: Stable but needs care (default)
        return 3
        
    def _determine_emergency_level(self, triage_assessment: Dict[str, Any]) -> str:
        """Determine emergency level from triage assessment"""
        triage_level = triage_assessment.get("triage_level", 3)
        
        if triage_level == 1:
            return "critical"
        elif triage_level == 2:
            return "emergency"
        elif triage_level == 3:
            return "urgent"
        else:
            return "routine"
            
    def _determine_next_agent(self, emergency_level: str, triage_assessment: Dict[str, Any]) -> str:
        """Determine next agent based on triage results"""
        if emergency_level in ["critical", "emergency"]:
            return "validation"  # Immediate validation for emergencies
        else:
            return "diagnostic"  # Normal diagnostic flow
            
    def _calculate_triage_confidence(self, 
                                   triage_assessment: Dict[str, Any],
                                   clinical_data: Dict[str, Any]) -> float:
        """Calculate confidence in triage assessment"""
        factors = {}
        
        # Data completeness
        vital_signs = clinical_data.get("vital_signs", {})
        factors["data_completeness"] = min(1.0, len(vital_signs) / 5)  # 5 basic vitals
        
        # Symptom clarity
        symptoms = clinical_data.get("symptoms", [])
        factors["symptom_clarity"] = min(1.0, len(symptoms) / 3)
        
        # Red flags certainty
        red_flags = triage_assessment.get("red_flags_detected", [])
        factors["red_flag_certainty"] = 1.0 if red_flags else 0.8
        
        # Vital signs reliability
        vital_analysis = triage_assessment.get("vital_analysis", {})
        factors["vital_reliability"] = 1.0 if vital_analysis.get("status") != "unknown" else 0.5
        
        return self.calculate_confidence(factors)
        
    def _format_assessment_criteria(self) -> str:
        """Format assessment criteria for prompt"""
        return """
ESI Triage Levels:
- Level 1: Immediate life-threatening conditions
- Level 2: High-risk situations requiring rapid assessment
- Level 3: Urgent but stable conditions
- Level 4: Less urgent, routine care
- Level 5: Non-urgent, minor conditions

Critical Vital Sign Thresholds:
- Heart Rate: <40 or >150 bpm
- Blood Pressure: <70 or >200 systolic, >110 diastolic
- Respiratory Rate: <8 or >30 per minute
- Temperature: <94°F or >104°F
- Oxygen Saturation: <85%
"""
        
    def _format_triage_message(self, 
                             triage_assessment: Dict[str, Any],
                             emergency_level: str) -> str:
        """Format triage assessment for display"""
        
        triage_level = triage_assessment.get("triage_level", 3)
        level_names = {1: "IMMEDIATE", 2: "EMERGENT", 3: "URGENT", 4: "LESS URGENT", 5: "NON-URGENT"}
        
        message = f"## general medical Assessment\n\n"
        
        # Emergency indicator
        if emergency_level in ["critical", "emergency"]:
            message += f"🚨 **{emergency_level.upper()} PRIORITY** 🚨\n\n"
            
        message += f"**Triage Level:** ESI {triage_level} - {level_names.get(triage_level, 'URGENT')}\n"
        message += f"**Emergency Level:** {emergency_level.title()}\n\n"
        
        # Immediate actions
        if triage_assessment.get("immediate_actions"):
            message += f"### Immediate Actions Required:\n"
            for action in triage_assessment["immediate_actions"]:
                message += f"- {action}\n"
            message += "\n"
            
        # Warning signs
        if triage_assessment.get("warning_signs"):
            message += f"### ⚠️ Warning Signs Identified:\n"
            for warning in triage_assessment["warning_signs"]:
                message += f"- **{warning}**\n"
            message += "\n"
            
        # Disposition
        disposition = triage_assessment.get("disposition", "Clinical evaluation")
        time_frame = triage_assessment.get("time_frame", "As soon as possible")
        message += f"**Recommended Disposition:** {disposition.replace('_', ' ').title()}\n"
        message += f"**Time Frame:** {time_frame.replace('_', ' ').title()}\n"
        
        return message

    def _get_system_prompt(self):
        print(self)
        return self.triage_prompt