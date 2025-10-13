from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from ..agents.base_agent import BaseMedicalAgent
from ..core.message_protocol import AgentMessage, MessageType, Priority, AgentRole
from ..core.state_manager import MedicalAssistantState

class ValidationAgent(BaseMedicalAgent):
    """
    Specialist agent for validating medical diagnoses and treatment recommendations
    """
    
    def __init__(self, model: Any, tools: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(
            name="validation",
            role=AgentRole.VALIDATION.value,
            model=model,
            tools=tools,
            config=config
        )
        
        self.validation_prompt = """
You are a medical validation specialist. Review and validate the following medical assessment:

Diagnosis History:
{diagnosis_history}

Treatment Plans:
{treatment_plans}

Drug Interactions:
{drug_interactions}

Medical Guidelines Context:
{guidelines_context}

Safety Assessment:
{safety_assessment}

Perform a comprehensive validation including:

1. CLINICAL ACCURACY
   - Verify diagnosis-symptom alignment
   - Check treatment appropriateness
   - Validate dosing recommendations
   - Assess contraindications

2. SAFETY VALIDATION
   - Drug interaction risks
   - Allergy considerations
   - Contraindication checks
   - Dosing safety

3. GUIDELINE COMPLIANCE
   - Standard of care adherence
   - Evidence-based recommendations
   - Best practice alignment
   - Protocol compliance

4. QUALITY ASSURANCE
   - Completeness of assessment
   - Missing critical information
   - Recommendation clarity
   - Documentation quality

5. RISK ASSESSMENT
   - Patient safety risks
   - Treatment complications
   - Monitoring requirements
   - Follow-up needs

6. VALIDATION RESULT
   - Overall validation status: APPROVED/REQUIRES_REVIEW/REJECTED
   - Confidence level (0-1)
   - Critical issues identified
   - Recommendations for improvement

Format your response as structured JSON for parsing.
"""

        self.safety_check_prompt = """
Perform a comprehensive safety check for the following medical recommendations:

Patient Profile:
- Allergies: {allergies}
- Current Medications: {medications}
- Medical History: {medical_history}
- Age/Demographics: {demographics}

Proposed Treatment:
{treatment_plan}

Identify:
1. Safety concerns
2. Potential adverse effects
3. Drug interactions
4. Contraindications
5. Monitoring requirements
"""
        
    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Validate that there's something to validate"""
        return bool(
            state.get("diagnosis_history") or 
            state.get("treatment_plans") or
            state.get("drug_interactions")
        )
        
    async def process(self, 
                     state: MedicalAssistantState,
                     message: Optional[AgentMessage] = None) -> Dict[str, Any]:
        """Process validation request and validate medical recommendations"""
        
        # Fetch guidelines context
        guidelines_context = await self._fetch_guidelines_context(state)
        
        # Perform safety assessment
        safety_assessment = await self._perform_safety_assessment(state)
        
        # Validate medical recommendations
        validation_result = await self._validate_recommendations(
            diagnosis_history=state.get("diagnosis_history", []),
            treatment_plans=state.get("treatment_plans", []),
            drug_interactions=state.get("drug_interactions", {}),
            guidelines_context=guidelines_context,
            safety_assessment=safety_assessment
        )
        
        # Calculate validation confidence
        confidence = self._calculate_validation_confidence(validation_result)
        
        # Determine next steps
        next_agent = self._determine_next_step(validation_result, state)
        
        # Check if human review is required
        needs_review = self._requires_human_review(validation_result, confidence, state)
        
        # Prepare updates
        updates = {
            "validation_history": state.get("validation_history", []) + [{
                "timestamp": datetime.utcnow().isoformat(),
                "validation_result": validation_result,
                "confidence": confidence,
                "agent": self.name
            }],
            "confidence_scores": {
                **state.get("confidence_scores", {}),
                "validation": confidence
            },
            "requires_human_review": needs_review,
            "current_agent": next_agent,
            "safety_checks": safety_assessment
        }
        
        # Add validation message
        updates["messages"] = state.get("messages", []) + [{
            "role": "assistant",
            "content": self._format_validation_message(validation_result),
            "metadata": {
                "agent": self.name,
                "confidence": confidence,
                "validation_status": validation_result.get("validation_status"),
                "validation_data": validation_result
            }
        }]
        
        # Log validation decision
        await self.log_decision(
            decision="validation_completed",
            reasoning=validation_result.get("summary", ""),
            confidence=confidence,
            state=state
        )
        
        return updates
        
    async def _fetch_guidelines_context(self, state: MedicalAssistantState) -> str:
        """Fetch relevant medical guidelines for validation"""
        context_parts = []
        
        # Get latest diagnosis
        diagnosis_history = state.get("diagnosis_history", [])
        if diagnosis_history:
            latest_diagnosis = diagnosis_history[-1]
            primary_condition = latest_diagnosis.get("diagnosis", {}).get("primary_diagnosis", {}).get("condition")
            
            if primary_condition and "guideline_checker" in self.tools:
                result = await self.call_tool("guideline_checker", {
                    "condition": primary_condition,
                    "guideline_types": ["diagnosis", "treatment", "monitoring"]
                })
                if result.get("guidelines"):
                    context_parts.append(f"Guidelines for {primary_condition}:\n{result['guidelines']}")
        
        # Get treatment guidelines
        treatment_plans = state.get("treatment_plans", [])
        if treatment_plans and "guideline_checker" in self.tools:
            latest_treatment = treatment_plans[-1]
            medications = latest_treatment.get("medications", [])
            
            for med in medications[:3]:  # Limit to first 3 medications
                result = await self.call_tool("guideline_checker", {
                    "medication": med.get("name"),
                    "indication": med.get("indication"),
                    "guideline_types": ["prescribing", "dosing", "monitoring"]
                })
                if result.get("guidelines"):
                    context_parts.append(f"Guidelines for {med.get('name')}:\n{result['guidelines']}")
        
        return "\n\n".join(context_parts) if context_parts else "No specific guidelines found"
        
    async def _perform_safety_assessment(self, state: MedicalAssistantState) -> Dict[str, Any]:
        """Perform comprehensive safety assessment"""
        patient_context = state.get("patient_context", {})
        
        safety_data = {
            "allergies": state.get("allergies", []),
            "medications": state.get("medications", []),
            "medical_history": patient_context.get("medical_history", {}),
            "demographics": patient_context.get("demographics", {})
        }
        
        # Get latest treatment plan
        treatment_plans = state.get("treatment_plans", [])
        treatment_plan = treatment_plans[-1] if treatment_plans else {}
        
        prompt = self.safety_check_prompt.format(
            allergies=", ".join(safety_data["allergies"]) or "None reported",
            medications=", ".join([m.get("name", "") for m in safety_data["medications"]]) or "None",
            medical_history=json.dumps(safety_data["medical_history"]),
            demographics=json.dumps(safety_data["demographics"]),
            treatment_plan=json.dumps(treatment_plan)
        )
        
        # Use safety validator tool if available
        if "safety_validator" in self.tools:
            safety_result = await self.call_tool("safety_validator", {
                "patient_data": safety_data,
                "treatment_plan": treatment_plan
            })
            if safety_result:
                return safety_result
        
        # Fallback to LLM analysis
        response = ""
        async for chunk in self.generate_llm_response(prompt):
            response += chunk
            
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "safety_concerns": [],
                "risk_level": "unknown",
                "monitoring_required": True,
                "raw_assessment": response
            }
            
    async def _validate_recommendations(self, **kwargs) -> Dict[str, Any]:
        """Validate medical recommendations using LLM"""
        
        prompt = self.validation_prompt.format(
            diagnosis_history=json.dumps(kwargs.get("diagnosis_history", []), indent=2),
            treatment_plans=json.dumps(kwargs.get("treatment_plans", []), indent=2),
            drug_interactions=json.dumps(kwargs.get("drug_interactions", {}), indent=2),
            guidelines_context=kwargs.get("guidelines_context", "No guidelines available"),
            safety_assessment=json.dumps(kwargs.get("safety_assessment", {}), indent=2)
        )
        
        response = ""
        async for chunk in self.generate_llm_response(prompt):
            response += chunk
            
        try:
            validation_result = json.loads(response)
            
            # Ensure required fields
            required_fields = ["clinical_accuracy", "safety_validation", "guideline_compliance", 
                             "quality_assurance", "risk_assessment", "validation_result"]
            for field in required_fields:
                if field not in validation_result:
                    validation_result[field] = {}
                    
            # Ensure validation status is set
            if "validation_status" not in validation_result.get("validation_result", {}):
                validation_result["validation_result"]["validation_status"] = "REQUIRES_REVIEW"
                
            return validation_result
            
        except json.JSONDecodeError:
            return {
                "clinical_accuracy": {"status": "unknown"},
                "safety_validation": {"status": "unknown"},
                "guideline_compliance": {"status": "unknown"},
                "quality_assurance": {"status": "unknown"},
                "risk_assessment": {"risk_level": "high"},
                "validation_result": {
                    "validation_status": "REJECTED",
                    "confidence": 0.1,
                    "critical_issues": ["Validation parsing failed"],
                    "recommendations": ["Manual review required"]
                },
                "raw_response": response
            }
            
    def _calculate_validation_confidence(self, validation_result: Dict[str, Any]) -> float:
        """Calculate validation confidence score"""
        factors = {}
        
        # Validation status
        status = validation_result.get("validation_result", {}).get("validation_status", "REQUIRES_REVIEW")
        if status == "APPROVED":
            factors["status"] = 1.0
        elif status == "REQUIRES_REVIEW":
            factors["status"] = 0.6
        else:  # REJECTED
            factors["status"] = 0.2
            
        # Clinical accuracy
        clinical_acc = validation_result.get("clinical_accuracy", {})
        factors["clinical"] = clinical_acc.get("score", 0.5)
        
        # Safety validation
        safety_val = validation_result.get("safety_validation", {})
        factors["safety"] = safety_val.get("score", 0.5)
        
        # Guideline compliance
        guideline_comp = validation_result.get("guideline_compliance", {})
        factors["guidelines"] = guideline_comp.get("score", 0.5)
        
        # Critical issues count
        critical_issues = validation_result.get("validation_result", {}).get("critical_issues", [])
        factors["issues"] = max(0.1, 1.0 - len(critical_issues) * 0.2)
        
        return self.calculate_confidence(factors)
        
    def _determine_next_step(self, validation_result: Dict[str, Any], state: MedicalAssistantState) -> str:
        """Determine the next agent or step based on validation results"""
        
        validation_status = validation_result.get("validation_result", {}).get("validation_status", "REQUIRES_REVIEW")
        
        # If approved, go to final output
        if validation_status == "APPROVED":
            return "final_output"
            
        # If rejected, check what needs fixing
        if validation_status == "REJECTED":
            critical_issues = validation_result.get("validation_result", {}).get("critical_issues", [])
            
            # Check if diagnostic issues
            if any("diagnosis" in issue.lower() for issue in critical_issues):
                return "supervisor"  # Route back through supervisor
                
            # Check if treatment issues
            if any("treatment" in issue.lower() for issue in critical_issues):
                return "supervisor"  # Route back through supervisor
                
        # Default to human review for REQUIRES_REVIEW or unclear cases
        return "human_review"
        
    def _requires_human_review(self, 
                              validation_result: Dict[str, Any], 
                              confidence: float,
                              state: MedicalAssistantState) -> bool:
        """Determine if human review is required"""
        
        # Always require review if validation was rejected
        if validation_result.get("validation_result", {}).get("validation_status") == "REJECTED":
            return True
            
        # Require review for low confidence
        if confidence < self.config.get("min_validation_confidence", 0.7):
            return True
            
        # Require review if critical safety issues
        safety_val = validation_result.get("safety_validation", {})
        if safety_val.get("risk_level") == "high":
            return True
            
        # Require review if emergency case
        if state.get("emergency_flag"):
            return True
            
        # Check for critical issues
        critical_issues = validation_result.get("validation_result", {}).get("critical_issues", [])
        if critical_issues:
            return True
            
        return False
        
    def _format_validation_message(self, validation_result: Dict[str, Any]) -> str:
        """Format validation results for display"""
        
        val_result = validation_result.get("validation_result", {})
        status = val_result.get("validation_status", "UNKNOWN")
        confidence = val_result.get("confidence", 0)
        
        message = f"## Validation Assessment\n\n"
        
        # Status indicator
        status_emoji = {
            "APPROVED": "✅",
            "REQUIRES_REVIEW": "⚠️",
            "REJECTED": "❌"
        }
        
        message += f"{status_emoji.get(status, '❓')} **Status:** {status}\n"
        message += f"- Validation Confidence: {confidence:.1%}\n\n"
        
        # Clinical accuracy
        clinical = validation_result.get("clinical_accuracy", {})
        if clinical:
            message += f"### Clinical Accuracy\n"
            message += f"- Score: {clinical.get('score', 0):.1%}\n"
            if clinical.get("issues"):
                for issue in clinical["issues"]:
                    message += f"- ⚠️ {issue}\n"
            message += "\n"
            
        # Safety validation
        safety = validation_result.get("safety_validation", {})
        if safety:
            message += f"### Safety Assessment\n"
            message += f"- Risk Level: {safety.get('risk_level', 'Unknown').upper()}\n"
            if safety.get("concerns"):
                for concern in safety["concerns"]:
                    message += f"- ⚠️ {concern}\n"
            message += "\n"
            
        # Guideline compliance
        guidelines = validation_result.get("guideline_compliance", {})
        if guidelines:
            message += f"### Guideline Compliance\n"
            message += f"- Compliance Score: {guidelines.get('score', 0):.1%}\n"
            if guidelines.get("deviations"):
                for deviation in guidelines["deviations"]:
                    message += f"- ⚠️ {deviation}\n"
            message += "\n"
            
        # Critical issues
        critical_issues = val_result.get("critical_issues", [])
        if critical_issues:
            message += f"### 🚨 Critical Issues\n"
            for issue in critical_issues:
                message += f"- **{issue}**\n"
            message += "\n"
            
        # Recommendations
        recommendations = val_result.get("recommendations", [])
        if recommendations:
            message += f"### Recommendations\n"
            for rec in recommendations:
                message += f"- {rec}\n"
            message += "\n"
                
        return message

    def _get_system_prompt(self):
        return self.validation_prompt