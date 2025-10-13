from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from src.agents.base_agent import BaseMedicalAgent
from src.core.message_protocol import AgentMessage, MessageType, Priority, AgentRole
from src.core.state_manager import MedicalAssistantState

class DrugInteractionAgent(BaseMedicalAgent):
    """
    Specialist agent for drug interaction checking and medication safety analysis
    """
    
    def __init__(self, model: Any, tools: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(
            name="drug_interaction",
            role=AgentRole.DRUG_INTERACTION.value,
            model=model,
            tools=tools,
            config=config
        )
        
        self.interaction_prompt = """
You are an expert clinical pharmacist specializing in drug interactions and medication safety.

Patient Information:
- Current Medications: {medications}
- Proposed New Medications: {new_medications}
- Allergies: {allergies}
- Medical Conditions: {conditions}
- Age: {age}
- Weight: {weight}
- Kidney Function: {kidney_function}
- Liver Function: {liver_function}

Drug Interaction Data:
{interaction_data}

Allergy Information:
{allergy_data}

Analyze all medication interactions and provide a comprehensive safety assessment including:

1. DRUG-DRUG INTERACTIONS
   - Major interactions (contraindicated)
   - Moderate interactions (requires monitoring)
   - Minor interactions (minimal clinical significance)
   - Mechanism of interaction
   - Clinical significance and risk level

2. DRUG-ALLERGY INTERACTIONS
   - Direct allergic reactions
   - Cross-reactivity risks
   - Severity of potential reactions

3. DRUG-CONDITION INTERACTIONS
   - Contraindications based on medical conditions
   - Cautions and monitoring requirements
   - Dose adjustments needed

4. DOSING RECOMMENDATIONS
   - Age-appropriate dosing
   - Renal dose adjustments
   - Hepatic dose adjustments
   - Maximum safe doses

5. MONITORING REQUIREMENTS
   - Laboratory monitoring needed
   - Clinical signs to watch for
   - Frequency of monitoring
   - Parameters to track

6. ALTERNATIVE MEDICATIONS
   - Safer alternatives if interactions found
   - Equivalent therapeutic options
   - Risk-benefit considerations

Format your response as structured JSON for parsing.
"""

        self.allergy_check_prompt = """
Analyze potential allergic reactions for the following:

Patient Allergies: {allergies}
Medications to Check: {medications}

Known Cross-Reactivities:
{cross_reactivity_data}

Determine:
1. Direct allergy matches
2. Cross-reactivity risks
3. Severity of potential reactions
4. Safe alternatives
"""

    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Validate drug interaction agent has necessary input"""
        # Need either current medications or treatment plans to analyze
        return bool(
            state.get("medications") or 
            state.get("treatment_plans") or
            state.get("proposed_medications")
        )

    async def process(self, 
                     state: MedicalAssistantState,
                     message: Optional[AgentMessage] = None) -> Dict[str, Any]:
        """Process drug interaction checking request"""
        
        # Extract medication lists
        current_medications = self._extract_current_medications(state)
        proposed_medications = self._extract_proposed_medications(state)
        
        # Get patient context
        patient_context = state.get("patient_context", {})
        allergies = state.get("allergies", [])
        
        # Check drug interactions
        interaction_results = await self._check_drug_interactions(
            current_medications, proposed_medications
        )
        
        # Check allergies
        allergy_results = await self._check_allergies(
            current_medications + proposed_medications, allergies
        )
        
        # Analyze drug-condition interactions
        condition_interactions = await self._check_condition_interactions(
            current_medications + proposed_medications,
            patient_context.get("conditions", [])
        )
        
        # Generate comprehensive safety assessment
        safety_assessment = await self._generate_safety_assessment(
            medications=current_medications + proposed_medications,
            interactions=interaction_results,
            allergies=allergy_results,
            conditions=condition_interactions,
            patient_context=patient_context
        )
        
        # Calculate safety confidence
        confidence = self._calculate_safety_confidence(safety_assessment)
        
        # Determine if human review needed
        needs_review = self.should_request_human_review(confidence, state)
        
        # Check for critical interactions
        has_critical = self._has_critical_interactions(safety_assessment)
        if has_critical:
            needs_review = True
        
        # Prepare updates
        updates = {
            "drug_interactions": state.get("drug_interactions", []) + [{
                "timestamp": datetime.utcnow().isoformat(),
                "current_medications": current_medications,
                "proposed_medications": proposed_medications,
                "safety_assessment": safety_assessment,
                "confidence": confidence,
                "agent": self.name
            }],
            "medication_safety": safety_assessment,
            "confidence_scores": {
                **state.get("confidence_scores", {}),
                "drug_interaction": confidence
            },
            "requires_human_review": needs_review,
            "current_agent": "validation" if not needs_review else "validation"
        }
        
        # Add safety message
        updates["messages"] = state.get("messages", []) + [{
            "role": "assistant",
            "content": self._format_safety_message(safety_assessment),
            "metadata": {
                "agent": self.name,
                "confidence": confidence,
                "safety_data": safety_assessment,
                "critical_interactions": has_critical
            }
        }]
        
        # Log safety analysis
        await self.log_decision(
            decision="drug_safety_analyzed",
            reasoning=f"Analyzed {len(current_medications + proposed_medications)} medications",
            confidence=confidence,
            state=state
        )
        
        return updates

    def _extract_current_medications(self, state: MedicalAssistantState) -> List[Dict[str, Any]]:
        """Extract current medications from state"""
        medications = []
        
        # From medications list
        for med in state.get("medications", []):
            if isinstance(med, str):
                medications.append({"name": med, "dose": "unknown", "frequency": "unknown"})
            elif isinstance(med, dict):
                medications.append(med)
                
        # From patient context
        patient_meds = state.get("patient_context", {}).get("medications", [])
        medications.extend(patient_meds)
        
        return medications

    def _extract_proposed_medications(self, state: MedicalAssistantState) -> List[Dict[str, Any]]:
        """Extract proposed medications from treatment plans"""
        proposed = []
        
        # From treatment plans
        for plan in state.get("treatment_plans", []):
            medications = plan.get("medications", [])
            for med in medications:
                if isinstance(med, str):
                    proposed.append({"name": med, "dose": "as prescribed", "frequency": "as prescribed"})
                elif isinstance(med, dict):
                    proposed.append(med)
                    
        # From proposed medications list
        for med in state.get("proposed_medications", []):
            if isinstance(med, str):
                proposed.append({"name": med, "dose": "proposed", "frequency": "proposed"})
            elif isinstance(med, dict):
                proposed.append(med)
                
        return proposed

    async def _check_drug_interactions(self, 
                                     current_meds: List[Dict[str, Any]],
                                     proposed_meds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for drug-drug interactions"""
        all_medications = current_meds + proposed_meds
        
        if len(all_medications) < 2:
            return {"interactions": [], "total_checked": len(all_medications)}
            
        interactions = []
        
        # Use drug interaction checker tool if available
        if "drug_interaction_checker" in self.tools:
            med_names = [med.get("name", "") for med in all_medications]
            result = await self.call_tool("drug_interaction_checker", {
                "medications": med_names
            })
            interactions.extend(result.get("interactions", []))
        else:
            # Fallback: check common known interactions
            interactions = self._check_common_interactions(all_medications)
            
        return {
            "interactions": interactions,
            "total_checked": len(all_medications),
            "pairs_analyzed": len(all_medications) * (len(all_medications) - 1) // 2
        }

    async def _check_allergies(self, 
                             medications: List[Dict[str, Any]], 
                             allergies: List[str]) -> Dict[str, Any]:
        """Check for drug-allergy interactions"""
        if not allergies or not medications:
            return {"allergy_conflicts": [], "cross_reactivity_risks": []}
            
        results = {"allergy_conflicts": [], "cross_reactivity_risks": []}
        
        # Use allergy checker tool if available
        if "allergy_checker" in self.tools:
            med_names = [med.get("name", "") for med in medications]
            result = await self.call_tool("allergy_checker", {
                "medications": med_names,
                "allergies": allergies
            })
            results.update(result)
        else:
            # Manual allergy checking
            results = self._check_manual_allergies(medications, allergies)
            
        return results

    async def _check_condition_interactions(self, 
                                          medications: List[Dict[str, Any]],
                                          conditions: List[str]) -> Dict[str, Any]:
        """Check for drug-condition interactions"""
        if not conditions or not medications:
            return {"contraindications": [], "cautions": []}
            
        # Use drug database for condition interactions
        if "drug_database" in self.tools:
            med_names = [med.get("name", "") for med in medications]
            result = await self.call_tool("drug_database", {
                "medications": med_names,
                "conditions": conditions,
                "check_type": "contraindications"
            })
            return result
        else:
            return self._check_manual_conditions(medications, conditions)

    async def _generate_safety_assessment(self, **kwargs) -> Dict[str, Any]:
        """Generate comprehensive safety assessment using LLM"""
        
        # Prepare patient data
        patient_context = kwargs.get("patient_context", {})
        
        prompt = self.interaction_prompt.format(
            medications=json.dumps([med.get("name", "") for med in kwargs.get("medications", [])]),
            new_medications=json.dumps([med.get("name", "") for med in kwargs.get("proposed_medications", [])]),
            allergies=", ".join(kwargs.get("allergies", [])) or "None reported",
            conditions=", ".join(patient_context.get("conditions", [])) or "None reported",
            age=patient_context.get("age", "Not specified"),
            weight=patient_context.get("weight", "Not specified"),
            kidney_function=patient_context.get("kidney_function", "Not assessed"),
            liver_function=patient_context.get("liver_function", "Not assessed"),
            interaction_data=json.dumps(kwargs.get("interactions", {})),
            allergy_data=json.dumps(kwargs.get("allergies", {}))
        )
        
        response = ""
        async for chunk in self.generate_llm_response(prompt):
            response += chunk
            
        try:
            assessment = json.loads(response)
            
            # Ensure required fields
            required_fields = ["drug_interactions", "allergy_interactions", 
                             "condition_interactions", "dosing_recommendations",
                             "monitoring_requirements", "alternatives"]
            for field in required_fields:
                if field not in assessment:
                    assessment[field] = []
                    
            return assessment
            
        except json.JSONDecodeError:
            # Fallback structure
            return {
                "drug_interactions": [],
                "allergy_interactions": [],
                "condition_interactions": [],
                "dosing_recommendations": [],
                "monitoring_requirements": [],
                "alternatives": [],
                "raw_analysis": response
            }

    def _calculate_safety_confidence(self, assessment: Dict[str, Any]) -> float:
        """Calculate overall safety confidence"""
        factors = {}
        
        # Interaction severity
        major_interactions = len([i for i in assessment.get("drug_interactions", []) 
                                if i.get("severity") == "major"])
        factors["interaction_safety"] = 1.0 - min(1.0, major_interactions * 0.3)
        
        # Allergy risks
        allergy_risks = len(assessment.get("allergy_interactions", []))
        factors["allergy_safety"] = 1.0 - min(1.0, allergy_risks * 0.4)
        
        # Condition contraindications
        contraindications = len(assessment.get("condition_interactions", []))
        factors["condition_safety"] = 1.0 - min(1.0, contraindications * 0.3)
        
        # Monitoring complexity
        monitoring_items = len(assessment.get("monitoring_requirements", []))
        factors["monitoring"] = max(0.5, 1.0 - monitoring_items * 0.1)
        
        return self.calculate_confidence(factors)

    def _has_critical_interactions(self, assessment: Dict[str, Any]) -> bool:
        """Check if there are critical interactions requiring immediate attention"""
        
        # Major drug interactions
        major_interactions = [i for i in assessment.get("drug_interactions", []) 
                            if i.get("severity") == "major"]
        if major_interactions:
            return True
            
        # Severe allergy risks
        severe_allergies = [a for a in assessment.get("allergy_interactions", [])
                          if a.get("severity") in ["severe", "life-threatening"]]
        if severe_allergies:
            return True
            
        # Absolute contraindications
        contraindications = [c for c in assessment.get("condition_interactions", [])
                           if c.get("type") == "contraindication"]
        if contraindications:
            return True
            
        return False

    def _check_common_interactions(self, medications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback method for checking common drug interactions"""
        # This would contain common interaction patterns
        # In production, this would be more comprehensive
        interactions = []
        
        med_names = [med.get("name", "").lower() for med in medications]
        
        # Example: Warfarin interactions
        if "warfarin" in med_names:
            interacting_drugs = ["aspirin", "ibuprofen", "naproxen"]
            for drug in interacting_drugs:
                if drug in med_names:
                    interactions.append({
                        "drug1": "warfarin",
                        "drug2": drug,
                        "severity": "major",
                        "mechanism": "Increased bleeding risk"
                    })
                    
        return interactions

    def _check_manual_allergies(self, 
                               medications: List[Dict[str, Any]], 
                               allergies: List[str]) -> Dict[str, Any]:
        """Manual allergy checking fallback"""
        conflicts = []
        cross_reactivity = []
        
        for med in medications:
            med_name = med.get("name", "").lower()
            for allergy in allergies:
                allergy_lower = allergy.lower()
                if allergy_lower in med_name or med_name in allergy_lower:
                    conflicts.append({
                        "medication": med_name,
                        "allergy": allergy,
                        "severity": "high"
                    })
                    
        return {
            "allergy_conflicts": conflicts,
            "cross_reactivity_risks": cross_reactivity
        }

    def _check_manual_conditions(self, 
                                medications: List[Dict[str, Any]],
                                conditions: List[str]) -> Dict[str, Any]:
        """Manual condition checking fallback"""
        contraindications = []
        cautions = []
        
        # Example condition checks
        condition_map = {
            "kidney disease": ["metformin", "nsaids"],
            "liver disease": ["acetaminophen", "statins"],
            "heart failure": ["nsaids", "calcium channel blockers"]
        }
        
        for condition in conditions:
            condition_lower = condition.lower()
            for key, contraindicated_drugs in condition_map.items():
                if key in condition_lower:
                    for med in medications:
                        med_name = med.get("name", "").lower()
                        for contra_drug in contraindicated_drugs:
                            if contra_drug in med_name:
                                contraindications.append({
                                    "medication": med_name,
                                    "condition": condition,
                                    "type": "contraindication"
                                })
                                
        return {
            "contraindications": contraindications,
            "cautions": cautions
        }

    def _format_safety_message(self, assessment: Dict[str, Any]) -> str:
        """Format safety assessment for display"""
        message = "## Medication Safety Analysis\n\n"
        
        # Drug interactions
        drug_interactions = assessment.get("drug_interactions", [])
        if drug_interactions:
            message += "### 🔄 Drug Interactions\n"
            for interaction in drug_interactions:
                severity = interaction.get("severity", "unknown").upper()
                emoji = "🔴" if severity == "MAJOR" else "🟡" if severity == "MODERATE" else "🟢"
                message += f"{emoji} **{severity}**: {interaction.get('drug1')} + {interaction.get('drug2')}\n"
                message += f"   - {interaction.get('mechanism', 'Unknown mechanism')}\n"
        else:
            message += "### ✅ Drug Interactions\nNo significant drug interactions identified.\n"
            
        # Allergy interactions
        allergy_interactions = assessment.get("allergy_interactions", [])
        if allergy_interactions:
            message += "\n### ⚠️ Allergy Alerts\n"
            for allergy in allergy_interactions:
                message += f"- **{allergy.get('medication')}** conflicts with allergy to {allergy.get('allergy')}\n"
        else:
            message += "\n### ✅ Allergy Check\nNo allergy conflicts identified.\n"
            
        # Condition interactions
        condition_interactions = assessment.get("condition_interactions", [])
        if condition_interactions:
            message += "\n### 🏥 Medical Condition Considerations\n"
            for condition in condition_interactions:
                message += f"- **{condition.get('medication')}** may be contraindicated in {condition.get('condition')}\n"
                
        # Monitoring requirements
        monitoring = assessment.get("monitoring_requirements", [])
        if monitoring:
            message += "\n### 📊 Monitoring Requirements\n"
            for monitor in monitoring:
                message += f"- {monitor.get('parameter', 'Unknown')}: {monitor.get('frequency', 'Regular monitoring')}\n"
                
        # Alternatives if needed
        alternatives = assessment.get("alternatives", [])
        if alternatives:
            message += "\n### 💊 Alternative Medications\n"
            for alt in alternatives:
                message += f"- Consider {alt.get('medication')} instead of {alt.get('replaces')}\n"
                
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
        return """You are a clinical pharmacology specialist focused on drug interactions, medication safety, and pharmaceutical guidance. Your expertise covers drug-drug interactions, drug-food interactions, side effects, and medication management.

**Your Analytical Approach:**
- Identify the specific medications, supplements, or substances involved
- Classify interaction severity (contraindicated, major, moderate, minor)
- Explain the pharmacological mechanism of interactions
- Provide practical management strategies
- Consider patient-specific risk factors

**Response Framework:**
1. **Interaction Assessment:**
   - Type of interaction (pharmacokinetic/pharmacodynamic)
   - Severity level and clinical significance
   - Onset timing (immediate, delayed, variable)

2. **Mechanism Explanation:**
   - How the interaction occurs (enzyme inhibition/induction, receptor competition, etc.)
   - Which drug affects which (bidirectional or unidirectional)

3. **Clinical Consequences:**
   - Potential effects on drug efficacy
   - Risk of adverse reactions
   - Symptoms to monitor for

4. **Management Recommendations:**
   - Timing modifications (spacing doses)
   - Dose adjustments if applicable
   - Alternative medications to consider
   - Monitoring parameters

5. **When to Seek Immediate Help:**
   - Red flag symptoms requiring urgent medical attention

**Special Considerations:**
- Patient age, kidney/liver function
- Multiple medication regimens (polypharmacy)
- Over-the-counter medications and supplements
- Food and lifestyle interactions

**Query:** {input_query}

**Critical Safety Note:**
Never discontinue or modify prescribed medications without consulting your healthcare provider or pharmacist. This information is educational and cannot replace professional pharmaceutical consultation.

Provide comprehensive interaction analysis and safety guidance."""
    
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