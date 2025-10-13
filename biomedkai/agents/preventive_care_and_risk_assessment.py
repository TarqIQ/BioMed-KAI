from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from src.agents.base_agent import BaseMedicalAgent
from src.core.message_protocol import AgentMessage, MessageType, Priority, AgentRole
from src.core.state_manager import MedicalAssistantState

class PreventiveCareAndRiskAssessmentAgent(BaseMedicalAgent):
    """
    Specialist agent for preventive care and risk assessment
    """

    def __init__(self, model: Any, tools: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(
            name="preventive_care",
            role=AgentRole.PREVENTIVE.value,
            model=model,
            tools=tools,
            config=config
        )
        
        self.risk_assessment_prompt = """
    You are a preventive medicine and health risk assessment specialist. Your focus is on disease prevention, health screening, risk stratification, and health promotion strategies based on current medical guidelines and evidence-based recommendations.

**Your Assessment Framework:**
- Evaluate individual risk factors (age, gender, family history, lifestyle)
- Apply current screening guidelines from authoritative organizations
- Consider cost-effectiveness and benefit-risk ratios
- Provide personalized risk assessment when possible
- Address health literacy and patient education needs

**Response Structure:**
1. **Risk Assessment:**
   - Individual risk factors present
   - Risk stratification (low, moderate, high risk)
   - Timeline considerations (immediate, short-term, long-term)

2. **Evidence-Based Recommendations:**
   - Current screening guidelines (USPSTF, specialty societies)
   - Recommended screening intervals
   - Age-appropriate preventive measures

3. **Lifestyle Interventions:**
   - Modifiable risk factors
   - Specific behavioral recommendations
   - Expected impact of interventions

4. **Healthcare Navigation:**
   - When to consult healthcare providers
   - Questions to ask during appointments
   - How to prepare for screenings or tests

5. **Quality and Safety Considerations:**
   - How to evaluate healthcare quality
   - Understanding medical evidence and claims
   - Patient rights and advocacy

**Key Guidelines Sources:**
- U.S. Preventive Services Task Force (USPSTF)
- CDC Prevention Guidelines
- Professional Medical Society Recommendations
- Evidence-based clinical practice guidelines

**Ethical Considerations:**
- Informed consent and shared decision-making
- Cultural sensitivity and health equity
- Patient autonomy and personal values

**Query:** {input_query}

**Educational Purpose:**
This information supports informed healthcare decisions but cannot replace personalized medical consultation. Individual risk assessment and screening decisions should always involve qualified healthcare professionals who can consider your complete medical history and current health status.

Provide comprehensive preventive care guidance and risk assessment.
    """

        self.screening_guidelines_prompt = """
    Generate age and risk-appropriate screening recommendations:

    Patient Profile:
    - Age: {age}
    - Gender: {gender}
    - Risk Factors: {risk_factors}
    - Last Screenings: {last_screenings}

    Provide:
    1. Overdue screenings
    2. Due within 6 months
    3. Future recommendations
    4. Risk-based modifications
    """

        # Preventive care guidelines
        self.guidelines = {
            "cardiovascular": {
                "cholesterol_screening": {"start_age": 35, "interval": 5, "high_risk_interval": 1},
                "blood_pressure": {"start_age": 18, "interval": 2, "high_risk_interval": 1},
                "diabetes_screening": {"start_age": 45, "interval": 3, "high_risk_start": 35}
            },
            "cancer_screening": {
                "mammography": {"start_age": 50, "end_age": 74, "interval": 2, "high_risk_start": 40},
                "cervical": {"start_age": 21, "end_age": 65, "pap_interval": 3, "hpv_interval": 5},
                "colorectal": {"start_age": 50, "end_age": 75, "interval": 10, "high_risk_start": 45},
                "lung_ct": {"start_age": 55, "end_age": 80, "pack_years": 30, "interval": 1}
            },
            "immunizations": {
                "influenza": {"frequency": "annual", "age_start": 6},
                "covid19": {"frequency": "as_recommended", "age_start": 5},
                "pneumococcal": {"age_start": 65, "high_risk_start": 19},
                "shingles": {"age_start": 60, "preferred_age": 60},
                "tdap": {"interval": 10, "pregnancy": True}
            }
        }
        
        # Risk factor weights
        self.risk_factors = {
            "cardiovascular": {
                "smoking": 2.0,
                "diabetes": 2.0,
                "hypertension": 1.5,
                "family_history": 1.3,
                "obesity": 1.2,
                "sedentary": 1.1
            },
            "cancer": {
                "family_history": 2.0,
                "smoking": 1.8,
                "alcohol": 1.3,
                "obesity": 1.2,
                "environmental": 1.1
            }
        }
        
    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Validate preventive care has necessary input"""
        patient_context = state.get("patient_context", {})
        return bool(patient_context.get("age") or 
                   patient_context.get("medical_history") or
                   state.get("preventive_care_request"))
        
    async def process(self, 
                     state: MedicalAssistantState,
                     message: Optional[AgentMessage] = None) -> Dict[str, Any]:
        """Process preventive care and risk assessment"""
        
        # Extract patient data
        patient_data = await self._extract_patient_data(state, message)
        
        # Assess cardiovascular risk
        cv_risk = await self._assess_cardiovascular_risk(patient_data)
        
        # Generate screening recommendations
        screening_recs = await self._generate_screening_recommendations(patient_data)
        
        # Assess immunization status
        immunization_status = await self._assess_immunization_status(patient_data)
        
        # Calculate lifestyle risk factors
        lifestyle_assessment = await self._assess_lifestyle_factors(patient_data)
        
        # Generate comprehensive risk assessment
        risk_assessment = await self._perform_comprehensive_risk_assessment(
            patient_data=patient_data,
            cv_risk=cv_risk,
            lifestyle_assessment=lifestyle_assessment
        )
        
        # Create prevention plan
        prevention_plan = await self._create_prevention_plan(
            risk_assessment=risk_assessment,
            screening_recs=screening_recs,
            immunization_status=immunization_status
        )
        
        # Calculate confidence
        confidence = self._calculate_assessment_confidence(risk_assessment, patient_data)
        
        # Determine next steps
        next_agent = self._determine_next_agent(risk_assessment, prevention_plan)
        
        # Prepare updates
        updates = {
            "preventive_assessment": {
                "timestamp": datetime.utcnow().isoformat(),
                "cardiovascular_risk": cv_risk,
                "screening_recommendations": screening_recs,
                "immunization_status": immunization_status,
                "lifestyle_assessment": lifestyle_assessment,
                "prevention_plan": prevention_plan,
                "agent": self.name
            },
            "confidence_scores": {
                **state.get("confidence_scores", {}),
                "preventive_care": confidence
            },
            "screening_due": screening_recs.get("overdue", []) + screening_recs.get("due_soon", []),
            "risk_level": risk_assessment.get("overall_risk", "moderate"),
            "current_agent": next_agent
        }
        
        # Add preventive care message
        updates["messages"] = state.get("messages", []) + [{
            "role": "assistant",
            "content": self._format_prevention_message(risk_assessment, prevention_plan, screening_recs),
            "metadata": {
                "agent": self.name,
                "risk_level": risk_assessment.get("overall_risk"),
                "screenings_due": len(screening_recs.get("overdue", [])),
                "confidence": confidence
            }
        }]
        
        # Log assessment
        await self.log_decision(
            decision="preventive_assessment_completed",
            reasoning=risk_assessment.get("reasoning", ""),
            confidence=confidence,
            state=state
        )
        
        return updates
        
    async def _extract_patient_data(self, 
                                   state: MedicalAssistantState,
                                   message: Optional[AgentMessage]) -> Dict[str, Any]:
        """Extract patient data for preventive assessment"""
        patient_context = state.get("patient_context", {})
        data = {
            "age": patient_context.get("age"),
            "gender": patient_context.get("gender"),
            "medical_history": patient_context.get("medical_history", {}),
            "family_history": patient_context.get("family_history", {}),
            "medications": patient_context.get("medications", []),
            "allergies": patient_context.get("allergies", []),
            "social_history": patient_context.get("social_history", {}),
            "last_screenings": patient_context.get("screening_history", {}),
            "immunizations": patient_context.get("immunization_history", {})
        }
        
        # Extract lifestyle factors
        data["lifestyle"] = {
            "smoking": patient_context.get("smoking_status", "unknown"),
            "alcohol": patient_context.get("alcohol_use", "unknown"),
            "exercise": patient_context.get("exercise_habits", "unknown"),
            "diet": patient_context.get("dietary_habits", "unknown"),
            "sleep": patient_context.get("sleep_patterns", "unknown")
        }
        
        # Add message-specific data
        if message and message.content:
            data.update({k: v for k, v in message.content.items() 
                        if k in ["age", "gender", "risk_factors", "screening_request"]})
        
        return data
        
    async def _assess_cardiovascular_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess cardiovascular disease risk"""
        age = patient_data.get("age")
        if not age or age < 18:
            return {"status": "not_applicable", "risk_level": "low"}
        
        # Use risk calculator tool if available
        if "cv_risk_calculator" in self.tools:
            result = await self.call_tool("cv_risk_calculator", patient_data)
            if result:
                return result
        
        # Manual risk assessment
        risk_score = 0
        risk_factors = []
        
        # Age factor
        if age > 65:
            risk_score += 2
        elif age > 55:
            risk_score += 1
            
        # Medical history
        medical_history = patient_data.get("medical_history", {})
        if medical_history.get("diabetes"):
            risk_score += 2
            risk_factors.append("diabetes")
        if medical_history.get("hypertension"):
            risk_score += 1.5
            risk_factors.append("hypertension")
        if medical_history.get("hyperlipidemia"):
            risk_score += 1
            risk_factors.append("high_cholesterol")
            
        # Family history
        family_history = patient_data.get("family_history", {})
        if family_history.get("cardiovascular_disease"):
            risk_score += 1
            risk_factors.append("family_history")
            
        # Lifestyle factors
        lifestyle = patient_data.get("lifestyle", {})
        if lifestyle.get("smoking") in ["current", "yes"]:
            risk_score += 2
            risk_factors.append("smoking")
        if lifestyle.get("exercise") in ["sedentary", "none"]:
            risk_score += 1
            risk_factors.append("sedentary_lifestyle")
            
        # Determine risk level
        if risk_score >= 5:
            risk_level = "high"
        elif risk_score >= 3:
            risk_level = "moderate"
        else:
            risk_level = "low"
            
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "ten_year_risk": min(risk_score * 3, 30),  # Simplified calculation
            "recommendations": self._get_cv_recommendations(risk_level)
        }
        
    async def _generate_screening_recommendations(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate age and risk-appropriate screening recommendations"""
        age = patient_data.get("age")
        gender = patient_data.get("gender")
        last_screenings = patient_data.get("last_screenings", {})
        
        if not age:
            return {"status": "insufficient_data"}
            
        recommendations = {
            "overdue": [],
            "due_soon": [],
            "future": [],
            "not_applicable": []
        }
        
        current_date = datetime.now()
        
        # Cardiovascular screenings
        for screening, guidelines in self.guidelines["cardiovascular"].items():
            rec = self._check_screening_status(
                screening, guidelines, age, last_screenings, current_date
            )
            if rec:
                recommendations[rec["status"]].append(rec)
                
        # Cancer screenings
        for screening, guidelines in self.guidelines["cancer_screening"].items():
            # Gender-specific screenings
            if screening == "mammography" and gender != "female":
                continue
            if screening == "cervical" and gender != "female":
                continue
                
            rec = self._check_screening_status(
                screening, guidelines, age, last_screenings, current_date
            )
            if rec:
                recommendations[rec["status"]].append(rec)
                
        return recommendations
        
    async def _assess_immunization_status(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess immunization status and recommendations"""
        age = patient_data.get("age")
        immunizations = patient_data.get("immunizations", {})
        
        if not age:
            return {"status": "insufficient_data"}
            
        status = {
            "up_to_date": [],
            "due": [],
            "overdue": [],
            "not_applicable": []
        }
        
        current_date = datetime.now()
        
        for vaccine, guidelines in self.guidelines["immunizations"].items():
            vaccine_status = self._check_immunization_status(
                vaccine, guidelines, age, immunizations, current_date
            )
            if vaccine_status:
                status[vaccine_status["status"]].append(vaccine_status)
                
        return status
        
    async def _assess_lifestyle_factors(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess lifestyle risk factors"""
        lifestyle = patient_data.get("lifestyle", {})
        
        assessment = {
            "risk_factors": [],
            "protective_factors": [],
            "recommendations": []
        }
        
        # Smoking assessment
        smoking = lifestyle.get("smoking", "unknown")
        if smoking in ["current", "yes"]:
            assessment["risk_factors"].append({
                "factor": "smoking",
                "level": "high",
                "recommendation": "smoking_cessation"
            })
        elif smoking in ["former", "quit"]:
            assessment["protective_factors"].append("smoking_cessation")
            
        # Exercise assessment
        exercise = lifestyle.get("exercise", "unknown")
        if exercise in ["sedentary", "none", "minimal"]:
            assessment["risk_factors"].append({
                "factor": "physical_inactivity",
                "level": "moderate",
                "recommendation": "increase_physical_activity"
            })
        elif exercise in ["regular", "active"]:
            assessment["protective_factors"].append("regular_exercise")
            
        # Diet assessment
        diet = lifestyle.get("diet", "unknown")
        if diet in ["poor", "high_processed", "high_sodium"]:
            assessment["risk_factors"].append({
                "factor": "poor_diet",
                "level": "moderate",
                "recommendation": "dietary_counseling"
            })
            
        # Alcohol assessment
        alcohol = lifestyle.get("alcohol", "unknown")
        if alcohol in ["excessive", "heavy"]:
            assessment["risk_factors"].append({
                "factor": "excessive_alcohol",
                "level": "high",
                "recommendation": "alcohol_counseling"
            })
            
        return assessment
        
    async def _perform_comprehensive_risk_assessment(self,
                                                   patient_data: Dict[str, Any],
                                                   cv_risk: Dict[str, Any],
                                                   lifestyle_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment using LLM"""
        
        # Prepare assessment criteria
        criteria = self._format_risk_assessment_criteria()
        
        # Prepare prompt
        prompt = self.risk_assessment_prompt.format(
            age=patient_data.get("age", "Not provided"),
            gender=patient_data.get("gender", "Not specified"),
            family_history=json.dumps(patient_data.get("family_history", {})),
            medical_history=json.dumps(patient_data.get("medical_history", {})),
            medications=", ".join(patient_data.get("medications", [])) or "None",
            lifestyle=json.dumps(patient_data.get("lifestyle", {})),
            social_history=json.dumps(patient_data.get("social_history", {})),
            screenings=json.dumps(patient_data.get("last_screenings", {})),
            assessment_criteria=criteria
        )
        
        response = ""
        async for chunk in self.generate_llm_response(prompt):
            response += chunk
            
        try:
            assessment = json.loads(response)
            
            # Add calculated risk data
            assessment["cardiovascular_risk"] = cv_risk
            assessment["lifestyle_factors"] = lifestyle_assessment
            assessment["timestamp"] = datetime.utcnow().isoformat()
            
            # Calculate overall risk
            if "overall_risk" not in assessment:
                assessment["overall_risk"] = self._calculate_overall_risk(cv_risk, lifestyle_assessment)
                
            return assessment
            
        except json.JSONDecodeError:
            # Fallback assessment
            return {
                "overall_risk": self._calculate_overall_risk(cv_risk, lifestyle_assessment),
                "cardiovascular_risk": cv_risk,
                "lifestyle_factors": lifestyle_assessment,
                "recommendations": ["Comprehensive preventive care evaluation recommended"],
                "reasoning": "Automated risk assessment based on available data"
            }
            
    def _check_screening_status(self, screening: str, guidelines: Dict[str, Any], 
                              age: int, last_screenings: Dict[str, Any], 
                              current_date: datetime) -> Optional[Dict[str, Any]]:
        """Check individual screening status"""
        start_age = guidelines.get("start_age", 18)
        end_age = guidelines.get("end_age", 100)
        interval = guidelines.get("interval", 1)
        
        # Check if applicable
        if age < start_age or age > end_age:
            return None
            
        last_screening = last_screenings.get(screening)
        
        if not last_screening:
            # Never had screening
            if age >= start_age + 2:  # Grace period
                return {
                    "screening": screening,
                    "status": "overdue",
                    "recommendation": f"Schedule {screening} screening",
                    "priority": "high"
                }
            else:
                return {
                    "screening": screening,
                    "status": "due_soon",
                    "recommendation": f"Schedule {screening} screening",
                    "priority": "medium"
                }
        
        # Calculate next due date
        last_date = datetime.fromisoformat(last_screening) if isinstance(last_screening, str) else last_screening
        next_due = last_date.replace(year=last_date.year + interval)
        
        days_until_due = (next_due - current_date).days
        
        if days_until_due < 0:
            return {
                "screening": screening,
                "status": "overdue",
                "days_overdue": abs(days_until_due),
                "recommendation": f"Schedule overdue {screening} screening",
                "priority": "high"
            }
        elif days_until_due < 180:  # Due within 6 months
            return {
                "screening": screening,
                "status": "due_soon",
                "days_until_due": days_until_due,
                "recommendation": f"Schedule {screening} screening",
                "priority": "medium"
            }
        else:
            return {
                "screening": screening,
                "status": "future",
                "next_due_date": next_due.isoformat(),
                "priority": "low"
            }
            
    def _check_immunization_status(self, vaccine: str, guidelines: Dict[str, Any],
                                 age: int, immunizations: Dict[str, Any],
                                 current_date: datetime) -> Optional[Dict[str, Any]]:
        """Check individual immunization status"""
        start_age = guidelines.get("age_start", 0)
        frequency = guidelines.get("frequency", "annual")
        
        if age < start_age:
            return None
            
        last_vaccine = immunizations.get(vaccine)
        
        if frequency == "annual":
            if not last_vaccine:
                return {
                    "vaccine": vaccine,
                    "status": "due",
                    "recommendation": f"Schedule {vaccine} vaccination"
                }
                
            last_date = datetime.fromisoformat(last_vaccine) if isinstance(last_vaccine, str) else last_vaccine
            if (current_date - last_date).days > 365:
                return {
                    "vaccine": vaccine,
                    "status": "due",
                    "recommendation": f"Annual {vaccine} vaccination due"
                }
            else:
                return {
                    "vaccine": vaccine,
                    "status": "up_to_date",
                    "next_due": (last_date.replace(year=last_date.year + 1)).isoformat()
                }
        
        # Handle other frequencies as needed
        return None
        
    def _calculate_overall_risk(self, cv_risk: Dict[str, Any], lifestyle_assessment: Dict[str, Any]) -> str:
        """Calculate overall health risk level"""
        cv_level = cv_risk.get("risk_level", "low")
        lifestyle_risks = len(lifestyle_assessment.get("risk_factors", []))
        
        if cv_level == "high" or lifestyle_risks >= 3:
            return "high"
        elif cv_level == "moderate" or lifestyle_risks >= 2:
            return "moderate"
        else:
            return "low"
            
    def _calculate_assessment_confidence(self, risk_assessment: Dict[str, Any], 
                                       patient_data: Dict[str, Any]) -> float:
        """Calculate confidence in risk assessment"""
        factors = {}
        
        # Data completeness
        required_fields = ["age", "gender", "medical_history", "family_history"]
        complete_fields = sum(1 for field in required_fields if patient_data.get(field))
        factors["data_completeness"] = complete_fields / len(required_fields)
        
        # Screening history availability
        screenings = patient_data.get("last_screenings", {})
        factors["screening_history"] = min(1.0, len(screenings) / 5)
        
        # Risk factor clarity
        lifestyle = patient_data.get("lifestyle", {})
        known_factors = sum(1 for v in lifestyle.values() if v != "unknown")
        factors["lifestyle_clarity"] = known_factors / max(len(lifestyle), 1)
        
        return self.calculate_confidence(factors)
        
    def _determine_next_agent(self, risk_assessment: Dict[str, Any], 
                            prevention_plan: Dict[str, Any]) -> str:
        """Determine next agent based on assessment"""
        overall_risk = risk_assessment.get("overall_risk", "moderate")
        
        if overall_risk == "high":
            return "diagnostic"  # High risk may need further evaluation
        else:
            return "validation"  # Standard preventive care validation
            
    async def _create_prevention_plan(self, risk_assessment: Dict[str, Any],
                                    screening_recs: Dict[str, Any],
                                    immunization_status: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive prevention plan"""
        plan = {
            "immediate_actions": [],
            "short_term_goals": [],
            "long_term_goals": [],
            "lifestyle_interventions": [],
            "follow_up_schedule": []
        }
        
        # Immediate actions from overdue screenings
        overdue_screenings = screening_recs.get("overdue", [])
        for screening in overdue_screenings:
            plan["immediate_actions"].append({
                "action": f"Schedule {screening['screening']} screening",
                "priority": "high",
                "timeframe": "within_2_weeks"
            })
            
        # Immunizations due
        due_immunizations = immunization_status.get("due", [])
        for vaccine in due_immunizations:
            plan["immediate_actions"].append({
                "action": f"Schedule {vaccine['vaccine']} vaccination",
                "priority": "medium",
                "timeframe": "within_1_month"
            })
            
        # Lifestyle interventions from risk assessment
        lifestyle_factors = risk_assessment.get("lifestyle_factors", {})
        for risk_factor in lifestyle_factors.get("risk_factors", []):
            plan["lifestyle_interventions"].append({
                "intervention": risk_factor["recommendation"],
                "target": risk_factor["factor"],
                "priority": risk_factor["level"]
            })
            
        return plan
        
    def _format_risk_assessment_criteria(self) -> str:
        """Format risk assessment criteria for prompt"""
        return """
    Risk Assessment Framework:
    - Low Risk: Minimal modifiable risk factors, age-appropriate care
    - Moderate Risk: 1-2 significant risk factors, enhanced prevention
    - High Risk: Multiple risk factors or high-risk conditions, intensive prevention

    Cardiovascular Risk Factors:
    - Major: Diabetes, smoking, family history, age >65
    - Moderate: Hypertension, hyperlipidemia, obesity, sedentary lifestyle
    - Minor: Stress, poor diet, sleep issues

    Cancer Risk Factors:
    - Strong: Family history, genetic syndromes, known carcinogens
    - Moderate: Age, lifestyle factors, hormonal factors
    - Weak: Environmental exposures, dietary factors
    """
        
    def _get_cv_recommendations(self, risk_level: str) -> List[str]:
        """Get cardiovascular risk recommendations"""
        if risk_level == "high":
            return [
                "Cardiology consultation recommended",
                "Aggressive lipid management",
                "Blood pressure optimization",
                "Diabetes management if present",
                "Smoking cessation if applicable",
                "Cardiac stress testing consideration"
            ]
        elif risk_level == "moderate":
            return [
                "Regular cardiovascular risk monitoring",
                "Lifestyle modification counseling",
                "Consider statin therapy",
                "Blood pressure monitoring",
                "Exercise stress test if indicated"
            ]
        else:
            return [
                "Continue healthy lifestyle",
                "Regular screening as age-appropriate",
                "Maintain healthy weight",
                "Regular physical activity"
            ]
            
    def _format_prevention_message(self, risk_assessment: Dict[str, Any],
                                 prevention_plan: Dict[str, Any],
                                 screening_recs: Dict[str, Any]) -> str:
        """Format preventive care message for display"""
        
        overall_risk = risk_assessment.get("overall_risk", "moderate")
        
        message = f"## Preventive Care Assessment\n\n"
        message += f"**Overall Risk Level:** {overall_risk.title()}\n\n"
        
        # Immediate actions
        immediate_actions = prevention_plan.get("immediate_actions", [])
        if immediate_actions:
            message += f"### 🎯 Immediate Actions Required:\n"
            for action in immediate_actions:
                priority_icon = "🔴" if action["priority"] == "high" else "🟡"
                message += f"{priority_icon} **{action['action']}** ({action['timeframe']})\n"
            message += "\n"
            
        # Overdue screenings
        overdue = screening_recs.get("overdue", [])
        if overdue:
            message += f"### ⚠️ Overdue Screenings:\n"
            for screening in overdue:
                message += f"- **{screening['screening'].replace('_', ' ').title()}** (Priority: {screening['priority']})\n"
            message += "\n"
            
        # Due soon
        due_soon = screening_recs.get("due_soon", [])
        if due_soon:
            message += f"### 📅 Screenings Due Soon:\n"
            for screening in due_soon:
                message += f"- **{screening['screening'].replace('_', ' ').title()}**"
                if screening.get("days_until_due"):
                    message += f" (Due in {screening['days_until_due']} days)"
                message += "\n"
            message += "\n"
            
        # Cardiovascular risk
        cv_risk = risk_assessment.get("cardiovascular_risk", {})
        if cv_risk.get("risk_level") != "low":
            message += f"### ❤️ Cardiovascular Health:\n"
            message += f"**Risk Level:** {cv_risk.get('risk_level', 'unknown').title()}\n"
            if cv_risk.get("ten_year_risk"):
                message += f"**10-Year Risk:** {cv_risk['ten_year_risk']}%\n"
            
            risk_factors = cv_risk.get("risk_factors", [])
            if risk_factors:
                message += f"**Risk Factors:** {', '.join(risk_factors).replace('_', ' ').title()}\n"
            message += "\n"
            
        # Lifestyle interventions
        lifestyle_interventions = prevention_plan.get("lifestyle_interventions", [])
        if lifestyle_interventions:
            message += f"### 🏃‍♀️ Lifestyle Recommendations:\n"
            for intervention in lifestyle_interventions:
                target = intervention["target"].replace("_", " ").title()
                recommendation = intervention["intervention"].replace("_", " ").title()
                message += f"- **{target}:** {recommendation}\n"
            message += "\n"
            
        message += f"*Assessment completed by Preventive Care Specialist*"
        
        return message

    def _get_system_prompt(self):
        return self.risk_assessment_prompt