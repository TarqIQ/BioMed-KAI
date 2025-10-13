from typing import Dict, List, Any, Optional
import re
from datetime import datetime
from ...tools.base_tool import BaseTool

class MedicalCalculator(BaseTool):
    """
    Performs medical calculations including BMI, dosages, scores, and clinical metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="medical_calculator",
            description="Performs various medical calculations and clinical scoring",
            config=config
        )
        
        # Calculation types
        self.calculation_types = {
            "bmi": "Body Mass Index calculation",
            "bsa": "Body Surface Area calculation",
            "gfr": "Glomerular Filtration Rate estimation",
            "creatinine_clearance": "Creatinine clearance calculation",
            "dosage": "Medication dosage calculation",
            "wells_score": "Wells score for PE/DVT",
            "chads2_vasc": "CHA2DS2-VASc score for stroke risk",
            "apache_ii": "APACHE II severity score",
            "glasgow_coma": "Glasgow Coma Scale",
            "pediatric_weight": "Pediatric weight estimation",
            "fluid_requirements": "Daily fluid requirements",
            "caloric_needs": "Caloric requirements calculation"
        }
        
        # Normal ranges
        self.normal_ranges = {
            "bmi": {"underweight": (0, 18.5), "normal": (18.5, 25), "overweight": (25, 30), "obese": (30, 100)},
            "gfr": {"normal": (90, 200), "mild_decrease": (60, 89), "moderate_decrease": (30, 59), "severe_decrease": (15, 29), "kidney_failure": (0, 14)},
            "blood_pressure": {"normal": {"systolic": (90, 120), "diastolic": (60, 80)}, "elevated": {"systolic": (120, 129), "diastolic": (60, 80)}},
            "heart_rate": {"bradycardia": (0, 60), "normal": (60, 100), "tachycardia": (100, 200)}
        }
        
        # Risk categories
        self.risk_categories = {
            "low": "Low risk - routine monitoring",
            "moderate": "Moderate risk - regular follow-up",
            "high": "High risk - frequent monitoring required",
            "critical": "Critical risk - immediate intervention needed"
        }
        
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Perform medical calculation"""
        calculation_type = kwargs.get("calculation_type", "").lower()
        
        if calculation_type not in self.calculation_types:
            available_types = list(self.calculation_types.keys())
            return {
                "error": f"Unknown calculation type. Available types: {available_types}",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Route to appropriate calculation method
        result = {}
        
        if calculation_type == "bmi":
            result = self._calculate_bmi(kwargs)
        elif calculation_type == "bsa":
            result = self._calculate_bsa(kwargs)
        elif calculation_type == "gfr":
            result = self._calculate_gfr(kwargs)
        elif calculation_type == "creatinine_clearance":
            result = self._calculate_creatinine_clearance(kwargs)
        elif calculation_type == "dosage":
            result = self._calculate_dosage(kwargs)
        elif calculation_type == "wells_score":
            result = self._calculate_wells_score(kwargs)
        elif calculation_type == "chads2_vasc":
            result = self._calculate_chads2_vasc(kwargs)
        elif calculation_type == "glasgow_coma":
            result = self._calculate_glasgow_coma(kwargs)
        elif calculation_type == "pediatric_weight":
            result = self._calculate_pediatric_weight(kwargs)
        elif calculation_type == "fluid_requirements":
            result = self._calculate_fluid_requirements(kwargs)
        elif calculation_type == "caloric_needs":
            result = self._calculate_caloric_needs(kwargs)
        else:
            result = {"error": "Calculation not implemented yet"}
        
        result["calculation_type"] = calculation_type
        result["timestamp"] = datetime.utcnow().isoformat()
        
        return result
        
    def validate_params(self, **kwargs) -> bool:
        """Validate input parameters"""
        calculation_type = kwargs.get("calculation_type", "").lower()
        return calculation_type in self.calculation_types
        
    def _calculate_bmi(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Body Mass Index"""
        try:
            weight = float(params.get("weight", 0))  # kg
            height = float(params.get("height", 0))  # meters
            
            if weight <= 0 or height <= 0:
                return {"error": "Weight and height must be positive values"}
            
            bmi = weight / (height ** 2)
            category = self._get_bmi_category(bmi)
            
            return {
                "value": round(bmi, 2),
                "unit": "kg/m²",
                "category": category,
                "interpretation": self._interpret_bmi(category),
                "normal_range": "18.5 - 24.9 kg/m²"
            }
        except (ValueError, TypeError):
            return {"error": "Invalid weight or height values"}
            
    def _calculate_bsa(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Body Surface Area using DuBois formula"""
        try:
            weight = float(params.get("weight", 0))  # kg
            height = float(params.get("height", 0)) * 100  # convert m to cm
            
            if weight <= 0 or height <= 0:
                return {"error": "Weight and height must be positive values"}
            
            # DuBois formula
            bsa = 0.007184 * (weight ** 0.425) * (height ** 0.725)
            
            return {
                "value": round(bsa, 2),
                "unit": "m²",
                "formula": "DuBois formula",
                "normal_range": "1.6 - 2.0 m² (adult)"
            }
        except (ValueError, TypeError):
            return {"error": "Invalid weight or height values"}
            
    def _calculate_gfr(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate estimated GFR using MDRD equation"""
        try:
            creatinine = float(params.get("creatinine", 0))  # mg/dL
            age = int(params.get("age", 0))
            gender = params.get("gender", "").lower()
            race = params.get("race", "").lower()
            
            if creatinine <= 0 or age <= 0:
                return {"error": "Creatinine and age must be positive values"}
            
            # MDRD equation
            gfr = 175 * (creatinine ** -1.154) * (age ** -0.203)
            
            if gender == "female":
                gfr *= 0.742
            
            if race in ["african american", "black"]:
                gfr *= 1.212
                
            category = self._get_gfr_category(gfr)
            
            return {
                "value": round(gfr, 1),
                "unit": "mL/min/1.73m²",
                "category": category,
                "interpretation": self._interpret_gfr(category),
                "formula": "MDRD equation"
            }
        except (ValueError, TypeError):
            return {"error": "Invalid parameter values"}
            
    def _calculate_creatinine_clearance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate creatinine clearance using Cockcroft-Gault"""
        try:
            creatinine = float(params.get("creatinine", 0))  # mg/dL
            age = int(params.get("age", 0))
            weight = float(params.get("weight", 0))  # kg
            gender = params.get("gender", "").lower()
            
            if creatinine <= 0 or age <= 0 or weight <= 0:
                return {"error": "All values must be positive"}
            
            # Cockcroft-Gault equation
            ccr = ((140 - age) * weight) / (72 * creatinine)
            
            if gender == "female":
                ccr *= 0.85
                
            return {
                "value": round(ccr, 1),
                "unit": "mL/min",
                "formula": "Cockcroft-Gault equation",
                "normal_range": "90-120 mL/min"
            }
        except (ValueError, TypeError):
            return {"error": "Invalid parameter values"}
            
    def _calculate_dosage(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate medication dosage"""
        try:
            weight = float(params.get("weight", 0))  # kg
            dose_per_kg = float(params.get("dose_per_kg", 0))  # mg/kg
            frequency = params.get("frequency", "daily")
            
            if weight <= 0 or dose_per_kg <= 0:
                return {"error": "Weight and dose per kg must be positive"}
            
            total_dose = weight * dose_per_kg
            
            return {
                "total_dose": round(total_dose, 2),
                "unit": "mg",
                "frequency": frequency,
                "dose_per_kg": dose_per_kg,
                "patient_weight": weight
            }
        except (ValueError, TypeError):
            return {"error": "Invalid dosage parameters"}
            
    def _calculate_wells_score(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Wells score for PE probability"""
        criteria = {
            "clinical_signs_dvt": params.get("clinical_signs_dvt", False),
            "pe_likely": params.get("pe_likely", False),
            "heart_rate_over_100": params.get("heart_rate_over_100", False),
            "immobilization": params.get("immobilization", False),
            "previous_pe_dvt": params.get("previous_pe_dvt", False),
            "hemoptysis": params.get("hemoptysis", False),
            "malignancy": params.get("malignancy", False)
        }
        
        score = 0
        score += 3 if criteria["clinical_signs_dvt"] else 0
        score += 3 if criteria["pe_likely"] else 0
        score += 1.5 if criteria["heart_rate_over_100"] else 0
        score += 1.5 if criteria["immobilization"] else 0
        score += 1.5 if criteria["previous_pe_dvt"] else 0
        score += 1 if criteria["hemoptysis"] else 0
        score += 1 if criteria["malignancy"] else 0
        
        if score <= 4:
            risk = "low"
            probability = "PE unlikely"
        else:
            risk = "high"
            probability = "PE likely"
            
        return {
            "score": score,
            "risk_level": risk,
            "probability": probability,
            "recommendation": "Consider D-dimer if low risk, CT if high risk"
        }
        
    def _calculate_chads2_vasc(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate CHA2DS2-VASc score"""
        criteria = {
            "chf": params.get("chf", False),
            "hypertension": params.get("hypertension", False),
            "age": int(params.get("age", 0)),
            "diabetes": params.get("diabetes", False),
            "stroke_history": params.get("stroke_history", False),
            "vascular_disease": params.get("vascular_disease", False),
            "gender": params.get("gender", "").lower()
        }
        
        score = 0
        score += 1 if criteria["chf"] else 0
        score += 1 if criteria["hypertension"] else 0
        score += 2 if criteria["age"] >= 75 else (1 if criteria["age"] >= 65 else 0)
        score += 1 if criteria["diabetes"] else 0
        score += 2 if criteria["stroke_history"] else 0
        score += 1 if criteria["vascular_disease"] else 0
        score += 1 if criteria["gender"] == "female" else 0
        
        if score == 0:
            risk = "low"
            recommendation = "No anticoagulation recommended"
        elif score == 1:
            risk = "moderate"
            recommendation = "Consider anticoagulation"
        else:
            risk = "high"
            recommendation = "Anticoagulation recommended"
            
        return {
            "score": score,
            "risk_level": risk,
            "recommendation": recommendation,
            "annual_stroke_risk": self._get_stroke_risk_percentage(score)
        }
        
    def _calculate_glasgow_coma(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Glasgow Coma Scale"""
        eye_response = int(params.get("eye_response", 4))  # 1-4
        verbal_response = int(params.get("verbal_response", 5))  # 1-5
        motor_response = int(params.get("motor_response", 6))  # 1-6
        
        total_score = eye_response + verbal_response + motor_response
        
        if total_score >= 13:
            severity = "mild"
        elif total_score >= 9:
            severity = "moderate"
        else:
            severity = "severe"
            
        return {
            "eye_response": eye_response,
            "verbal_response": verbal_response,
            "motor_response": motor_response,
            "total_score": total_score,
            "severity": severity,
            "max_score": 15
        }
        
    def _calculate_pediatric_weight(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate estimated pediatric weight"""
        age_months = int(params.get("age_months", 0))
        
        if age_months < 0:
            return {"error": "Age must be positive"}
        
        if age_months <= 12:
            # For infants: (age in months + 9) / 2
            estimated_weight = (age_months + 9) / 2
        else:
            # For children: (age in years × 2) + 8
            age_years = age_months / 12
            estimated_weight = (age_years * 2) + 8
            
        return {
            "estimated_weight": round(estimated_weight, 1),
            "unit": "kg",
            "age_months": age_months,
            "note": "This is an estimation - actual weight may vary"
        }
        
    def _calculate_fluid_requirements(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate daily fluid requirements"""
        try:
            weight = float(params.get("weight", 0))  # kg
            
            if weight <= 0:
                return {"error": "Weight must be positive"}
            
            # Holliday-Segar method
            if weight <= 10:
                fluid_ml = weight * 100
            elif weight <= 20:
                fluid_ml = 1000 + (weight - 10) * 50
            else:
                fluid_ml = 1500 + (weight - 20) * 20
                
            return {
                "daily_requirement": fluid_ml,
                "unit": "mL/day",
                "hourly_rate": round(fluid_ml / 24, 1),
                "method": "Holliday-Segar"
            }
        except (ValueError, TypeError):
            return {"error": "Invalid weight value"}
            
    def _calculate_caloric_needs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate caloric requirements using Harris-Benedict equation"""
        try:
            weight = float(params.get("weight", 0))  # kg
            height = float(params.get("height", 0)) * 100  # convert m to cm
            age = int(params.get("age", 0))
            gender = params.get("gender", "").lower()
            activity_level = params.get("activity_level", "sedentary").lower()
            
            if weight <= 0 or height <= 0 or age <= 0:
                return {"error": "All values must be positive"}
            
            # Harris-Benedict equation
            if gender == "male":
                bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
            else:
                bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
            
            # Activity multipliers
            activity_multipliers = {
                "sedentary": 1.2,
                "light": 1.375,
                "moderate": 1.55,
                "active": 1.725,
                "very_active": 1.9
            }
            
            multiplier = activity_multipliers.get(activity_level, 1.2)
            total_calories = bmr * multiplier
            
            return {
                "bmr": round(bmr, 0),
                "total_calories": round(total_calories, 0),
                "unit": "kcal/day",
                "activity_level": activity_level,
                "formula": "Harris-Benedict equation"
            }
        except (ValueError, TypeError):
            return {"error": "Invalid parameter values"}
    
    def _get_bmi_category(self, bmi: float) -> str:
        """Get BMI category"""
        if bmi < 18.5:
            return "underweight"
        elif bmi < 25:
            return "normal"
        elif bmi < 30:
            return "overweight"
        else:
            return "obese"
            
    def _interpret_bmi(self, category: str) -> str:
        """Interpret BMI category"""
        interpretations = {
            "underweight": "Below normal weight range",
            "normal": "Healthy weight range",
            "overweight": "Above normal weight range",
            "obese": "Significantly above normal weight range"
        }
        return interpretations.get(category, "Unknown category")
        
    def _get_gfr_category(self, gfr: float) -> str:
        """Get GFR category"""
        if gfr >= 90:
            return "normal"
        elif gfr >= 60:
            return "mild_decrease"
        elif gfr >= 30:
            return "moderate_decrease"
        elif gfr >= 15:
            return "severe_decrease"
        else:
            return "kidney_failure"
            
    def _interpret_gfr(self, category: str) -> str:
        """Interpret GFR category"""
        interpretations = {
            "normal": "Normal or high kidney function",
            "mild_decrease": "Mildly decreased kidney function",
            "moderate_decrease": "Moderately decreased kidney function",
            "severe_decrease": "Severely decreased kidney function",
            "kidney_failure": "Kidney failure"
        }
        return interpretations.get(category, "Unknown category")
        
    def _get_stroke_risk_percentage(self, score: int) -> str:
        """Get annual stroke risk percentage for CHA2DS2-VASc score"""
        risk_percentages = {
            0: "0%", 1: "1.3%", 2: "2.2%", 3: "3.2%", 4: "4.0%",
            5: "6.7%", 6: "9.8%", 7: "9.6%", 8: "6.7%", 9: "15.2%"
        }
        return risk_percentages.get(score, f"{score * 2}%")