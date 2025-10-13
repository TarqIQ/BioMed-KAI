from typing import Dict, List, Any, Optional
import re
from datetime import datetime

from ...tools.base_tool import BaseTool


class SymptomAnalyzer(BaseTool):
    """
    Analyzes symptoms for medical significance and patterns
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="symptom_analyzer",
            description="Analyzes symptoms for patterns, severity, and medical significance",
            config=config
        )
        
        # Symptom categories
        self.symptom_categories = {
            "constitutional": ["fever", "chills", "fatigue", "weight loss", "night sweats"],
            "respiratory": ["cough", "shortness of breath", "wheezing", "chest pain", "hemoptysis"],
            "cardiovascular": ["chest pain", "palpitations", "edema", "syncope", "claudication"],
            "gastrointestinal": ["nausea", "vomiting", "diarrhea", "abdominal pain", "bloating"],
            "neurological": ["headache", "dizziness", "seizure", "weakness", "numbness"],
            "musculoskeletal": ["joint pain", "muscle pain", "stiffness", "swelling", "limited mobility"],
            "dermatological": ["rash", "itching", "lesions", "discoloration", "swelling"],
            "genitourinary": ["dysuria", "frequency", "urgency", "hematuria", "discharge"],
            "psychiatric": ["anxiety", "depression", "insomnia", "confusion", "hallucinations"],
            "endocrine": ["polyuria", "polydipsia", "heat intolerance", "cold intolerance"]
        }
        
        # Severity indicators
        self.severity_keywords = {
            "mild": ["slight", "minor", "occasional", "mild", "little"],
            "moderate": ["moderate", "some", "noticeable", "persistent"],
            "severe": ["severe", "intense", "extreme", "unbearable", "constant"],
            "critical": ["acute", "sudden onset", "rapidly worsening", "emergency"]
        }
        
        # Red flag symptoms
        self.red_flags = [
            "chest pain", "difficulty breathing", "sudden severe headache",
            "altered consciousness", "severe bleeding", "severe abdominal pain",
            "suicidal thoughts", "signs of stroke", "anaphylaxis symptoms"
        ]
        
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Analyze symptoms"""
        text = kwargs.get("text", "")
        symptoms_list = kwargs.get("symptoms", [])
        
        # Extract symptoms from text if not provided
        if not symptoms_list and text:
            symptoms_list = self._extract_symptoms_from_text(text)
            
        # Analyze each symptom
        analyzed_symptoms = []
        for symptom in symptoms_list:
            analysis = self._analyze_single_symptom(symptom)
            analyzed_symptoms.append(analysis)
            
        # Determine overall patterns
        patterns = self._identify_patterns(analyzed_symptoms)
        
        # Calculate severity
        overall_severity = self._calculate_overall_severity(analyzed_symptoms)
        
        # Check for red flags
        has_red_flags = self._check_red_flags(symptoms_list)
        
        # Generate system review
        system_review = self._generate_system_review(analyzed_symptoms)
        
        return {
            "symptoms": analyzed_symptoms,
            "patterns": patterns,
            "severity": overall_severity,
            "red_flags": has_red_flags,
            "urgent": overall_severity in ["severe", "critical"] or bool(has_red_flags),
            "affected_systems": system_review,
            "recommendation": self._generate_recommendation(overall_severity, has_red_flags),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    def validate_params(self, **kwargs) -> bool:
        """Validate input parameters"""
        return bool(kwargs.get("text") or kwargs.get("symptoms"))
        
    def _extract_symptoms_from_text(self, text: str) -> List[str]:
        """Extract symptoms from free text"""
        symptoms = []
        text_lower = text.lower()
        
        # Look for symptom keywords
        all_symptoms = []
        for category_symptoms in self.symptom_categories.values():
            all_symptoms.extend(category_symptoms)
            
        for symptom in set(all_symptoms):
            if symptom in text_lower:
                symptoms.append(symptom)
                
        # Look for custom patterns
        # Pattern: "experiencing/having/feeling [symptom]"
        pattern = r"(?:experiencing|having|feeling|suffering from|complaining of)\s+(\w+(?:\s+\w+)?)"
        matches = re.findall(pattern, text_lower)
        symptoms.extend(matches)
        
        return list(set(symptoms))
        
    def _analyze_single_symptom(self, symptom: str) -> Dict[str, Any]:
        """Analyze a single symptom"""
        symptom_lower = symptom.lower()
        
        # Determine category
        category = "other"
        for cat, symptoms in self.symptom_categories.items():
            if any(s in symptom_lower for s in symptoms):
                category = cat
                break
                
        # Determine severity
        severity = self._determine_severity(symptom)
        
        # Check if red flag
        is_red_flag = any(flag in symptom_lower for flag in self.red_flags)
        
        return {
            "symptom": symptom,
            "category": category,
            "severity": severity,
            "is_red_flag": is_red_flag
        }
        
    def _determine_severity(self, symptom: str) -> str:
        """Determine symptom severity"""
        symptom_lower = symptom.lower()
        
        for severity, keywords in self.severity_keywords.items():
            if any(keyword in symptom_lower for keyword in keywords):
                return severity
                
        return "moderate"  # Default
        
    def _identify_patterns(self, analyzed_symptoms: List[Dict[str, Any]]) -> List[str]:
        """Identify symptom patterns"""
        patterns = []
        
        # Count symptoms by category
        category_counts = {}
        for symptom in analyzed_symptoms:
            category = symptom["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
            
        # Identify multi-system involvement
        if len([c for c in category_counts.values() if c > 0]) > 3:
            patterns.append("multi-system involvement")
            
        # Check for specific patterns
        categories = list(category_counts.keys())
        
        if "constitutional" in categories and "respiratory" in categories:
            patterns.append("possible infectious process")
            
        if "cardiovascular" in categories and "respiratory" in categories:
            patterns.append("cardiopulmonary pattern")
            
        if "neurological" in categories and any(s["is_red_flag"] for s in analyzed_symptoms):
            patterns.append("neurological emergency pattern")
            
        return patterns
        
    def _calculate_overall_severity(self, analyzed_symptoms: List[Dict[str, Any]]) -> str:
        """Calculate overall severity"""
        if not analyzed_symptoms:
            return "none"
            
        # Check for critical symptoms
        if any(s["severity"] == "critical" for s in analyzed_symptoms):
            return "critical"
            
        # Check for severe symptoms
        severe_count = sum(1 for s in analyzed_symptoms if s["severity"] == "severe")
        if severe_count >= 2 or (severe_count == 1 and len(analyzed_symptoms) >= 3):
            return "severe"
            
        # Check for red flags
        if any(s["is_red_flag"] for s in analyzed_symptoms):
            return "severe"
            
        # Check for moderate symptoms
        moderate_count = sum(1 for s in analyzed_symptoms if s["severity"] == "moderate")
        if moderate_count >= 3:
            return "moderate"
            
        return "mild"
        
    def _check_red_flags(self, symptoms: List[str]) -> List[str]:
        """Check for red flag symptoms"""
        found_red_flags = []
        
        for symptom in symptoms:
            symptom_lower = symptom.lower()
            for red_flag in self.red_flags:
                if red_flag in symptom_lower:
                    found_red_flags.append(red_flag)
                    
        return list(set(found_red_flags))
        
    def _generate_system_review(self, analyzed_symptoms: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Generate review of systems"""
        system_review = {}
        
        for symptom in analyzed_symptoms:
            category = symptom["category"]
            if category not in system_review:
                system_review[category] = []
            system_review[category].append(symptom["symptom"])
            
        return system_review
        
    def _generate_recommendation(self, severity: str, red_flags: List[str]) -> str:
        """Generate recommendation based on analysis"""
        if red_flags:
            return "Immediate medical attention required - red flag symptoms present"
        elif severity == "critical":
            return "Seek emergency medical care immediately"
        elif severity == "severe":
            return "Seek medical attention as soon as possible"
        elif severity == "moderate":
            return "Schedule medical consultation within 24-48 hours"
        else:
            return "Monitor symptoms and seek care if worsening"