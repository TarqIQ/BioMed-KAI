import aiohttp
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import re
from ...tools.base_tool import BaseTool
from ...config.settings import settings

class DrugInteractionChecker(BaseTool):
    """
    Check for drug interactions using FDA and medical databases
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="drug_interaction_checker",
            description="Check for drug interactions and contraindications",
            config=config
        )
        
        # FDA OpenFDA API endpoints
        self.fda_base_url = "https://api.fda.gov"
        self.drug_label_url = f"{self.fda_base_url}/drug/label.json"
        self.drug_event_url = f"{self.fda_base_url}/drug/event.json"
        
        # Known drug interaction patterns
        self.interaction_keywords = [
            "contraindicated", "interaction", "avoid", "caution",
            "increase", "decrease", "concurrent", "concomitant"
        ]
        
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute drug interaction check"""
        primary_drug = kwargs.get("primary_drug", "")
        secondary_drugs = kwargs.get("secondary_drugs", [])
        patient_conditions = kwargs.get("patient_conditions", [])
        
        if isinstance(secondary_drugs, str):
            secondary_drugs = [secondary_drugs]
            
        results = {
            "primary_drug": primary_drug,
            "secondary_drugs": secondary_drugs,
            "interactions": [],
            "warnings": [],
            "contraindications": [],
            "recommendations": [],
            "severity_summary": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check interactions between primary and secondary drugs
        for secondary_drug in secondary_drugs:
            interaction = await self._check_drug_interaction(primary_drug, secondary_drug)
            if interaction:
                results["interactions"].append(interaction)
                
        # Check contraindications with patient conditions
        contraindications = await self._check_contraindications(
            primary_drug, patient_conditions
        )
        results["contraindications"].extend(contraindications)
        
        # Check for general warnings
        warnings = await self._get_drug_warnings(primary_drug)
        results["warnings"].extend(warnings)
        
        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)
        
        # Calculate severity summary
        results["severity_summary"] = self._calculate_severity_summary(results)
        
        return results
        
    def validate_params(self, **kwargs) -> bool:
        """Validate parameters"""
        primary_drug = kwargs.get("primary_drug", "")
        return bool(primary_drug and len(primary_drug.strip()) > 0)
        
    async def _check_drug_interaction(self, drug1: str, drug2: str) -> Optional[Dict[str, Any]]:
        """Check interaction between two drugs"""
        
        # Search for drug labels mentioning both drugs
        interaction_data = await self._search_interaction_data(drug1, drug2)
        
        if not interaction_data:
            return None
            
        # Parse interaction information
        interaction = {
            "drug1": drug1,
            "drug2": drug2,
            "severity": "unknown",
            "mechanism": "",
            "clinical_effect": "",
            "management": "",
            "evidence_level": "low",
            "sources": []
        }
        
        # Analyze the interaction data
        interaction.update(self._analyze_interaction_data(interaction_data, drug1, drug2))
        
        return interaction
        
    async def _search_interaction_data(self, drug1: str, drug2: str) -> List[Dict[str, Any]]:
        """Search FDA database for interaction data"""
        results = []
        
        # Search drug labels
        label_data = await self._search_drug_labels(drug1, drug2)
        results.extend(label_data)
        
        # Search adverse event reports
        event_data = await self._search_adverse_events(drug1, drug2)
        results.extend(event_data)
        
        return results
        
    async def _search_drug_labels(self, drug1: str, drug2: str) -> List[Dict[str, Any]]:
        """Search FDA drug labels for interaction information"""
        results = []
        
        # Search for drug1 labels mentioning drug2
        query = f'"{drug1}" AND "{drug2}"'
        
        params = {
            "search": f"drug_interactions:({query}) OR contraindications:({query}) OR warnings:({query})",
            "limit": 10
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.drug_label_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for result in data.get("results", []):
                            # Extract relevant sections
                            label_data = {
                                "type": "drug_label",
                                "drug_interactions": result.get("drug_interactions", []),
                                "contraindications": result.get("contraindications", []),
                                "warnings": result.get("warnings", []),
                                "brand_name": result.get("openfda", {}).get("brand_name", []),
                                "generic_name": result.get("openfda", {}).get("generic_name", [])
                            }
                            results.append(label_data)
                            
            except Exception as e:
                self.logger.error(f"FDA label search error: {str(e)}")
                
        return results
        
    async def _search_adverse_events(self, drug1: str, drug2: str) -> List[Dict[str, Any]]:
        """Search FDA adverse event database"""
        results = []
        
        # Search for cases with both drugs
        params = {
            "search": f'patient.drug.medicinalproduct:"{drug1}" AND patient.drug.medicinalproduct:"{drug2}"',
            "count": "patient.reaction.reactionmeddrapt.exact",
            "limit": 20
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.drug_event_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for result in data.get("results", []):
                            event_data = {
                                "type": "adverse_event",
                                "reaction": result.get("term", ""),
                                "count": result.get("count", 0)
                            }
                            results.append(event_data)
                            
            except Exception as e:
                self.logger.error(f"FDA adverse event search error: {str(e)}")
                
        return results
        
    def _analyze_interaction_data(self, data: List[Dict[str, Any]], drug1: str, drug2: str) -> Dict[str, Any]:
        """Analyze interaction data to extract key information"""
        
        analysis = {
            "severity": "unknown",
            "mechanism": "",
            "clinical_effect": "",
            "management": "",
            "evidence_level": "low",
            "sources": []
        }
        
        # Analyze drug label data
        for item in data:
            if item["type"] == "drug_label":
                # Check drug interactions section
                interactions = item.get("drug_interactions", [])
                for interaction_text in interactions:
                    if isinstance(interaction_text, str):
                        severity = self._extract_severity(interaction_text)
                        if severity != "unknown":
                            analysis["severity"] = severity
                            
                        mechanism = self._extract_mechanism(interaction_text)
                        if mechanism:
                            analysis["mechanism"] = mechanism
                            
                        management = self._extract_management(interaction_text)
                        if management:
                            analysis["management"] = management
                            
                # Check contraindications
                contraindications = item.get("contraindications", [])
                for contra_text in contraindications:
                    if isinstance(contra_text, str) and (drug2.lower() in contra_text.lower()):
                        analysis["severity"] = "major"
                        analysis["clinical_effect"] = "Contraindicated"
                        
            elif item["type"] == "adverse_event":
                # High frequency adverse events suggest possible interaction
                if item.get("count", 0) > 10:
                    analysis["clinical_effect"] = f"Increased risk of {item['reaction']}"
                    analysis["evidence_level"] = "moderate"
                    
        # Set default values if not found
        if analysis["severity"] == "unknown":
            analysis["severity"] = "moderate"  # Conservative approach
            
        return analysis
        
    def _extract_severity(self, text: str) -> str:
        """Extract severity from interaction text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["contraindicated", "avoid", "severe"]):
            return "major"
        elif any(word in text_lower for word in ["caution", "monitor", "moderate"]):
            return "moderate"
        elif any(word in text_lower for word in ["minor", "mild"]):
            return "minor"
            
        return "unknown"
        
    def _extract_mechanism(self, text: str) -> str:
        """Extract interaction mechanism"""
        mechanisms = {
            "cyp": "CYP enzyme interaction",
            "p-glycoprotein": "P-glycoprotein interaction",
            "protein binding": "Protein binding displacement",
            "renal": "Renal clearance interaction",
            "absorption": "Absorption interference",
            "metabolism": "Metabolic interaction"
        }
        
        text_lower = text.lower()
        for key, mechanism in mechanisms.items():
            if key in text_lower:
                return mechanism
                
        return ""
        
    def _extract_management(self, text: str) -> str:
        """Extract management recommendations"""
        text_lower = text.lower()
        
        if "monitor" in text_lower:
            return "Monitor patient closely"
        elif "dose" in text_lower and ("reduce" in text_lower or "adjust" in text_lower):
            return "Consider dose adjustment"
        elif "avoid" in text_lower:
            return "Avoid concurrent use"
        elif "separate" in text_lower:
            return "Separate administration times"
            
        return ""
        
    async def _check_contraindications(self, drug: str, conditions: List[str]) -> List[Dict[str, Any]]:
        """Check for contraindications with patient conditions"""
        contraindications = []
        
        # Search drug labels for contraindications
        for condition in conditions:
            params = {
                "search": f'contraindications:"{condition}" AND openfda.generic_name:"{drug}"',
                "limit": 5
            }
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(self.drug_label_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for result in data.get("results", []):
                                contraindications_text = result.get("contraindications", [])
                                
                                for contra_text in contraindications_text:
                                    if isinstance(contra_text, str) and condition.lower() in contra_text.lower():
                                        contraindications.append({
                                            "condition": condition,
                                            "drug": drug,
                                            "description": contra_text[:200] + "...",
                                            "severity": "major"
                                        })
                                        
                except Exception as e:
                    self.logger.error(f"Contraindication search error: {str(e)}")
                    
        return contraindications
        
    async def _get_drug_warnings(self, drug: str) -> List[Dict[str, Any]]:
        """Get general warnings for a drug"""
        warnings = []
        
        params = {
            "search": f'openfda.generic_name:"{drug}"',
            "limit": 3
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.drug_label_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for result in data.get("results", []):
                            warnings_text = result.get("warnings", [])
                            boxed_warning = result.get("boxed_warning", [])
                            
                            # Add boxed warnings (most serious)
                            for warning in boxed_warning:
                                if isinstance(warning, str):
                                    warnings.append({
                                        "type": "boxed_warning",
                                        "description": warning[:300] + "...",
                                        "severity": "major"
                                    })
                                    
                            # Add general warnings
                            for warning in warnings_text[:3]:  # Limit to 3
                                if isinstance(warning, str):
                                    warnings.append({
                                        "type": "warning",
                                        "description": warning[:200] + "...",
                                        "severity": "moderate"
                                    })
                                    
            except Exception as e:
                self.logger.error(f"Drug warnings search error: {str(e)}")
                
        return warnings
        
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate clinical recommendations"""
        recommendations = []
        
        # Check for major interactions
        major_interactions = [i for i in results["interactions"] if i.get("severity") == "major"]
        if major_interactions:
            recommendations.append("Consider alternative medications due to major drug interactions")
            
        # Check for contraindications
        if results["contraindications"]:
            recommendations.append("Review patient conditions for contraindications")
            
        # Check for multiple interactions
        if len(results["interactions"]) > 2:
            recommendations.append("Multiple drug interactions detected - consider medication review")
            
        # General monitoring
        if results["interactions"] or results["warnings"]:
            recommendations.append("Monitor patient for adverse effects")
            
        # Default recommendation
        if not recommendations:
            recommendations.append("Continue monitoring for potential drug interactions")
            
        return recommendations
        
    def _calculate_severity_summary(self, results: Dict[str, Any]) -> Dict[str, int]:
        """Calculate summary of severity levels"""
        summary = {"major": 0, "moderate": 0, "minor": 0}
        
        # Count interaction severities
        for interaction in results["interactions"]:
            severity = interaction.get("severity", "unknown")
            if severity in summary:
                summary[severity] += 1
                
        # Count contraindication severities
        for contra in results["contraindications"]:
            severity = contra.get("severity", "unknown")
            if severity in summary:
                summary[severity] += 1
                
        # Count warning severities
        for warning in results["warnings"]:
            severity = warning.get("severity", "unknown")
            if severity in summary:
                summary[severity] += 1
                
        return summary