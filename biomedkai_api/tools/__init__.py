# from typing import Dict, Any
# from ..tools.medical_tools.medical_calculator import  MedicalCalculator
# from ..tools.search_tools.drug_interaction_checker import DrugInteractionChecker
# from ..tools.medical_tools.symptom_analyzer import SymptomAnalyzer
# # from ..tools.medical_tools.lab_interpreter import LabInterpreter
# from ..tools.search_tools.web_search import WebSearchTool
# from ..tools.search_tools.pubmed_search import PubMedSearchTool
# from ..tools.neo4j_tools.knowledge_graph_search import KnowledgeGraphSearchTool
# from ..tools.neo4j_tools.entity_extractor import EntityExtractorTool

# def create_tool_registry() -> Dict[str, Any]:
#     """
#     Create and return a registry of all available medical tools.
    
#     Returns:
#         Dict[str, Any]: Dictionary mapping tool names to tool instances
#     """
#     tools = {
#         # Medical tools
#         "symptom_analyzer": SymptomAnalyzer(),
#         "symptom_extractor": SymptomAnalyzer(),  # Alias
        
#         # Search tools
#         "web_search": WebSearchTool(),
#         "pubmed_search": PubMedSearchTool(),
        
#         # Neo4j tools
#         "knowledge_graph_search": KnowledgeGraphSearchTool(),
#         "entity_extractor": EntityExtractorTool(),
#     }
    
#     # Medical calculation tools
#     tools["medical_calculator"] = MedicalCalculator()
    
#     # Drug interaction checker
#     tools["drug_interaction_checker"] = DrugInteractionChecker()
    
#     # Symptom analysis tool
#     tools["symptom_analyzer"] = SymptomAnalyzer()
#     tools["entity_extractor"].kg_tool = tools["knowledge_graph_search"]
    
#     # Lab result interpreter
#     # tools["lab_interpreter"] = LabInterpreter()
    
#     return tools


from typing import Dict, Any
import asyncio

from ..tools.medical_tools.symptom_analyzer import SymptomAnalyzer
from ..tools.search_tools.web_search import WebSearchTool
from ..tools.search_tools.pubmed_search import PubMedSearchTool
from ..tools.neo4j_tools.knowledge_graph_search import KnowledgeGraphSearchTool
from ..tools.neo4j_tools.entity_extractor import EntityExtractorTool


# Simple query classifier tool
async def query_classifier(text: str) -> Dict[str, Any]:
    """Simple query classification"""
    query_lower = text.lower()
    
    categories = []
    if any(word in query_lower for word in ["symptom", "pain", "fever", "cough"]):
        categories.append("symptoms")
    if any(word in query_lower for word in ["diagnose", "diagnosis", "what is wrong"]):
        categories.append("diagnosis")
    if any(word in query_lower for word in ["treat", "medication", "therapy"]):
        categories.append("treatment")
    if any(word in query_lower for word in ["research", "study", "evidence"]):
        categories.append("research")
        
    return {
        "categories": categories,
        "primary_category": categories[0] if categories else "general"
    }


def create_tool_registry() -> Dict[str, Any]:
    """Create and return tool registry"""
    
    tools = {
        # Medical tools
        "symptom_analyzer": SymptomAnalyzer(),
        "symptom_extractor": SymptomAnalyzer(),  # Alias
        
        # Search tools
        "web_search": WebSearchTool(),
        "pubmed_search": PubMedSearchTool(),
        
        # Neo4j tools
        "knowledge_graph_search": KnowledgeGraphSearchTool(),
        "entity_extractor": EntityExtractorTool(),
        
        # Simple tools as functions
        "query_classifier": query_classifier,
        
        # Placeholder tools that return mock data
        "guideline_checker": lambda **kwargs: {"guidelines": "Standard treatment guidelines apply"},
        "drug_database": lambda **kwargs: {"drug_info": "Drug information placeholder"},
        "drug_interaction_checker": lambda **kwargs: {"interactions": []},
        "allergy_checker": lambda **kwargs: {"allergy_risk": False},
        "lab_interpreter": lambda **kwargs: {"interpretation": "Lab results within normal range"},
        "clinical_trials_search": lambda **kwargs: {"trials": []},
        "safety_validator": lambda **kwargs: {"safe": True},
        "vital_signs_analyzer": lambda **kwargs: {"status": "stable"},
    }
    
    return tools