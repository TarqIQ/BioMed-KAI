from typing import Dict, List, Any, Optional

from ...tools.base_tool import BaseTool
from ...tools.neo4j_tools.knowledge_graph_search import KnowledgeGraphSearchTool
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Try to import the function, provide fallback if not available
try:
    from model.dr import detect_entities_from_index
except ImportError:
    # Fallback implementation
    def detect_entities_from_index(text: str) -> List[Dict[str, Any]]:
        # Simple fallback for testing
        entities = []
        medical_terms = ["fever", "cough", "pain", "diabetes", "hypertension", "cancer"]
        for term in medical_terms:
            if term in text.lower():
                entities.append({
                    "text": term,
                    "label": "CONDITION",
                    "score": 0.8,
                    "o_label": "Disease"
                })
        return entities

class EntityExtractorTool(BaseTool):
    """
    Extract medical entities using your existing Neo4j index
    """
    kg_tool : KnowledgeGraphSearchTool = None
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="entity_extractor",
            description="Extract medical entities from text using Neo4j index",
            config=config
        )
        
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Extract entities from text"""
        text = kwargs.get("text", "")
        
        if not text:
            return {"entities": []}
            
        # Use your existing function
        if not self.kg_tool:
            self.kg_tool = KnowledgeGraphSearchTool()
        entities = detect_entities_from_index(text)
        
        return {
            "entities": entities,
            "count": len(entities)
        }
        
    def validate_params(self, **kwargs) -> bool:
        """Validate input parameters"""
        return bool(kwargs.get("text"))