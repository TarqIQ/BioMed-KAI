from __future__ import annotations
from typing import Dict, Any, List
from .interfaces import Agent
from ..retrieval.pipeline import care_rag
from ..retrieval.kg_store import KGStore
from ..providers.interfaces import LLM

class EvidenceAgentImpl:
    name = "evidence_agent"
    requires_kg = True
    modalities: List[str] = ["text"]
    max_hops = 2
    cost_class = "med"
    def __init__(self, kg: KGStore, llm: LLM, top_k: int = 25):
        self.kg = kg; self.llm = llm; self.top_k = top_k
    def run(self, query_en: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return care_rag(query_en, self.kg, self.llm, top_k=self.top_k)
