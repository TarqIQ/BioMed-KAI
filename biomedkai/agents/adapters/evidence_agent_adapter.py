from __future__ import annotations
from typing import Dict, Any, List

try:
    from legacy.evidence import run as legacy_evidence_run  # CHANGE this import to your path
except Exception:
    legacy_evidence_run = None

try:
    from legacy.evidence_agent import LegacyEvidenceAgent  # CHANGE this import to your path
except Exception:
    LegacyEvidenceAgent = None

class EvidenceAgent:
    name = "evidence_agent"
    requires_kg = True
    modalities: List[str] = ["text"]
    max_hops = 2
    cost_class = "med"

    def __init__(self, kg, llm, cfg):
        self.kg = kg; self.llm = llm; self.cfg = cfg
        self._legacy_obj = None
        if LegacyEvidenceAgent is not None:
            self._legacy_obj = LegacyEvidenceAgent(kg=kg, llm=llm, cfg=cfg)

    def run(self, query_en: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if self._legacy_obj is not None:
            return self._legacy_obj.run(query_en)
        if legacy_evidence_run is not None:
            return legacy_evidence_run(query_en, self.kg, self.llm, self.cfg)
        from ...retrieval.pipeline import care_rag
        return care_rag(query_en, self.kg, self.llm, top_k=self.cfg.retrieval.top_k)
