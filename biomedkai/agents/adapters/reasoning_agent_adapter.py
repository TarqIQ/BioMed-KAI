from __future__ import annotations
from typing import Dict, Any, List

try:
    from legacy.reasoning_agent import LegacyReasoningAgent
except Exception:
    LegacyReasoningAgent = None

try:
    from legacy.reasoning import run as legacy_reasoning_run
except Exception:
    legacy_reasoning_run = None

class ReasoningAgent:
    name = "reasoning_agent"
    requires_kg = False
    modalities: List[str] = ["text"]
    max_hops = 0
    cost_class = "high"

    def __init__(self, llm, cfg):
        self.llm = llm; self.cfg = cfg
        self._legacy_obj = None
        if LegacyReasoningAgent is not None:
            self._legacy_obj = LegacyReasoningAgent(llm=llm, cfg=cfg)

    def run(self, query_en: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if self._legacy_obj is not None:
            return self._legacy_obj.run(query_en)
        if legacy_reasoning_run is not None:
            return legacy_reasoning_run(query_en, self.llm, self.cfg)
        answer = self.llm.generate(f"Answer as a biomedical reasoning agent:\n{query_en}")
        return {"answer": answer, "nodes": []}
