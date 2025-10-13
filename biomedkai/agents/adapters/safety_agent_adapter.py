from __future__ import annotations
from typing import Dict, Any, List

try:
    from legacy.safety import check as legacy_safety_check
except Exception:
    legacy_safety_check = None

class SafetyAgent:
    name = "safety_agent"
    requires_kg = False
    modalities: List[str] = ["text"]
    max_hops = 0
    cost_class = "low"

    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, query_en: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if legacy_safety_check is not None:
            return legacy_safety_check(query_en, context)
        return {"safe": True, "reasons": []}
