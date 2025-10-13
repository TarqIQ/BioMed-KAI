from __future__ import annotations
from typing import Protocol, Dict, Any, List

class Agent(Protocol):
    name: str
    requires_kg: bool
    modalities: List[str]
    max_hops: int
    cost_class: str
    def run(self, query_en: str, context: Dict[str, Any]) -> Dict[str, Any]: ...
