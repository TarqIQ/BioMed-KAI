from __future__ import annotations
from typing import Dict, Any, Tuple, List
from ..agents.registry import all_agents
from .policies import rule_based_policy, classifier_policy, BanditState
from .classifier import load_router, score_with_router

_BANDIT: BanditState | None = None
_CLASSIFIER = None

def route(query_en: str, cfg) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
    global _CLASSIFIER
    agents = all_agents()
    candidates = []
    for name, a in agents.items():
        candidates.append({
            "name": name,
            "requires_kg": getattr(a, "requires_kg", False),
            "modalities": getattr(a, "modalities", ["text"]),
            "max_hops": getattr(a, "max_hops", 1),
            "cost_class": getattr(a, "cost_class", "med")
        })
    candidates = [c for c in candidates if "text" in c["modalities"]]
    policy = getattr(cfg.routing, "policy", "rule")
    if policy == "rule":
        choice, scores = rule_based_policy(query_en, candidates)
    elif policy == "classifier":
        if _CLASSIFIER is None:
            _CLASSIFIER = load_router(cfg)
        logits = score_with_router(_CLASSIFIER, query_en, [c["name"] for c in candidates])
        choice, scores = classifier_policy(query_en, candidates, logits)
    elif policy == "bandit":
        global _BANDIT
        if _BANDIT is None:
            _BANDIT = BanditState([c["name"] for c in candidates])
        choice = _BANDIT.ucb1()
        scores = {c["name"]: 0.0 for c in candidates}
    else:
        choice, scores = rule_based_policy(query_en, candidates)
    info = {"candidates": candidates, "policy": policy}
    return choice, scores, info
