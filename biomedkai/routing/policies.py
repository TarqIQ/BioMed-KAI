from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math

def rule_based_policy(query_en: str, candidates: List[Dict[str, Any]]) -> Tuple[str, Dict[str, float]]:
    q = query_en.lower()
    scores: Dict[str, float] = {}
    for c in candidates:
        s = 0.0
        if any(k in q for k in ["trial","guideline","evidence","study","odds ratio","hazard ratio"]):
            s += 5.0 if c["name"].startswith("evidence") else 0.0
        if any(k in q for k in ["why","explain","derive","prove","mechanism","pathway","reason"]):
            s += 5.0 if c["name"].startswith("reasoning") else 0.0
        if any(k in q for k in ["is it safe","contraindicated","pregnant","lactation","children"]):
            s += 5.0 if c["name"].startswith("safety") else 0.0
        if s == 0.0:
            s += 0.2 if c["cost_class"] == "low" else 0.1 if c["cost_class"] == "med" else 0.0
        scores[c["name"]] = s
    choice = max(scores, key=scores.get)
    return choice, scores

def classifier_policy(query_en: str, candidates: List[Dict[str, Any]], logits: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
    if not logits:
        return rule_based_policy(query_en, candidates)
    cand_names = {c["name"] for c in candidates}
    scores = {k: v for k, v in logits.items() if k in cand_names}
    if not scores:
        return rule_based_policy(query_en, candidates)
    choice = max(scores, key=scores.get)
    return choice, scores

class BanditState:
    def __init__(self, names: List[str]):
        self.names = names
        self.counts = {n: 1 for n in names}
        self.rewards = {n: 0.0 for n in names}
    def ucb1(self) -> str:
        t = sum(self.counts.values())
        def ucb(n):
            return (self.rewards[n]/self.counts[n]) + math.sqrt(2.0*math.log(t)/self.counts[n])
        return max(self.names, key=ucb)
    def update(self, name: str, reward: float):
        self.counts[name] += 1
        self.rewards[name] += reward
