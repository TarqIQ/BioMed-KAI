from __future__ import annotations
from typing import Dict
from .interfaces import Agent

_REGISTRY: Dict[str, Agent] = {}

def register(agent: Agent) -> None:
    _REGISTRY[agent.name] = agent

def get(name: str):
    return _REGISTRY.get(name)

def all_agents():
    return dict(_REGISTRY)
