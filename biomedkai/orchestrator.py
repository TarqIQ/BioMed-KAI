from __future__ import annotations
from typing import Optional
from .config import load_config
from .providers.llamacpp_llm import LlamaCppLLM
from .providers.hf_translator import NLLBTranslator
from .retrieval.kg_store import KGStore
from .agents.registry import register, get
from .agents.adapters.evidence_agent_adapter import EvidenceAgent
from .agents.adapters.reasoning_agent_adapter import ReasoningAgent
from .agents.adapters.safety_agent_adapter import SafetyAgent
from .routing.router import route

def _make_llm(cfg):
    if cfg.llm.provider == "llamacpp":
        return LlamaCppLLM(
            model_path=cfg.llm.model_path,
            n_ctx=cfg.llm.n_ctx,
            n_threads=cfg.llm.n_threads,
            n_gpu_layers=cfg.llm.n_gpu_layers,
            temperature=cfg.llm.temperature,
            top_p=cfg.llm.top_p,
            repeat_penalty=cfg.llm.repeat_penalty,
            chat_format=cfg.llm.chat_format
        )
    raise ValueError(f"Unsupported llm.provider: {cfg.llm.provider}")

def run_query(query: str, language: str = "en", cfg_path: Optional[str] = None, force_agent: Optional[str] = None):
    cfg = load_config(cfg_path)
    llm = _make_llm(cfg)
    tr = NLLBTranslator()
    q_en = tr.translate(query, src=language, tgt="en") if language != "en" else query
    kg = KGStore(cfg.kg.uri, cfg.kg.user, cfg.kg.password)

    register(EvidenceAgent(kg=kg, llm=llm, cfg=cfg))
    register(ReasoningAgent(llm=llm, cfg=cfg))
    register(SafetyAgent(cfg=cfg))

    if force_agent:
        agent = get(force_agent)
        if agent is None:
            raise ValueError(f"Unknown agent: {force_agent}")
        out = agent.run(q_en, context={"config": cfg, "forced": True})
        if language != "en":
            out["answer_mt_out"] = tr.translate(out["answer"], src="en", tgt=language)
        out["agent"] = force_agent
        out["routing_scores"] = {}
        out["routing_info"] = {"candidates": [force_agent], "policy": "forced"}
        return out

    agent_name, scores, info = route(q_en, cfg)
    agent = get(agent_name)
    out = agent.run(q_en, context={"config": cfg, "scores": scores, "routing_info": info})
    if language != "en":
        out["answer_mt_out"] = tr.translate(out["answer"], src="en", tgt=language)
    out["agent"] = agent_name
    out["routing_scores"] = scores
    out["routing_info"] = info
    return out
