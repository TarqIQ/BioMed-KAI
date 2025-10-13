from typing import Dict, Any
from .kg_store import KGStore
from ..providers.interfaces import LLM

def care_rag(query_en: str, kg: KGStore, llm: LLM, top_k: int = 25) -> Dict[str, Any]:
    nodes = kg.query(query_en, top_k=top_k)
    context = "\n".join(n.id for n in nodes[:5])
    answer = llm.generate(f"CONTEXT:\n{context}\nQUESTION:{query_en}")
    return {"answer": answer, "nodes": [n.__dict__ for n in nodes]}
