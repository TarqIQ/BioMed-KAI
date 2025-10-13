from __future__ import annotations
from typing import Dict, Any
def load_router(cfg) -> Any:
    typ = getattr(cfg.routing, "classifier_type", "none")
    path = getattr(cfg.routing, "classifier_path", "")
    if typ == "none" or not path:
        return None
    if typ == "sklearn":
        import joblib
        return joblib.load(path)
    if typ == "pytorch":
        import torch
        model = torch.load(path, map_location="cpu")
        model.eval()
        return model
    if typ == "onnx":
        import onnxruntime as ort
        return ort.InferenceSession(path)
    if typ == "python":
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("router_module", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["router_module"] = mod
        spec.loader.exec_module(mod)  # type: ignore
        return mod
    raise ValueError(f"Unsupported classifier_type: {typ}")

def score_with_router(router, query_en: str, candidate_names: list[str]) -> Dict[str, float]:
    if router is None:
        return {}
    if hasattr(router, "predict_proba"):
        X = [query_en]
        probs = router.predict_proba(X)[0]
        classes = list(getattr(router, "classes_", candidate_names))
        return {str(c): float(p) for c, p in zip(classes, probs) if str(c) in candidate_names}
    if hasattr(router, "__call__"):
        out = router(query_en)
        return {k: float(v) for k, v in out.items() if k in candidate_names}
    try:
        import onnxruntime as ort
        if isinstance(router, ort.InferenceSession):
            return {}
    except Exception:
        pass
    return {}
