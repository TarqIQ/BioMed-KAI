import json, pathlib
from typing import List, Optional
from ..orchestrator import run_query
from .metrics import jaccard

def run_multilingual_pilot(langs: List[str], prompts_dir: str, out_dir: str, cfg_path: Optional[str] = None):
    out_root = pathlib.Path(out_dir); out_root.mkdir(parents=True, exist_ok=True)
    prompts_files = ["tbqa_sample_100.jsonl","general_biomed_25.jsonl","medhalt_like_25.jsonl"]
    english_baseline = {}

    # 1) English baseline (using the English versions of prompts)
    for pf in prompts_files:
        pfile_en = pathlib.Path(prompts_dir)/pf
        for line in pfile_en.read_text(encoding="utf-8").splitlines():
            ex = json.loads(line)
            qid = ex["id"]
            res = run_query(ex["prompt_en"], language="en", cfg_path=cfg_path)
            english_baseline[qid] = {"answer": res["answer"], "nodes": [n["id"] for n in res["nodes"]]}

    # 2) Multilingual runs
    summary = []
    for lang in langs:
        per_lang = []
        for pf in prompts_files:
            pfile = pathlib.Path(prompts_dir)/pf
            for line in pfile.read_text(encoding="utf-8").splitlines():
                ex = json.loads(line)
                qid = ex["id"]
                prompt = ex.get(f"prompt_{lang}", ex.get("prompt_native", ex["prompt_en"]))
                res = run_query(prompt, language=lang, cfg_path=cfg_path)
                nodes_mt = [n["id"] for n in res["nodes"]]
                nodes_en = english_baseline[qid]["nodes"]
                per_lang.append({
                    "id": qid, "lang": lang,
                    "jaccard_nodes@25": jaccard(nodes_en, nodes_mt)
                })
        # save per-language results
        outp = out_root / f"{lang}.jsonl"
        outp.write_text("\n".join(json.dumps(r) for r in per_lang), encoding="utf-8")
        # simple summary
        j = [r["jaccard_nodes@25"] for r in per_lang]
        summary.append({"lang": lang, "n": len(j), "jaccard@25_mean": float(sum(j)/len(j)) if j else 0.0})

    (out_root/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
