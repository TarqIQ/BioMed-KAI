from typing import Optional
from .interfaces import LLM
# try:
#     from llama_cpp import Llama
# except Exception:
#     Llama = None


try:
    import llama_cpp
except:
    llama_cpp = None

try:
    import llama_cpp_cuda
except:
    llama_cpp_cuda = None

try:
    import llama_cpp_cuda_tensorcores
except:
    llama_cpp_cuda_tensorcores = None


def llama_cpp_lib():
    """Select the best available llama-cpp library"""
    if llama_cpp_cuda_tensorcores is not None:
        return llama_cpp_cuda_tensorcores
    elif llama_cpp_cuda is not None:
        return llama_cpp_cuda
    else:
        return llama_cpp

Llama = llama_cpp_lib().Llama

class LlamaCppLLM(LLM):
    def __init__(self, model_path: str, n_ctx: int = 8192, n_threads: int = 8,
                 n_gpu_layers: int = 0, temperature: float = 0.2, top_p: float = 0.9,
                 repeat_penalty: float = 1.1, chat_format: Optional[str] = None):
        if Llama is None:
            raise RuntimeError("llama-cpp-python not installed")
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                         n_gpu_layers=n_gpu_layers, chat_format=chat_format)
        self.temperature = temperature; self.top_p = top_p; self.repeat_penalty = repeat_penalty; self.chat_format = chat_format
    def generate(self, prompt: str, **kw) -> str:
        if self.chat_format:
            out = self.llm.create_chat_completion(messages=[{"role":"user","content":prompt}],
                                                  temperature=self.temperature, top_p=self.top_p,
                                                  repeat_penalty=self.repeat_penalty)
            return out["choices"][0]["message"]["content"]
        out = self.llm(prompt, temperature=self.temperature, top_p=self.top_p,
                       repeat_penalty=self.repeat_penalty, max_tokens=kw.get("max_tokens",1024))
        return out["choices"][0]["text"]
