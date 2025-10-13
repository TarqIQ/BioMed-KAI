from .interfaces import LLM

class OpenAILLM(LLM):
    def __init__(self, model: str, max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
    def generate(self, prompt: str, **kw) -> str:
        # Placeholder – integrate real SDK here.
        return f"[MOCK {self.model}] " + prompt[:200]
