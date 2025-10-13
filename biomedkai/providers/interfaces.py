from typing import Protocol, List

class LLM(Protocol):
    def generate(self, prompt: str, **kw) -> str: ...

class Embedder(Protocol):
    def encode(self, texts: List[str]) -> list[list[float]]: ...

class Translator(Protocol):
    def translate(self, text: str, src: str, tgt: str) -> str: ...
