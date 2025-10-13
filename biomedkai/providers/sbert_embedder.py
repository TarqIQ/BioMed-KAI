from .interfaces import Embedder
class SBertEmbedder(Embedder):
    def __init__(self, model: str):
        self.model = model
    def encode(self, texts):
        # Return fixed-size vectors for scaffolding
        return [[0.0]*384 for _ in texts]
