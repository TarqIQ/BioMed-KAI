from .interfaces import Translator
class NLLBTranslator(Translator):
    def translate(self, text: str, src: str, tgt: str) -> str:
        return text  # stub for offline runs
