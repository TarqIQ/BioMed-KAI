from dataclasses import dataclass
from typing import List

@dataclass
class KGNode:
    id: str
    label: str
    score: float

class KGStore:
    def __init__(self, uri: str, user: str, password: str):
        self.uri, self.user, self.password = uri, user, password
    def query(self, text: str, top_k: int = 25) -> List[KGNode]:
        return [KGNode(id=f"N{i}", label="Entity", score=1.0/(i+1)) for i in range(top_k)]
