from fastapi import FastAPI
from pydantic import BaseModel
from ..orchestrator import run_query

app = FastAPI(title="BioMed-KAI API")

class Query(BaseModel):
    text: str
    language: str = "en"

@app.post("/query")
def query(q: Query):
    return run_query(query=q.text, language=q.language)
