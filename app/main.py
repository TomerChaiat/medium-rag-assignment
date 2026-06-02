from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import CHUNK_SIZE, OVERLAP_RATIO, TOP_K
from app.rag import answer_question

app = FastAPI(title="Medium Article RAG Assistant")


class PromptRequest(BaseModel):
    question: str


@app.post("/api/prompt")
def prompt(request: PromptRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        return answer_question(question)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/api/stats")
def stats():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K,
    }
