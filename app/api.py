# File to define fastapi endpoints

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag_pipeline import build_rag_chain
from app.data_loader import load_and_split_documents
from app.vector_store import create_vector_store

app = FastAPI(title="Resume RAG API")

# Define input/output schema
class QuestionRequest(BaseModel):
    question: str

class QAResponse(BaseModel):
    answer: str
    confidence_level: str

# Load resume & setup chain on startup
chunks = load_and_split_documents("docs/ManishKumawatResume.pdf")
vectorstore = create_vector_store(chunks)
rag_chain = build_rag_chain(vectorstore)

@app.post("/ask", response_model=QAResponse)
def ask_question(payload: QuestionRequest):
    try:
        result = rag_chain.invoke({"question": payload.question})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
