# api/app.py
import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# Ensure project root in sys.path
# -----------------------------
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# Import wrappers/adapters
from wrapper.llm_adapter import MockAdapter
from wrapper.wrapper import LLMWrapper
from backend.ingestion import FinanceVectorStore

# -----------------------------
# Init FastAPI
# -----------------------------
app = FastAPI(title="Finance Privacy App", version="1.0")

# -----------------------------
# Initialize wrapper + vector store
# -----------------------------
adapter = MockAdapter()
vector_store = FinanceVectorStore()
vector_store.load()
wrapper = LLMWrapper(adapter, vector_store=vector_store)

# -----------------------------
# Models
# -----------------------------
class QueryRequest(BaseModel):
    user_id: str
    query: str
    use_rag: bool = True
    top_k: int = 3

# -----------------------------
# Routes
# -----------------------------
@app.post("/query")
def query_endpoint(req: QueryRequest):
    resp = wrapper.generate(
        req.query,
        user_id=req.user_id,
        use_rag=req.use_rag,
        top_k=req.top_k
    )
    return {"response": resp}

@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Run standalone
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)  # ✅ runs on port 8002
