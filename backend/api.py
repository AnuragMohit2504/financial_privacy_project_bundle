# backend/api.py
import os
import sys
import hashlib
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn

# -----------------------------
# Init FastAPI
# -----------------------------
app = FastAPI(title="Finance Privacy Backend", version="1.0")

# -----------------------------
# Ensure project root in sys.path
# -----------------------------
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# Local imports
from privacy.masker import mask_text
from wrapper.wrapper import LLMWrapper
from wrapper.llm_adapter import MockAdapter
from backend.ingestion import FinanceVectorStore, load_statements, DATA_DIR, INDEX_FILE, STORE_FILE

# -----------------------------
# Initialize wrapper + vector store
# -----------------------------
adapter = MockAdapter()
vector_store: Optional[FinanceVectorStore] = None
try:
    vector_store = FinanceVectorStore()
    vector_store.load()
    print("✅ Vector store loaded successfully.")
except Exception as e:
    print("⚠️ Could not load vector store, continuing without RAG:", e)

wrapper = LLMWrapper(adapter, vector_store=vector_store)

# -----------------------------
# Models
# -----------------------------
class ChatRequest(BaseModel):
    user_id: str
    message: str
    use_rag: Optional[bool] = False
    top_k: Optional[int] = 3

# -----------------------------
# Helpers
# -----------------------------
def compute_prompt_hash(masked_prompt: str) -> str:
    return hashlib.sha256(masked_prompt.encode("utf-8")).hexdigest()

def ensure_safe_text(text: str) -> bool:
    import re
    if not text:
        return True
    if re.search(r"\b\d{10,}\b", text):  # 10+ digit sequences
        return False
    if re.search(r"\b[A-Z]{5}\d{4}[A-Z]\b", text):  # PAN pattern
        return False
    return True

# -----------------------------
# Routes
# -----------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    masked_prompt = mask_text(req.message)
    prompt_hash = compute_prompt_hash(masked_prompt)

    try:
        raw_response = wrapper.generate(
            req.message,
            user_id=req.user_id,
            use_rag=req.use_rag,
            top_k=req.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    if not ensure_safe_text(raw_response):
        safe = mask_text(raw_response)
        return {
            "summary": "Response redacted for safety.",
            "answer": safe,
            "actionable": ["Retry with masked input"],
            "confidence": "low",
            "sources": [],
            "prompt_hash": prompt_hash,
        }

    summary = raw_response.splitlines()[0] if raw_response else "Response generated"
    return {
        "summary": summary[:140],
        "answer": raw_response,
        "actionable": ["Download masked report", "Ask a follow-up question"],
        "confidence": "medium",
        "sources": vector_store.texts[:5] if vector_store else [],
        "prompt_hash": prompt_hash,
    }

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF, reset index, and ingest only this file."""
    try:
        # Ensure data folder exists
        os.makedirs(DATA_DIR, exist_ok=True)
        save_path = os.path.join(DATA_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())

        # Remove old index + store (reset vector store each time)
        if os.path.exists(INDEX_FILE):
            os.remove(INDEX_FILE)
        if os.path.exists(STORE_FILE):
            os.remove(STORE_FILE)

        # Parse new file
        docs = load_statements(DATA_DIR, file=file.filename)
        if not docs:
            return {"status": "no_docs", "file": file.filename}

        # Build a fresh vector store
        global vector_store
        vector_store = FinanceVectorStore()
        vector_store.add_documents(docs)

        # Update wrapper with fresh vector store
        global wrapper
        wrapper = LLMWrapper(adapter, vector_store=vector_store)

        return {"status": "ok", "file": file.filename, "docs": len(docs)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

# -----------------------------
# Run standalone
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)
