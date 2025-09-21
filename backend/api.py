import os
import sys
import hashlib
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
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

# Import local modules
from privacy.masker import mask_text
from wrapper.wrapper import LLMWrapper
from wrapper.llm_adapter import MockAdapter  # swap later with OpenAIAdapter/HFAdapter

# -----------------------------
# Initialize wrapper
# -----------------------------
adapter = MockAdapter()
wrapper = LLMWrapper(adapter)

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
    """Generate SHA256 hash of masked prompt for auditing."""
    return hashlib.sha256(masked_prompt.encode("utf-8")).hexdigest()

def ensure_safe_text(text: str) -> bool:
    """Prevent raw PII leakage: long digit sequences, PAN patterns."""
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
    # Mask incoming text
    masked_prompt = mask_text(req.message)
    prompt_hash = compute_prompt_hash(masked_prompt)

    # Call LLM wrapper
    try:
        raw_response = wrapper.generate(
            req.message,
            user_id=req.user_id,
            use_rag=req.use_rag,
            top_k=req.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # Safety check
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

    # Normal structured response
    summary = raw_response.splitlines()[0] if raw_response else "Response generated"
    return {
        "summary": summary[:140],
        "answer": raw_response,
        "actionable": ["Download masked report", "Ask a follow-up question"],
        "confidence": "medium",
        "sources": getattr(wrapper.vector_store, "last_retrieved_ids", []) if wrapper.vector_store else [],
        "prompt_hash": prompt_hash,
    }

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

# -----------------------------
# Run standalone
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
