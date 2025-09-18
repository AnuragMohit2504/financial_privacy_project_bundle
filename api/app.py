from fastapi import FastAPI
from pydantic import BaseModel
from wrapper.llm_adapter import MockAdapter
from wrapper.wrapper import LLMWrapper
app = FastAPI()
adapter = MockAdapter()
wrapper = LLMWrapper(adapter)
class QueryRequest(BaseModel):
    user_id: str
    query: str
    use_rag: bool = True
@app.post("/query")
def query_endpoint(req: QueryRequest):
    resp = wrapper.generate(req.query, user_id=req.user_id, use_rag=req.use_rag)
    return {"response": resp}
