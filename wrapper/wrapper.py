from privacy.masker import mask_text
class LLMWrapper:
    def __init__(self, adapter, vector_store=None):
        self.adapter = adapter
        self.vector_store = vector_store
    def generate(self, prompt: str, user_id: str = 'anon', use_rag: bool = True, top_k: int = 3, **kwargs):
        masked_prompt = mask_text(prompt)
        contexts = []
        if use_rag and self.vector_store is not None:
            contexts = self.vector_store.retrieve(masked_prompt, top_k=top_k)
        context_block = "\n\nContext:\n" + "\n---\n".join(contexts) if contexts else ""
        final_prompt = f"You are a privacy-aware financial assistant. Do not expose raw PII.\n\nUser Query (masked):\n{masked_prompt}{context_block}\n\nAnswer concisely:"
        raw_response = self.adapter.generate(final_prompt, **kwargs)
        safe_response = mask_text(raw_response)
        return safe_response
