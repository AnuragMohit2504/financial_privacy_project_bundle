class MockAdapter:
    def __init__(self, name='mock'):
        self.name = name
    def generate(self, prompt: str, max_tokens: int = 256, **kwargs) -> str:
        summary = "\n\n[MockModel] Generated safe summary based on input (development only)."
        safe = prompt if len(prompt) < 2000 else prompt[:2000] + "..."
        return safe + summary
