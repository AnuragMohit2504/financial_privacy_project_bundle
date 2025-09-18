from privacy.masker import mask_text
from wrapper.llm_adapter import MockAdapter
from wrapper.wrapper import LLMWrapper
print("Demo: masking sample...")
s = "Please show account 123456789012 and PAN ABCDE1234F for Mr. Rahul Sharma"
print("RAW:", s)
print("MASKED:", mask_text(s))
adapter = MockAdapter()
w = LLMWrapper(adapter)
print("\\nWrapper output:\\n", w.generate(s, user_id="demo"))