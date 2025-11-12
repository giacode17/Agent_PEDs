from ai_service import gen_ai_service
from ibm_watsonx_ai.foundation_models import ModelInferenceRuntimeContext

fake_context = ModelInferenceRuntimeContext(api_client=None)
fake_context.generate_token = lambda: "FAKE_TOKEN"
fake_context.get_json = lambda: {"messages": [{"role": "user", "content": "My child has a fever of 39C"}]}

generate, _ = gen_ai_service(fake_context)
response = generate(fake_context)
print(response)
