import os
from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.deployments import RuntimeContext

from ai_service import gen_ai_service  # uses your PediatricAgentService inside

# ---- 1. IBM Cloud auth ----
load_dotenv()
API_KEY = os.getenv("IBM_CLOUD_API_KEY") or os.getenv("WATSONX_APIKEY")
SPACE_ID = os.getenv("SPACE_ID") or os.getenv("WATSONX_SPACE_ID")

if not API_KEY:
    raise RuntimeError("Please set IBM_CLOUD_API_KEY in your environment.")
if not SPACE_ID:
    raise RuntimeError("Please set SPACE_ID in your environment.")

credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key=API_KEY,
)
client = APIClient(credentials)
client.set.default_space(SPACE_ID)

# ---- 2. Build fake chat payload ----

messages = [
    {
        "role": "user",
        "content": "My child has a fever of 37.8C and is vomiting twice. Is this okay?",
    }
]

# ---- 3. Create RuntimeContext and local function ----

# First context: used to construct the service (provides token)
service_context = RuntimeContext(api_client=client)

# gen_ai_service returns (generate, generate_stream)
streaming = False
index = 1 if streaming else 0
local_function = gen_ai_service(service_context, params={"space_id": SPACE_ID})[index]

# Second context: simulates the runtime request with payload
request_context = RuntimeContext(api_client=client, request_payload_json={"messages": messages})

# ---- 4. Call the local function ----

response = local_function(request_context)
print(response)
