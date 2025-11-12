import json
import os
from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient, Credentials
from ai_service import gen_ai_service

load_dotenv()

API_KEY = os.getenv("IBM_CLOUD_API_KEY") or os.getenv("WATSONX_APIKEY")
SPACE_ID = os.getenv("SPACE_ID") or os.getenv("WATSONX_SPACE_ID")
URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

if not API_KEY:
    raise RuntimeError("Set IBM_CLOUD_API_KEY or WATSONX_APIKEY env var.")
if not SPACE_ID:
    raise RuntimeError("Set SPACE_ID or WATSONX_SPACE_ID env var.")

print("model success")

credentials = Credentials(url=URL, api_key=API_KEY)
client = APIClient(credentials)
client.set.default_space(SPACE_ID)

# ---- 1. Load schemas as dicts (IMPORTANT) ----
with open("schema/request.json", "r") as f:
    request_schema = json.load(f)

with open("schema/response.json", "r") as f:
    response_schema = json.load(f)

# ---- 2. Get software spec id ----
software_spec_id = client.software_specifications.get_id_by_name("runtime-24.1-py3.11")

# ---- 3. Store AI service ----
ai_service_metadata = {
    client.repository.AIServiceMetaNames.NAME: "Peds_Post_Discharge_Agent",
    client.repository.AIServiceMetaNames.DESCRIPTION: "Pediatric post-discharge assistant",
    client.repository.AIServiceMetaNames.SOFTWARE_SPEC_ID: software_spec_id,
    client.repository.AIServiceMetaNames.CUSTOM: {},
    client.repository.AIServiceMetaNames.REQUEST_DOCUMENTATION: request_schema,   # ✅ dict
    client.repository.AIServiceMetaNames.RESPONSE_DOCUMENTATION: response_schema, # ✅ dict
    client.repository.AIServiceMetaNames.TAGS: ["wx-agent", "peds"],
}

ai_service_details = client.repository.store_ai_service(
    meta_props=ai_service_metadata,
    ai_service=gen_ai_service,
)

print("AI service stored OK.")

