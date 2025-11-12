# ai_service.py
#from src.agent import PediatricAgentService
import os
from dotenv import load_dotenv
from peds_post_discharge_agent.agent import PediatricAgentService

load_dotenv()
API_KEY = os.getenv("IBM_CLOUD_API_KEY") or os.getenv("WATSONX_APIKEY")
SPACE_ID = os.getenv("SPACE_ID") or os.getenv("WATSONX_SPACE_ID")

def gen_ai_service(context, params=None, **custom):
    """
    This is the function you'll register as the AI Service in Watsonx.
    It returns (generate, generate_stream) for online deployments.
    """
    service = PediatricAgentService(context, params=params or {})

    def generate(ctx):
        return service.generate(ctx)

    def generate_stream(ctx):
        return service.generate_stream(ctx)

    return generate, generate_stream


def deployable_ai_service(context, **online_params):
    """
    This is what the watsonx-ai CLI template expects when running locally:
    it calls deployable_ai_service(context, **online_params) and expects
    back a SINGLE callable (non-streaming or streaming).

    For now, we keep it simple and always return the NON-streaming version.
    That means in config.toml, stream must be set to false.
    """
    #generate, _ = gen_ai_service(context, params=online_params)
    params = {
    "space_id":SPACE_ID,
    "mlflow_enabled": True,
    "mlflow_tracking_uri": "file:./mlruns",
    "mlflow_experiment_name": "peds_post_discharge_agent",
    }
    generate, _ = gen_ai_service(context, params=params)

    
    return generate
