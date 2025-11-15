
import os
from dotenv import load_dotenv
from peds_post_discharge_agent.agent import PediatricAgentService

load_dotenv()
API_KEY = os.getenv("IBM_CLOUD_API_KEY") or os.getenv("WATSONX_APIKEY")
SPACE_ID = os.getenv("SPACE_ID") or os.getenv("WATSONX_SPACE_ID")

def gen_ai_service(context, params=None, **custom):
    
    service = PediatricAgentService(context, params=params or {})

    def generate(ctx):
        return service.generate(ctx)

    def generate_stream(ctx):
        return service.generate_stream(ctx)

    return generate, generate_stream

# Watsonx.ai CLI entry point for local deployment
def deployable_ai_service(context, **online_params):
    """
    Note: config.toml must have stream=false for this to work.
    Args:
        context: RuntimeContext from watsonx.ai
        **online_params: Additional parameters from deployment config

    Returns:
        Callable: The generate function for processing requests
    """
    params = {
        "space_id": SPACE_ID,
        "mlflow_enabled": True,
        "mlflow_tracking_uri": "file:./mlruns",
        "mlflow_experiment_name": "peds_post_discharge_agent",
    }
    generate, _ = gen_ai_service(context, params=params)
    return generate
