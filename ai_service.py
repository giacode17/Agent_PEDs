# ai_service.py
from src.agent import PediatricAgentService

def gen_ai_service(context, params=None, **custom):
    """
    Entry point for Watsonx AI Service.
    Returns (generate, generate_stream) callables.
    """
    service = PediatricAgentService(context, params=params)

    def generate(ctx):
        return service.generate(ctx)

    def generate_stream(ctx):
        return service.generate_stream(ctx)

    return generate, generate_stream
