# src/agent.py
from ibm_watsonx_ai import APIClient
from langchain_ibm import ChatWatsonx
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from .tools import pediatric_system_prompt, SymptomInput, evaluate_risk

"""# Try to import create_agent; if unavailable, fall back to create_react_agent
try:
    from langchain.agents import create_agent
    USE_CREATE_AGENT = True
except ImportError:
    from langgraph.prebuilt import create_react_agent
    USE_CREATE_AGENT = False
"""

class PediatricAgentService:
    """
    Encapsulates LangGraph agent + watsonx.ai LLM for pediatric post-discharge.
    """

    def __init__(self, context, params: dict | None = None):
        self.context = context
        self.params = params or {}
        self.service_url = "https://us-south.ml.cloud.ibm.com"
        self.space_id = self.params.get("space_id")
        self.model_id = "ibm/granite-3-3-8b-instruct"

        # Inner client uses token from runtime context
        credentials = {
            "url": self.service_url,
            "token": context.generate_token()
        }
        self.api_client = APIClient(credentials)
        if self.space_id:
            self.api_client.set.default_space(self.space_id)

        self._llm = self._create_chat_model()
        self._memory = MemorySaver()

    def _create_chat_model(self) -> ChatWatsonx:
        params = {
            "frequency_penalty": 0,
            "max_tokens": 2000,
            "presence_penalty": 0,
            "temperature": 0.2,
            "top_p": 1,
        }
        return ChatWatsonx(
            model_id=self.model_id,
            url=self.service_url,
            space_id=self.space_id,
            params=params,
            watsonx_client=self.api_client,
        )

    def _convert_messages(self, messages_json):
        lc_messages = []
        for m in messages_json:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        return lc_messages

    def _build_agent_graph(self, messages_json):
        system_text = pediatric_system_prompt()

        # Allow extra system messages to extend instructions
        for m in messages_json:
            if m.get("role") == "system":
                system_text += "\n\nAdditional instructions:\n" + m.get("content", "")

        if USE_CREATE_AGENT:
            # Newer style: create_agent from langchain.agents
            agent = create_agent(
                model=self._llm,
                tools=[],                          # add tools list here later
                system_prompt=system_text,
                checkpointer=self._memory,
                name="pediatric_post_discharge_agent",
            )
        else:
            # Fallback to LangGraph prebuilt create_react_agent
            from langchain_core.prompts import ChatPromptTemplate
            from langgraph.prebuilt import create_react_agent

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_text),
                    ("human", "{input}")
                ]
            )
            agent = create_react_agent(
                self._llm,
                tools=[],
                prompt=prompt,
                checkpointer=self._memory,
                name="pediatric_post_discharge_agent",
            )

        return agent

    # -------- Non-streaming --------
    def generate(self, context):
        import time
        payload = context.get_json()
        messages = payload.get("messages", [])

        start = time.time()

        agent = self._build_agent_graph(messages)
        state_input = {"messages": self._convert_messages(messages)}

        result_state = agent.invoke(
            state_input,
            {"configurable": {"thread_id": "peds-42"}}
        )

        last_msg = result_state["messages"][-1]
        generated_text = last_msg.content

        elapsed_ms = int((time.time() - start) * 1000)

        # Optional: plug MLflow here later (using elapsed_ms, etc.)

        return {
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    }
                }],
                "model_id": self.model_id
            }
        }

    # -------- Streaming --------
    def generate_stream(self, context):
        payload = context.get_json()
        messages = payload.get("messages", [])

        agent = self._build_agent_graph(messages)
        state_input = {"messages": self._convert_messages(messages)}

        # stream both updates and messages
        stream = agent.stream(
            state_input,
            {"configurable": {"thread_id": "peds-42"}},
            stream_mode=["updates", "messages"],
        )

        for chunk in stream:
            chunk_type = chunk[0]

            if chunk_type == "messages":
                message_object = chunk[1][0]
                if getattr(message_object, "type", "") == "AIMessageChunk" and message_object.content:
                    yield {
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": message_object.content
                            }
                        }],
                        "model_id": self.model_id
                    }

            elif chunk_type == "updates":
                # Tool / final updates – you can add rich streaming here later.
                # For now, just skip or handle final content similar to above.
                continue
