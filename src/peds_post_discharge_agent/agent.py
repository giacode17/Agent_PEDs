# src/agent.py
from ibm_watsonx_ai import APIClient
from langchain_ibm import ChatWatsonx
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
import mlflow
from datetime import datetime

from .tools import (
    pediatric_system_prompt,
    SymptomInput,
    evaluate_risk,
    set_medication_reminder,
    list_medication_reminders,
    cancel_medication_reminder,
    search_knowledge_base
)
from langchain_core.tools import tool


 #Try to import create_agent; if unavailable, fall back to create_react_agent
try:
    from langchain.agents import create_agent
    USE_CREATE_AGENT = True
    print("model success")
except ImportError:
    from langgraph.prebuilt import create_react_agent
    USE_CREATE_AGENT = False




class PediatricAgentService:
    """
    Encapsulates LangGraph agent + watsonx.ai LLM for peds post-discharge.
    """

    def __init__(self, context, params: dict | None = None):
        self.context = context
        self.params = params or {}
        self.service_url = "https://us-south.ml.cloud.ibm.com"
        self.space_id = self.params.get("space_id")
        self.model_id = "ibm/granite-3-3-8b-instruct"
        self.mlflow_enabled = self.params.get("mlflow_enabled", False)

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

        if self.mlflow_enabled:
            tracking_uri = self.params.get("mlflow_tracking_uri", "file:./mlruns")
            experiment_name = self.params.get("mlflow_experiment_name", "peds_post_discharge_agent")

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)


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

    def _get_tools(self):
        """Create LangChain tools for the agent."""
        # Wrap medication reminder functions as LangChain tools
        @tool
        def medication_reminder_tool(medication_instruction: str) -> str:
            """Set up a medication reminder alarm. Use when a guardian mentions a medication schedule like 'Take Zyrtec every 12 hours' or 'Ibuprofen every 6 hours for 3 days'."""
            return set_medication_reminder(medication_instruction)

        @tool
        def list_reminders_tool() -> str:
            """List all active medication reminders."""
            return list_medication_reminders()

        @tool
        def cancel_reminder_tool(medication_name: str) -> str:
            """Cancel a specific medication reminder by medication name."""
            return cancel_medication_reminder(medication_name)

        @tool
        def knowledge_base_tool(query: str) -> str:
            """Search the pediatric aftercare knowledge base for information about conditions, symptoms, care tips, medications, and red flags. Use this when guardians ask about specific conditions or symptoms."""
            return search_knowledge_base(query)

        return [
            medication_reminder_tool,
            list_reminders_tool,
            cancel_reminder_tool,
            knowledge_base_tool
        ]

    def _build_agent_graph(self, messages_json):
        system_text = pediatric_system_prompt()

        # Allow extra system messages to extend instructions
        for m in messages_json:
            if m.get("role") == "system":
                system_text += "\n\nAdditional instructions:\n" + m.get("content", "")

        # Get medication reminder tools
        tools = self._get_tools()

        if USE_CREATE_AGENT:
            # Newer style: create_agent from langchain.agents
            agent = create_agent(
                model=self._llm,
                tools=tools,
                system_prompt=system_text,
                checkpointer=self._memory,
                name="peds_post_discharge_agent",
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
                tools=tools,
                prompt=prompt,
                checkpointer=self._memory,
                name="peds_post_discharge_agent",
            )

        return agent

    # -------- Non-streaming --------

    def generate(self, context):
        import time
        payload = context.get_json()
        messages = payload.get("messages", [])

        start = time.time()

        # 1) Build agent graph & invoke
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
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        # 3) Very simple flags for demo (you can refine later)
        text_lower = (generated_text or "").lower()
        medication_reminder_triggered = any(
            kw in text_lower for kw in ["medication", "take your medicine", "next dose"]
        )
        escalation_triggered = any(
            kw in text_lower for kw in ["go to the emergency", "er", "call your doctor", "call 911"]
        )

        # 4) Optional MLflow logging
        if self.mlflow_enabled:
            with mlflow.start_run(run_name="peds_session"):
                # Params (categorical / text)
                mlflow.log_param("user_text", user_msg)
                mlflow.log_param("model_id", self.model_id)

                # Metrics (numbers)
                mlflow.log_metric("elapsed_ms", elapsed_ms)
                mlflow.log_metric("medication_reminder_flag",
                                  1.0 if medication_reminder_triggered else 0.0)
                mlflow.log_metric("escalation_flag",
                                  1.0 if escalation_triggered else 0.0)

                print("[MLflow] Logged run: elapsed_ms =", elapsed_ms)

        # 5) Return response in watsonx.ai format
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
