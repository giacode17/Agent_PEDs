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

 #To import create_agent
from langchain.agents import create_agent
USE_CREATE_AGENT = True
print("model success")

import logging
logger = logging.getLogger(__name__)


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
        llm = ChatWatsonx(
            model_id=self.model_id,
            url=self.service_url,
            space_id=self.space_id,
            params=params,
            watsonx_client=self.api_client,
        )
        return llm

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

        # Newer style: create_agent from langchain.agents
        agent = create_agent(
            model=self._llm,
            tools=tools,
            system_prompt=system_text,
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
            {
                "configurable": {"thread_id": "peds-42"},
                "recursion_limit": 25  # Allow multiple tool call iterations
            }
        )

        # Debug: Log message types
        logger.info(f"Total messages in result: {len(result_state['messages'])}")
        for i, msg in enumerate(result_state['messages']):
            msg_type = type(msg).__name__
            logger.info(f"Message {i}: {msg_type}")

        # Get the final assistant message (skip tool call messages)
        last_msg = None
        for msg in reversed(result_state["messages"]):
            # Look for AIMessage that's not a tool call
            if hasattr(msg, 'type') and msg.type == 'ai':
                if not hasattr(msg, 'tool_calls') or not msg.tool_calls:
                    last_msg = msg
                    break
            elif hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content:
                last_msg = msg
                break

        if not last_msg:
            last_msg = result_state["messages"][-1]

        generated_text = last_msg.content
        logger.info(f"Final generated text: {generated_text[:100] if generated_text else 'None'}...")

        elapsed_ms = int((time.time() - start) * 1000)

        # Optional: plug MLflow here later (using elapsed_ms, etc.)
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        # 3) Detect medication reminders and high-risk escalations
        text_lower = (generated_text or "").lower()

        # Medication reminder: Check if medication_reminder_tool was actually called
        medication_reminder_triggered = False
        for msg in result_state["messages"]:
            # Check for AI messages with tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if 'name' in tool_call and 'medication_reminder_tool' in str(tool_call.get('name', '')):
                        medication_reminder_triggered = True
                        logger.info("Detected medication_reminder_tool call")
                        break
            # Also check tool messages (responses from tools)
            if hasattr(msg, 'name') and msg.name == 'medication_reminder_tool':
                medication_reminder_triggered = True
                logger.info("Detected medication_reminder_tool execution")

        # Fallback: Also check response text for confirmation phrases
        if not medication_reminder_triggered:
            medication_reminder_triggered = any(
                kw in text_lower for kw in ["reminder set", "remind you", "alarm set", "medication schedule",
                                           "✓ reminder", "i have set", "i've set a reminder", "setting a reminder"]
            )
            if medication_reminder_triggered:
                logger.info("Detected medication reminder via text matching")

        # Escalation: only flag true emergencies (high-risk cases)
        # Check for emergency keywords AND exclude general advice
        has_emergency_keywords = any(
            kw in text_lower for kw in ["call 911", "go to the emergency", "seek immediate", "emergency room"]
        )
        has_general_advice = "if symptoms worsen" in text_lower or "contact your doctor if" in text_lower

        # Only flag as escalation if it's urgent language, not general advice
        escalation_triggered = has_emergency_keywords and not has_general_advice

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
                
                continue
