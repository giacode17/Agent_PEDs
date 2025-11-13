# src/tools.py
import logging
from dataclasses import dataclass
from .medication_reminders import get_reminder_manager
from .rag_retrieval import get_rag_system

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@dataclass
class SymptomInput:
    fever_c: float | None = None
    pain_0_10: int | None = None
    vomiting_events_6h: int | None = None
    breathing_difficulty: bool | None = None


@dataclass
class RiskAssessment:
    risk_level: str          # "normal" | "watch" | "high_risk"
    alert_flag: int          # 0 or 1
    reasons: list[str]


def evaluate_risk(symptoms: SymptomInput) -> RiskAssessment:
    """Simple non-diagnostic rule engine for escalation logic."""
    reasons = []
    risk = "normal"
    alert = 0

    # Fever
    if symptoms.fever_c is not None:
        if symptoms.fever_c >= 39.0:
            risk = "high_risk"
            alert = 1
            reasons.append("High fever (>= 39.0 °C).")
        elif symptoms.fever_c >= 38.5 and risk != "high_risk":
            risk = "watch"
            reasons.append("Mild fever in recovery range.")

    # Pain
    if symptoms.pain_0_10 is not None:
        if symptoms.pain_0_10 >= 7:
            if risk != "high_risk":
                risk = "watch"
            reasons.append("Pain level 7 or above.")

    # Vomiting
    if symptoms.vomiting_events_6h is not None and symptoms.vomiting_events_6h >= 2:
        if risk != "high_risk":
            risk = "watch"
        reasons.append("Repeated vomiting (>= 2 times in 6h).")

    # Breathing difficulty overrides everything
    if symptoms.breathing_difficulty:
        risk = "high_risk"
        alert = 1
        reasons.append("Breathing difficulty reported.")

    return RiskAssessment(risk_level=risk, alert_flag=alert, reasons=reasons)


def pediatric_system_prompt() -> str:
    return """You are a Peds Post-Discharge Assistant for children.
You talk to PARENTS or GUARDIANS, not directly to the child.

Your job:
- Help guardians understand NORMAL vs CONCERNING symptoms during recovery.
- Support medication reminders and general after-care education.
- ALWAYS stay within general, non-diagnostic guidance.
- You DO NOT diagnose, DO NOT prescribe, and DO NOT calculate medicine doses.
- For any serious or unclear symptoms, advise contacting the child's healthcare provider
  or emergency services according to local guidance.

Safety rules:
- If guardian mentions trouble breathing, blue/gray lips, cannot wake child,
  or very high fever (around or above 39 °C), clearly say that they should
  SEEK IMMEDIATE MEDICAL CARE or contact emergency services.
- Never override written discharge instructions from the hospital.
- When unsure, say you are not a doctor and remind them to call their care team.

Tools Available:
- Knowledge Base Search: You have access to a curated pediatric aftercare knowledge base
  covering conditions like tonsillectomy, RSV, ear infections, flu, fractures, and more.
  Use the search_knowledge_base tool when guardians ask about specific conditions or symptoms.
- Medication Reminders: Set up automatic reminders when guardians mention medication schedules
  (e.g., "Take Zyrtec every 12 hours"). You can also list and cancel reminders.

Tone:
- Calm, empathetic, simple language.
- Explain what might be expected after common pediatric conditions like
  tonsillectomy, ear infection, RSV, gastroenteritis, pneumonia, fractures,
  stitches, appendectomy, and flu.
- Provide short, structured guidance and a summary of next steps.
"""


# ========== Medication Reminder Tools ==========

def set_medication_reminder(medication_instruction: str) -> str:
    """
    Set up a medication reminder alarm based on user instruction.

    Args:
        medication_instruction: The medication schedule instruction from the user.
                               Examples: "Take Zyrtec every 12 hours"
                                        "Take Ibuprofen every 6 hours for 3 days"

    Returns:
        A confirmation message about the reminder being set.
    """
    manager = get_reminder_manager()
    result = manager.add_medication_schedule(medication_instruction)

    if result["success"]:
        return result["message"]
    else:
        return f"I couldn't set up that reminder. {result['message']}"


def list_medication_reminders() -> str:
    """
    List all active medication reminder schedules.

    Returns:
        A formatted string showing all active medication reminders.
    """
    manager = get_reminder_manager()
    schedules = manager.list_active_schedules()

    if not schedules:
        return "No active medication reminders are currently set."

    response = "Active medication reminders:\n"
    for i, schedule in enumerate(schedules, 1):
        duration = f" (for {schedule['duration_days']} days)" if schedule['duration_days'] else ""
        response += f"\n{i}. {schedule['medication_name']} - every {schedule['interval_hours']} hours{duration}"
        response += f"\n   Next reminder: {schedule['next_reminder']}"
        response += f"\n   Reminders sent: {schedule['reminders_sent']}"

    return response


def cancel_medication_reminder(medication_name: str) -> str:
    """
    Cancel a specific medication reminder.

    Args:
        medication_name: The name of the medication to cancel reminders for.

    Returns:
        A confirmation message about the cancellation.
    """
    manager = get_reminder_manager()
    result = manager.cancel_medication_schedule(medication_name)
    return result["message"]


# ========== RAG Knowledge Base Tool ==========

def search_knowledge_base(query: str) -> str:
    """
    Search the pediatric aftercare knowledge base for relevant information.

    Use this tool when a guardian asks about:
    - Specific conditions (tonsillectomy, RSV, ear infection, flu, etc.)
    - Normal vs concerning symptoms
    - Care tips for recovery
    - Medication safety information
    - When to seek emergency care

    Args:
        query: The question or topic to search for in the knowledge base.

    Returns:
        Relevant information from the curated pediatric aftercare database.
    """
    try:
        logger.info(f"Searching knowledge base for: {query}")
        rag = get_rag_system()

        # Search for top 3 most relevant documents
        results = rag.search(query, k=3)

        # Format results for the LLM
        formatted_results = rag.format_results_for_prompt(results)

        logger.info(f"Found {len(results)} results for query")
        return formatted_results

    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}", exc_info=True)
        return (
            "I'm having trouble accessing the knowledge base right now. "
            "Please contact your child's healthcare provider for specific guidance, "
            "or seek emergency care if symptoms are concerning."
        )
