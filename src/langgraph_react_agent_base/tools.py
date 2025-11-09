# src/tools.py
from dataclasses import dataclass

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
    return """You are a Pediatric Post-Discharge Assistant for children.
You talk to PARENTS or GUARDIANS, not directly to the child.

Your job:
- Help guardians understand NORMAL vs CONCERNING symptoms during recovery.
- Support medication reminders and general after-care education.
- ALWAYS stay within general, non-diagnostic guidance.
- You DO NOT diagnose, DO NOT prescribe, and DO NOT calculate medicine doses.
- For any serious or unclear symptoms, advise contacting the child’s healthcare provider
  or emergency services according to local guidance.

Safety rules:
- If guardian mentions trouble breathing, blue/gray lips, cannot wake child,
  or very high fever (around or above 39 °C), clearly say that they should
  SEEK IMMEDIATE MEDICAL CARE or contact emergency services.
- Never override written discharge instructions from the hospital.
- When unsure, say you are not a doctor and remind them to call their care team.

Tone:
- Calm, empathetic, simple language.
- Explain what might be expected after common pediatric conditions like
  tonsillectomy, ear infection, RSV, gastroenteritis, pneumonia, fractures,
  stitches, appendectomy, and flu.
- Provide short, structured guidance and a summary of next steps.
"""
