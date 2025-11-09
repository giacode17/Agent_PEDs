# tests/test_tools.py
from src.tools import SymptomInput, evaluate_risk

def test_high_fever_triggers_alert():
    symptoms = SymptomInput(fever_c=39.2)
    result = evaluate_risk(symptoms)
    assert result.risk_level == "high_risk"
    assert result.alert_flag == 1

def test_mild_fever_watch():
    symptoms = SymptomInput(fever_c=38.6)
    result = evaluate_risk(symptoms)
    assert result.risk_level == "watch"
    assert result.alert_flag == 0
