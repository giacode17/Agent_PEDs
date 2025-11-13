# Pediatric Post-Discharge Agent

AI-powered assistant for parents and guardians to support children's recovery after hospital discharge. Built with IBM Watsonx.ai, LangGraph, and RAG (Retrieval Augmented Generation).

## 🎯 Features

### Core Capabilities
- **Safety-First Design**: Non-diagnostic, guardian-focused guidance
- **Risk Assessment**: Evaluates symptoms and provides escalation recommendations
- **Medication Reminders**: Automated alarm system for prescription schedules
- **RAG Knowledge Base**: Retrieves information from curated pediatric aftercare data
- **MLflow Integration**: Tracks conversation metrics and model performance

### Supported Conditions
- Post-tonsillectomy care
- RSV/Bronchiolitis recovery
- Ear infections (Otitis Media)
- Seasonal flu
- Pneumonia aftercare
- Gastroenteritis
- Asthma exacerbations
- Fracture cast care
- Wound care (stitches/glue)
- Appendectomy recovery

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    IBM Watsonx.ai                           │
│                 (Granite-3-3-8B-Instruct)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                   LangGraph Agent                           │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Tools:                                             │    │
│  │  • Risk Assessment (fever, pain, symptoms)         │    │
│  │  • Medication Reminder Management                  │    │
│  │  • RAG Knowledge Base Search                       │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐   ┌──────────────────┐   ┌────────────────┐
│ Risk         │   │ Medication       │   │ ChromaDB       │
│ Evaluator    │   │ Scheduler        │   │ Vector Store   │
└──────────────┘   └──────────────────┘   └────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.11
- Poetry (package manager)
- IBM Cloud account with Watsonx.ai access

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd peds-post-discharge-agent
```

2. **Install dependencies**
```bash
poetry install
```

3. **Configure environment**
Create a `.env` file:
```bash
IBM_CLOUD_API_KEY=your_api_key_here
SPACE_ID=your_space_id_here
```

4. **Configure deployment**
Copy and edit `config.toml.example`:
```bash
cp config.toml.example config.toml
# Edit config.toml with your Space ID
```

## 🚀 Usage

### Local Development

Run the agent locally:
```bash
poetry run python run_local.py
```

### Testing

Run the full test suite:
```bash
poetry run pytest tests/ -v
```

Run specific test categories:
```bash
# Test medication reminders
poetry run pytest tests/test_medication_reminders.py -v

# Test RAG system
poetry run pytest tests/test_rag.py -v

# Test risk assessment
poetry run pytest tests/test_tools.py -v
```

### Interactive Testing

Test medication reminders:
```bash
poetry run python test_medication_reminders.py
```

Test RAG knowledge retrieval:
```bash
poetry run python test_rag_system.py
```

## 🔧 Key Components

### 1. Medication Reminder System

Automatically schedules and triggers medication reminders:

```python
from peds_post_discharge_agent.tools import set_medication_reminder

# Set a reminder
result = set_medication_reminder("Take Zyrtec every 12 hours")
# ✓ Reminder set for Zyrtec every 12.0 hours. First reminder at 14:30:00.

# With duration
result = set_medication_reminder("Take Ibuprofen every 6 hours for 3 days")
```

**Features:**
- Parses natural language medication schedules
- Configurable intervals (hours)
- Optional duration (days/weeks)
- Console alarm notifications
- Thread-safe scheduling

### 2. RAG Knowledge Base

Retrieves relevant information from curated pediatric datasets:

```python
from peds_post_discharge_agent.tools import search_knowledge_base

# Search for condition information
result = search_knowledge_base("RSV cough normal symptoms")
# Returns: Relevant aftercare guidance from knowledge base
```

**Data Sources:**
- `pediatric_aftercare.jsonl` - 10 common post-discharge conditions
- `medication_guides.jsonl` - Pediatric medication safety info
- Embedded using ChromaDB with all-MiniLM-L6-v2 model

### 3. Risk Assessment

Evaluates symptoms and determines risk level:

```python
from peds_post_discharge_agent.tools import SymptomInput, evaluate_risk

symptoms = SymptomInput(
    fever_c=39.2,
    pain_0_10=8,
    vomiting_events_6h=2,
    breathing_difficulty=False
)

risk = evaluate_risk(symptoms)
# Returns: RiskAssessment(
#   risk_level="high_risk",
#   alert_flag=1,
#   reasons=["High fever (>= 39.0 °C)"]
# )
```

## 📊 Project Structure

```
peds-post-discharge-agent/
├── src/peds_post_discharge_agent/
│   ├── agent.py                    # Main LangGraph agent
│   ├── tools.py                    # Agent tools & risk assessment
│   ├── medication_reminders.py     # Medication scheduler
│   └── rag_retrieval.py           # RAG system with ChromaDB
├── peds-dataset/
│   └── pediatric_agent_dataset/   # Curated medical knowledge
├── tests/                          # Comprehensive test suite
│   ├── test_tools.py              # Risk assessment tests
│   ├── test_medication_reminders.py
│   └── test_rag.py                # RAG retrieval tests
├── ai_service.py                  # Watsonx.ai service wrapper
├── run_local.py                   # Local development script
└── run_remote.py                  # Remote deployment script
```

## 🔒 Safety & Compliance

**This agent is designed for:**
- General after-care guidance
- Symptom education (normal vs. concerning)
- Medication reminder support
- Emergency escalation recommendations

**This agent does NOT:**
- Diagnose conditions
- Prescribe medications
- Calculate medication doses
- Replace medical professional advice

**Red Flag Detection:**
- High fever (≥ 39.0°C)
- Breathing difficulty
- Severe pain
- Dehydration signs
- Other emergency symptoms

## 📈 MLflow Tracking

The agent logs conversation metrics to MLflow:

- Response latency
- Medication reminder flags
- Escalation triggers
- Model parameters

Enable MLflow in `ai_service.py`:
```python
params = {
    "mlflow_enabled": True,
    "mlflow_tracking_uri": "file:./mlruns",
    "mlflow_experiment_name": "peds_post_discharge_agent",
}
```

View metrics:
```bash
mlflow ui --backend-store-uri file:./mlruns
```

## 🧪 Testing

**Test Coverage: 21 tests, 100% passing**

Test categories:
- **Risk Assessment** (2 tests) - Symptom evaluation logic
- **Medication Reminders** (10 tests) - Parsing, scheduling, cancellation
- **RAG System** (9 tests) - Search, retrieval, formatting

Run with coverage:
```bash
poetry run pytest tests/ --cov=src/peds_post_discharge_agent --cov-report=html
```

## 🚢 Deployment

### Watsonx.ai Deployment

1. Ensure `config.toml` is configured
2. Deploy using Watsonx CLI:
```bash
watsonx-ai deploy --config config.toml
```

3. Test the deployment:
```bash
poetry run python run_remote.py
```

## 🤝 Contributing

### Development Setup

1. Install development dependencies:
```bash
poetry install --with dev
```

2. Run linting:
```bash
poetry run black src/ tests/
poetry run pylint src/
```

3. Run tests before committing:
```bash
poetry run pytest tests/ -v
```

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

- IBM Watsonx.ai team for the Granite LLM
- LangChain/LangGraph for agent framework
- ChromaDB for vector storage
- Curated pediatric medical content (synthetic, educational)

## ⚠️ Disclaimer

This is a demonstration project with synthetic educational content. It is **not** intended for actual medical use. Always consult qualified healthcare providers for medical advice, diagnosis, or treatment.

---

**Built with ❤️ for pediatric care**
