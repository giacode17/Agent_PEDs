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

##  Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    IBM Watsonx.ai                           │
│                 (Granite-3-3-8B-Instruct)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                   LangGraph Agent                           │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Tools:                                            │     │
│  │  • Risk Assessment (fever, pain, symptoms)         │     │
│  │  • Medication Reminder Management                  │     │
│  │  • RAG Knowledge Base Search                       │     │
│  └────────────────────────────────────────────────────┘     │
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

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure environment**
Create a `.env` file:
```bash
IBM_CLOUD_API_KEY=your_api_key_here
SPACE_ID=your_space_id_here
```


## 🚀 Usage

### Run the Agent Locally

Start an interactive conversation with the agent:

```bash
watsonx-ai template invoke "<PROMPT>"
```

### Example Conversation

```
You: My child has a fever of 38.5°C and mild pain after surgery
Agent: I'll help assess those symptoms...
      [Performs risk assessment]
      ✓ Symptoms Appear Normal
      These symptoms are typically expected during recovery.
      Continue monitoring and follow discharge instructions.

You: Remind me to give Zyrtec every 12 hours
Agent: [Sets up medication reminder]
      ✓ Reminder set for Zyrtec every 12.0 hours.

You: What foods are okay after tonsillectomy?
Agent: [Searches knowledge base]
      Based on our medical guidance:
      • Soft foods like yogurt, pudding, ice cream
      • Cold foods help soothe throat pain
      • Avoid acidic or spicy foods...
```

### Testing

Run the full test suite (21 tests):
```bash
poetry run pytest tests/ -v
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
- Embedded using ChromaDB with default embeddings

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
#   reasons=["High fever (>= 39.0 °C)", "Severe pain (>= 7)"]
# )
```

**Risk Levels:**
- `normal` - Expected recovery symptoms
- `watch` - Monitor closely, contact doctor if worsens
- `high_risk` - Seek immediate medical care

## 📊 Project Structure

```
peds-post-discharge-agent/
├── src/peds_post_discharge_agent/
│   ├── agent.py                    # Main LangGraph agent
│   ├── tools.py                    # Agent tools & risk assessment
│   ├── medication_reminders.py     # Medication scheduler
│   └── rag_retrieval.py           # RAG system with ChromaDB
├── data/                           # Knowledge base data
│   └── pediatric_agent_dataset/
│       ├── pediatric_aftercare.jsonl
│       └── medication_guides.jsonl
├── tests/                          # Comprehensive test suite
│   ├── test_tools.py              # Risk assessment tests
│   ├── test_medication_reminders.py
│   └── test_rag.py                # RAG retrieval tests
├── extras/                         # Optional integrations
│   ├── watsonx-assistant-integration/  # Chatbot UI (optional)
│   └── watsonx-ai-deployment/     # Cloud deployment (optional)
├── ai_service.py                  # Watsonx.ai service wrapper
├── run_local.py                   # Local development script
├── run_remote.py                  # Remote API script
├── test_medication_reminders.py   # Interactive reminder test
└── test_rag_system.py            # Interactive RAG test
```



## 📈 MLflow Tracking

The agent logs conversation metrics to MLflow:

- Response latency
- Medication reminder flags
- Escalation triggers
- Model parameters

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

## 🌐 Optional Integrations

This project includes optional integrations in the `extras/` folder:

### Watsonx Assistant Integration
Add a conversational UI with visual dialog design, web chat, and multi-channel support.

See: `extras/watsonx-assistant-integration/WATSONX_ASSISTANT_SETUP.md`

### Watsonx.ai Cloud Deployment
Deploy the agent to IBM Cloud for scalable production hosting.

See: `extras/watsonx-ai-deployment/DEPLOYMENT.md`


## 🙏 Acknowledgments

- IBM Watsonx.ai team for the Granite LLM
- LangChain/LangGraph for agent framework
- ChromaDB for vector storage
- Curated pediatric medical content (synthetic, educational)


