# RetailMind AI

A smart retail analytics backend built with FastAPI, MongoDB, Azure OpenAI, and a multi-agent system. The project covers demand forecasting, anomaly detection, document search using RAG, and an agent-based chat interface — all connected to Azure cloud services.

---

## What this project does

RetailMind AI is a backend system for retail analytics. You can upload sales data, train a demand forecasting model, detect anomalies in sales patterns, search through policy documents using semantic search, and ask questions to an AI agent that routes your query to the right specialist.

The system has three agents:
- **Data Analyst Agent** — answers questions about sales, revenue, and product performance
- **Document Assistant Agent** — searches policy documents (return policy, discount policy, etc.)
- **ML Expert Agent** — explains model predictions and anomaly detection results

---

## Project Structure

```
RetailMind-AI/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── .env
│   ├── agents/
│   │   ├── agent_service.py     # Multi-agent routing and orchestration
│   │   └── mcp.py               # Model Context Protocol tool registry
│   ├── database/
│   │   └── db.py                # MongoDB connection and operations
│   ├── ml/
│   │   ├── preprocessing.py     # Data cleaning and feature engineering
│   │   ├── training.py          # RandomForest model training
│   │   ├── prediction.py        # Demand prediction using saved model
│   │   └── anomaly.py           # IsolationForest anomaly detection
│   ├── rag/
│   │   └── rag_service.py       # ChromaDB + TF-IDF RAG pipeline
│   ├── routes/
│   │   ├── ingestion.py         # POST /api/ingest
│   │   ├── documents.py         # POST /api/search
│   │   ├── agents.py            # POST /api/agent/chat, GET /api/dashboard
│   │   └── ml.py                # POST /api/train, POST /api/predict, GET /api/anomalies
│   ├── services/
│   │   └── azure_openai.py      # Azure OpenAI chat client
│   ├── saved_models/
│   │   ├── demand_model.pkl
│   │   └── preprocessor.pkl
│   └── tests/
│       ├── conftest.py
│       ├── test_agents.py
│       └── test_health.py
├── azure/
│   ├── ai_foundry_config.py     # Azure AI Foundry deployment config
│   ├── keyvault_config.py       # Azure Key Vault secret manager
│   ├── foundry_deployment.yaml  # Azure ML managed endpoint YAML
│   └── deployment_diagram.md   # Architecture and request flow diagrams
├── data/
│   ├── sample_retail_sales.csv
│   ├── cleaned_retail_sales.csv
│   └── documents/
│       ├── return_policy.txt
│       ├── discount_policy.txt
│       ├── inventory_policy.txt
│       └── product_policy.txt
├── chroma_db/
├── pytest.ini
├── requirements.txt
├── startup.txt
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/ingest` | Upload a CSV file to ingest and clean sales data |
| POST | `/api/train` | Train the demand forecasting model |
| POST | `/api/predict` | Predict units sold for a product |
| GET | `/api/anomalies` | Run anomaly detection on current data |
| POST | `/api/search` | Search policy documents using RAG |
| POST | `/api/agent/chat` | Chat with the multi-agent system |
| GET | `/api/agent/tools` | List all available MCP tools |
| GET | `/api/dashboard` | Get sales analytics summary |

---

## Setup

### 1. Clone the repo and create a virtual environment

```bash
git clone <repo-url>
cd RetailMind-AI
python -m venv .venv
.venv\Scripts\activate        # Windows
# or
source .venv/bin/activate     # Linux/Mac
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create `backend/.env` with the following:

```env
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?appName=<AppName>
MONGODB_DB=retailmind

AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1-mini
AZURE_OPENAI_API_VERSION=2025-01-01-preview
```

If you have Azure Key Vault set up, add:

```env
AZURE_KEYVAULT_URL=https://<your-keyvault>.vault.azure.net/
```

The app will automatically load secrets from Key Vault at startup and fall back to `.env` if Key Vault is not configured.

### 4. Run the server

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs will be available at `http://localhost:8000/docs`

---

## Using the API

### Ingest data

```bash
curl -X POST http://localhost:8000/api/ingest \
  -F "file=@data/sample_retail_sales.csv"
```

### Train the model

```bash
curl -X POST http://localhost:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{"n_estimators": 500, "test_size": 0.2}'
```

### Predict demand

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "P101",
    "date": "2025-06-15",
    "price": 999.0,
    "discount": 10.0,
    "store_id": "S01",
    "region": "North"
  }'
```

### Search policy documents

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the return policy for electronics?"}'
```

### Chat with the agent

```bash
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Which region has the highest sales?"}'
```

---

## ML Pipeline

The ML pipeline uses a **RandomForestRegressor** for demand forecasting and **IsolationForest** for anomaly detection.

Feature engineering includes:
- Date-based features (day, month, quarter, week of year, is_weekend, etc.)
- Price and discount features (effective price, discount squared, price-discount interaction)
- Lag features (lag_1, lag_7, lag_14, lag_30)
- Rolling statistics (rolling mean, std, max, min over 7/14/30 days)
- Aggregate features (product/store/category/region average units)
- Label encoding for categorical columns

The trained model and preprocessor are saved as `.pkl` files in `backend/saved_models/`.

---

## RAG Pipeline

Policy documents in `data/documents/` are chunked using LangChain's `RecursiveCharacterTextSplitter` and indexed in **ChromaDB** with 384-dimensional sentence-transformer embeddings. When a query comes in, the top 3 relevant chunks are retrieved and passed to Azure OpenAI for a grounded answer.

If ChromaDB is unavailable, the system automatically falls back to TF-IDF based retrieval.

---

## Multi-Agent System

The agent system in `backend/agents/agent_service.py` routes each user message to one of three agents based on keyword scoring:

- Questions about sales, revenue, trends → **Data Analyst Agent**
- Questions about policies, returns, discounts → **Document Assistant Agent**
- Questions about models, forecasts, anomalies → **ML Expert Agent**

Each agent has its own system prompt and uses MCP tools to fetch relevant data before calling Azure OpenAI for the final answer.

The MCP tool registry (`backend/agents/mcp.py`) has 10 tools including `sales_summary`, `top_products`, `revenue_by_category`, `monthly_trend`, `search_policy`, `anomaly_info`, and more.

---

## Azure Services Used

| Service | Purpose |
|---------|---------|
| Azure OpenAI (gpt-4.1-mini) | Agent chat, RAG answers, dashboard insights |
| Azure AI Foundry | Model registry and deployment endpoint |
| Azure App Service | Backend hosting (B2 SKU, Python 3.11) |
| Azure Key Vault | Secret management (API keys, DB URI) |
| Azure Managed Identity | Passwordless Key Vault access |
| MongoDB Atlas | Sales data persistence |

Deployment configuration is in `azure/foundry_deployment.yaml` and the full architecture diagram is in `azure/deployment_diagram.md`.

---

## Running Tests

```bash
pytest -v
```

The test suite has 22 tests covering agent routing logic, API response codes and schemas, RAG search, and health endpoints. The `conftest.py` automatically trains the model before tests run and restores the original state after.

```
22 passed in ~60s
```

---

## Security

- All secrets are loaded from environment variables, never hardcoded
- `.env` is in `.gitignore` and never committed
- In production, secrets are loaded from Azure Key Vault at startup
- The app uses system-assigned managed identity for Key Vault access (no stored credentials)

---

## Notes

- The first request after startup may be slightly slower as ChromaDB builds the vector index from the policy documents
- If Azure OpenAI credentials are not set, the API still works — it just returns the raw data without GPT-enhanced insights
- The model needs to be trained at least once via `POST /api/train` before predictions will work
