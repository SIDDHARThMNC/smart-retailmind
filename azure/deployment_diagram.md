# RetailMind AI — Azure Deployment Diagram

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          AZURE AI FOUNDRY PROJECT                               │
│                           (retailmind-foundry)  eastus                          │
│                                                                                 │
│   ┌──────────────────────┐          ┌──────────────────────────────────────┐   │
│   │   Model Registry      │          │         Deployments                  │   │
│   │  ──────────────────  │          │  ──────────────────────────────────  │   │
│   │  demand_model:1       │─────────▶│  Endpoint: retailmind-foundry-       │   │
│   │  (RandomForest)       │          │           endpoint                   │   │
│   │                       │          │                                      │   │
│   │  preprocessor:1       │          │  ┌─────────────────────────────┐    │   │
│   │  (LabelEncoders)      │          │  │  blue  (100% traffic)        │    │   │
│   └──────────────────────┘          │  │  Standard_DS2_v2 × 1         │    │   │
│                                      │  │  FastAPI + Uvicorn           │    │   │
│   ┌──────────────────────┐          │  │  Python 3.11                 │    │   │
│   │   Azure OpenAI        │          │  └─────────────────────────────┘    │   │
│   │  ──────────────────  │          └──────────────────────────────────────┘   │
│   │  gpt-4.1-mini         │◀──────────────────────────────────────────────┐    │
│   │  (Agent Chat + RAG)   │                                               │    │
│   │                       │                                               │    │
│   │  text-embedding-      │◀── TF-IDF RAG Pipeline                       │    │
│   │  ada-002 (future)     │                                               │    │
│   └──────────────────────┘                                               │    │
└───────────────────────────────────────────────────────────────────────────┼────┘
                                                                            │
                                          ┌─────────────────────────────────┘
                                          │
                          ┌───────────────▼──────────────────┐
                          │        Azure App Service          │
                          │        (retailmind-api)           │
                          │        SKU: B2  |  Python 3.11    │
                          │                                   │
                          │   POST /api/agent/chat            │
                          │   POST /api/search                │
                          │   POST /api/ingest                │
                          │   POST /api/train                 │
                          │   POST /api/predict               │
                          │   GET  /api/anomalies             │
                          │   GET  /api/dashboard             │
                          │   GET  /health                    │
                          └──────────────┬────────────────────┘
                                         │
                                         │  System-Assigned
                                         │  Managed Identity
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
              ▼                          ▼
┌─────────────────────┐   ┌─────────────────────┐
│   Azure Key Vault   │   │   MongoDB Atlas      │
│   (retailmind-kv)   │   │   (RetailMindDB)      │
│  ─────────────────  │   │  ─────────────────── │
│  azure-openai-      │   │  retailmind DB        │
│    api-key          │   │  sales collection     │
│  mongodb-uri        │   │  (product, date,      │
│  azure-openai-      │   │   revenue, region,    │
│    endpoint         │   │   store, category)    │
│  foundry-           │   └─────────────────────┘
│    connection-str   │
└─────────────────────┘
         │
         │  RBAC: "Key Vault Secrets User"
         │  (granted to App Service Managed Identity)
         │
         └──▶ Secrets injected into os.environ at startup
              via azure/keyvault_config.py → backend/config.py
```

---

## Request Flow Diagram

```
  User / Frontend
       │
       │  HTTP Request
       ▼
┌──────────────────┐
│  Azure App       │
│  Service         │
│  (FastAPI)       │
└──────┬───────────┘
       │
       ├──▶ POST /api/agent/chat
       │         │
       │         ├──▶ agent_service.py  (route to agent)
       │         │         │
       │         │         ├──▶ mcp.py  (select tool)
       │         │         │       │
       │         │         │       ├──▶ sales_summary / top_products /
       │         │         │       │    revenue_by_category / monthly_trend
       │         │         │       │    (reads cleaned_retail_sales.csv)
       │         │         │       │
       │         │         │       └──▶ search_policy
       │         │         │                │
       │         │         │                ▼
       │         │         │         rag_service.py
       │         │         │         (TF-IDF document search)
       │         │         │
       │         │         └──▶ azure_openai.py
       │         │                   │
       │         │                   ▼
       │         │            Azure OpenAI
       │         │            gpt-4.1-mini
       │         │            (grounded answer)
       │         │
       ├──▶ POST /api/train
       │         │
       │         └──▶ ml/training.py
       │                   │
       │                   ├──▶ ml/preprocessing.py  (feature engineering)
       │                   └──▶ saved_models/demand_model.pkl  (saved)
       │
       ├──▶ POST /api/predict
       │         │
       │         └──▶ ml/prediction.py
       │                   └──▶ saved_models/demand_model.pkl  (loaded)
       │
       ├──▶ GET /api/anomalies
       │         │
       │         └──▶ ml/anomaly.py
       │                   └──▶ IsolationForest on sales CSV
       │
       └──▶ POST /api/ingest
                 │
                 ├──▶ ml/preprocessing.py  (validate + clean)
                 ├──▶ data/cleaned_retail_sales.csv  (saved)
                 └──▶ MongoDB Atlas  (insert records)
```

---

## Security Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SECURITY LAYERS                          │
│                                                             │
│  Layer 1 — Secret Storage                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Azure Key Vault  (retailmind-kv)                   │   │
│  │  • azure-openai-api-key                             │   │
│  │  • mongodb-uri                                      │   │
│  │  • azure-openai-endpoint                            │   │
│  │  • foundry-connection-str                           │   │
│  │  NO plaintext secrets in code or YAML               │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  Layer 2 — Identity      │                                  │
│  ┌───────────────────────▼─────────────────────────────┐   │
│  │  System-Assigned Managed Identity                   │   │
│  │  (App Service → Key Vault)                          │   │
│  │  RBAC Role: "Key Vault Secrets User"                │   │
│  │  No credentials stored anywhere                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│  Layer 3 — Runtime       │                                  │
│  ┌───────────────────────▼─────────────────────────────┐   │
│  │  keyvault_config.py → config.py                     │   │
│  │  • Secrets loaded into os.environ at startup        │   │
│  │  • In-memory cache (no repeated KV calls)           │   │
│  │  • .env fallback for local development              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Local Dev: .env file (never committed to git)             │
└─────────────────────────────────────────────────────────────┘
```

---

## Azure Components Used

| Component | Purpose | Status |
|---|---|---|
| **Azure OpenAI** (gpt-4.1-mini) | Agent chat + RAG grounded answers | ✅ Active |
| **Azure AI Foundry** | Model registry, deployment orchestration, inference endpoint | ✅ Configured |
| **Azure App Service** (B2) | FastAPI backend hosting | ✅ Configured |
| **Azure Key Vault** | Secret management (API keys, DB URI) | ✅ Configured |
| **Azure Managed Identity** | Passwordless Key Vault access | ✅ Configured |
| **MongoDB Atlas** | Sales data persistence | ✅ Active |
| **TF-IDF RAG** | Document search (in-process, no external dependency) | ✅ Active |

---

## Fallback Strategy (Zero Downtime)

```
AZURE_FOUNDRY_CONNECTION_STRING set?
        │
        ├── YES ──▶ Use Azure AI Foundry inference client
        │
        └── NO  ──▶ Use direct AzureOpenAI (existing working setup)
                              │
                    AZURE_KEYVAULT_URL set?
                              │
                              ├── YES ──▶ Load secrets from Key Vault
                              │
                              └── NO  ──▶ Load secrets from .env
                                         (local dev — always works)
```
