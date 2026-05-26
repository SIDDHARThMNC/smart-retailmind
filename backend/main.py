import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("APP_ROOT", _root)
if _root not in sys.path:
    sys.path.insert(0, _root)
# ensure working directory is project root (safe on both local and Azure)
if os.getcwd() != _root:
    try:
        os.chdir(_root)
    except Exception:
        pass

from backend.routes import ingestion, documents, agents, ml
from backend.database.db import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
        logger.info("Database ready.")
    except Exception as e:
        logger.warning(f"DB init warning: {e}")

    # auto-train model on first deploy if pkl files are missing
    model_path = os.path.join(_root, "backend", "saved_models", "demand_model.pkl")
    data_path = os.path.join(_root, "data", "cleaned_retail_sales.csv")
    if not os.path.exists(model_path) and os.path.exists(data_path):
        try:
            from backend.ml.training import train_models
            logger.info("Model not found — training on startup...")
            train_models(data_path=data_path)
            logger.info("Startup model training complete.")
        except Exception as e:
            logger.warning(f"Startup model training failed: {e}")

    yield


app = FastAPI(
    title="RetailMind AI",
    description="Smart Retail Analytics Backend",
    version="1.0.0",
    lifespan=lifespan
)

_cors_origins_raw = os.environ.get("ALLOWED_ORIGINS", "*")
_cors_origins = [o.strip() for o in _cors_origins_raw.split(",")] if _cors_origins_raw != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {
        "message": "RetailMind AI API running successfully"
    }


@app.get("/health")
def health():
    return {"status": "running"}


app.include_router(ingestion.router)
app.include_router(documents.router)
app.include_router(agents.router)
app.include_router(ml.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
    )
