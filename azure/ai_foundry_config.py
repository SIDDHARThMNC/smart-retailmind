import os
import logging

logger = logging.getLogger(__name__)


class FoundryDeploymentConfig:
    FOUNDRY_PROJECT_NAME = os.getenv("AZURE_FOUNDRY_PROJECT_NAME", "retailmind-foundry")
    FOUNDRY_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "retailmind-rg")
    FOUNDRY_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "")
    FOUNDRY_LOCATION = os.getenv("AZURE_LOCATION", "eastus")

    DEPLOYMENTS = {
        "chat": {
            "model": "gpt-4.1-mini",
            "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini"),
            "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            "api_version": "2025-01-01-preview",
            "capacity_units": 10,
            "sku": "Standard",
        },
        "embeddings": {
            "model": "text-embedding-ada-002",
            "deployment_name": "retailmind-embeddings",
            "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            "api_version": "2024-02-01",
            "capacity_units": 5,
            "sku": "Standard",
        },
    }

    APP_SERVICE = {
        "name": "retailmind-api",
        "plan": "retailmind-plan",
        "sku": "B2",
        "python_version": "3.11",
        "startup_command": "uvicorn backend.main:app --host 0.0.0.0 --port 8000",
    }


class FoundryDeploymentClient:
    def __init__(self):
        self._client = None
        self._config = FoundryDeploymentConfig()

    def _get_client(self):
        if self._client is not None:
            return self._client
        conn_str = os.getenv("AZURE_FOUNDRY_CONNECTION_STRING", "")
        if not conn_str:
            return None
        try:
            from azure.ai.projects import AIProjectClient
            from azure.identity import DefaultAzureCredential
            self._client = AIProjectClient.from_connection_string(
                credential=DefaultAzureCredential(),
                conn_str=conn_str,
            )
            logger.info("Azure AI Foundry client connected.")
            return self._client
        except ImportError:
            logger.warning("azure-ai-projects not installed.")
            return None
        except Exception as e:
            logger.warning(f"Foundry client init failed: {e}. Falling back to direct Azure OpenAI.")
            return None

    def get_chat_client(self):
        client = self._get_client()
        if client:
            try:
                return client.inference.get_azure_openai_client(api_version="2025-01-01-preview")
            except Exception as e:
                logger.warning(f"Foundry inference client failed: {e}. Using direct client.")
        try:
            from openai import AzureOpenAI
            from backend.config import settings
            return AzureOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                timeout=15.0,
                max_retries=1,
            )
        except Exception as e:
            logger.error(f"Fallback AzureOpenAI client failed: {e}")
            return None

    def health_check(self) -> dict:
        client = self._get_client()
        return {
            "foundry_connected": client is not None,
            "chat_deployment": self._config.DEPLOYMENTS["chat"]["deployment_name"],
            "fallback_mode": client is None,
            "foundry_project": self._config.FOUNDRY_PROJECT_NAME,
            "location": self._config.FOUNDRY_LOCATION,
        }

    def list_deployments(self) -> list:
        return [
            {"name": name, "model": cfg["model"], "endpoint": cfg["endpoint"], "sku": cfg["sku"]}
            for name, cfg in self._config.DEPLOYMENTS.items()
        ]


foundry_client = FoundryDeploymentClient()
