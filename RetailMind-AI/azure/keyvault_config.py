import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

KEYVAULT_URL: str = os.getenv("AZURE_KEYVAULT_URL", "")


class KeyVaultSecretManager:
    SECRET_MAP = {
        "azure-openai-api-key": "AZURE_OPENAI_API_KEY",
        "mongodb-uri": "MONGODB_URI",
        "azure-openai-endpoint": "AZURE_OPENAI_ENDPOINT",
        "foundry-connection-str": "AZURE_FOUNDRY_CONNECTION_STRING",
    }

    def __init__(self):
        self._client = None
        self._cache: dict = {}

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not KEYVAULT_URL:
            return None
        try:
            from azure.keyvault.secrets import SecretClient
            from azure.identity import DefaultAzureCredential
            self._client = SecretClient(vault_url=KEYVAULT_URL, credential=DefaultAzureCredential())
            logger.info(f"Azure Key Vault connected: {KEYVAULT_URL}")
            return self._client
        except ImportError:
            logger.warning("azure-keyvault-secrets not installed.")
            return None
        except Exception as e:
            logger.warning(f"Key Vault init failed: {e}. Using .env fallback.")
            return None

    def get_secret(self, secret_name: str) -> Optional[str]:
        if secret_name in self._cache:
            return self._cache[secret_name]
        client = self._get_client()
        if client:
            try:
                value = client.get_secret(secret_name).value
                self._cache[secret_name] = value
                return value
            except Exception as e:
                logger.warning(f"Key Vault secret '{secret_name}' failed: {e}. Using env fallback.")
        env_key = self.SECRET_MAP.get(secret_name, secret_name.upper().replace("-", "_"))
        return os.getenv(env_key, "")

    def load_all(self) -> dict:
        client = self._get_client()
        if client is None:
            return {"mode": "env_fallback", "keyvault": "not_configured"}
        loaded = {}
        for kv_name, env_name in self.SECRET_MAP.items():
            try:
                value = client.get_secret(kv_name).value
                if value:
                    os.environ[env_name] = value
                    self._cache[kv_name] = value
                    loaded[env_name] = "loaded_from_keyvault"
            except Exception as e:
                logger.debug(f"KV secret '{kv_name}' not found: {e}")
        logger.info(f"Key Vault secrets loaded: {list(loaded.keys())}")
        return loaded


secret_manager = KeyVaultSecretManager()
