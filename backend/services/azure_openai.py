import logging
from openai import AzureOpenAI
from backend.config import settings

logger = logging.getLogger(__name__)
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = AzureOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            timeout=15.0,
            max_retries=1,
        )
    return _client


def chat(system_prompt: str, user_message: str, max_tokens: int = 400):
    if not settings.USE_AZURE_OPENAI:
        return None
    try:
        resp = _get_client().chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Azure OpenAI failed: {e}")
        return None
