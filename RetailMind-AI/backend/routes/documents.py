import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.rag.rag_service import search_documents
from backend.services.azure_openai import chat
from backend.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


class SearchRequest(BaseModel):
    query: str


@router.post("/api/search")
async def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        local = search_documents(req.query)

        gpt_answer = None
        if settings.USE_AZURE_OPENAI and local.get("answer"):
            try:
                sources_str = ", ".join(local.get("sources", [])) or "policy documents"
                system = (
                    "You are a knowledgeable retail policy advisor. "
                    "You are given text retrieved from official retail policy documents. "
                    "Your job is to synthesize this retrieved content into a clear, direct, helpful answer. "
                    "Rules: "
                    "1. Answer ONLY using the retrieved context — do not add outside knowledge. "
                    "2. If the context fully answers the question, give a complete structured answer. "
                    "3. If the context partially answers, state what is covered and what is not found. "
                    "4. Use plain language — no jargon. Be specific about conditions, timeframes, and exceptions. "
                    "5. If relevant, mention which policy document the information comes from."
                )
                user_msg = (
                    f"Retrieved context (from: {sources_str}):\n{local['answer']}\n\n"
                    f"User question: {req.query}\n\n"
                    f"Provide a complete, well-structured answer based strictly on the retrieved context."
                )
                gpt_answer = chat(system, user_msg, max_tokens=300)
            except Exception as e:
                logger.warning(f"GPT enhancement failed for /api/search: {e}")

        if gpt_answer:
            local["gpt_answer"] = gpt_answer
            local["gpt_enhanced"] = True
        else:
            local["gpt_enhanced"] = False

        return local
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
