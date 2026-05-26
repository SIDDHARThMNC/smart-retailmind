import os
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend.config import settings
from backend.services.azure_openai import chat

logger = logging.getLogger(__name__)

_APP_ROOT = os.environ.get("APP_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DOCUMENTS_PATH = os.path.join(_APP_ROOT, "data", "documents")

_tfidf_vectorizer = None
_tfidf_matrix = None
_tfidf_chunks = []

_RAG_SYSTEM = (
    "You are a retail policy assistant. "
    "Answer ONLY using the context provided. "
    "If the answer is not in the context, say: 'I could not find relevant information.' "
    "Do not add information from outside the context."
)


def _load_documents():
    if not os.path.exists(DOCUMENTS_PATH):
        return []
    docs = []
    for fname in sorted(os.listdir(DOCUMENTS_PATH)):
        if fname.endswith(".txt"):
            with open(os.path.join(DOCUMENTS_PATH, fname), encoding="utf-8") as f:
                docs.append({"text": f.read().strip(), "source": fname})
    return docs


def _build_store():
    global _tfidf_vectorizer, _tfidf_matrix, _tfidf_chunks

    docs = _load_documents()
    if not docs:
        logger.warning("No documents found in data/documents/")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = [
        {"text": chunk, "source": doc["source"]}
        for doc in docs
        for chunk in splitter.split_text(doc["text"])
    ]

    _tfidf_chunks = chunks
    _tfidf_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    _tfidf_matrix = _tfidf_vectorizer.fit_transform([c["text"] for c in chunks])
    logger.info(f"TF-IDF ready: {len(chunks)} chunks")


def _ensure_store():
    if _tfidf_matrix is None:
        _build_store()


def search_documents(query: str) -> dict:
    _ensure_store()

    if _tfidf_matrix is not None:
        scores = cosine_similarity(_tfidf_vectorizer.transform([query]), _tfidf_matrix).flatten()
        top = sorted([(i, s) for i, s in enumerate(scores) if s > 0.01], key=lambda x: x[1], reverse=True)[:3]
        if not top:
            return {"query": query, "answer": "No relevant information found.", "sources": []}
        chunks = [_tfidf_chunks[i]["text"] for i, _ in top]
        sources = list(dict.fromkeys(_tfidf_chunks[i]["source"] for i, _ in top))
    else:
        return {"query": query, "answer": "No documents available.", "sources": []}

    context = "\n\n".join(chunks)
    answer = None

    if settings.USE_AZURE_OPENAI:
        try:
            answer = chat(_RAG_SYSTEM, f"Context:\n{context}\n\nQuestion: {query}", max_tokens=300)
        except Exception as e:
            logger.warning(f"Azure OpenAI failed: {e}")

    if not answer:
        answer = chunks[0].strip()[:500]

    logger.info(f"RAG query: '{query}' | sources: {sources}")
    return {"query": query, "answer": answer, "sources": sources}
