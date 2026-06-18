"""RAG diagnostic and retrieval routes."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.api_response import api_response
from app.rag.explainer import build_rag_explanation
from app.rag.retriever import (
    build_vector_db_payload,
    get_retriever_status,
    retrieve_relevant_documents,
)

router = APIRouter(prefix="/rag", tags=["rag"])


class RAGTestRequest(BaseModel):
    """Manual retrieval request for testing RAG evidence matching."""

    symptoms: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    diseases: List[str] = Field(default_factory=list)
    department: Optional[str] = None
    top_k: int = 3


@router.post("/retrieve")
async def rag_retrieve_endpoint(request: RAGTestRequest) -> dict:
    """Retrieve relevant evidence from the cached RAG index."""
    docs = retrieve_relevant_documents(
        symptoms=request.symptoms,
        red_flags=request.red_flags,
        department=request.department,
        diseases=request.diseases,
        top_k=request.top_k,
    )
    explanation = build_rag_explanation(docs)

    return api_response(
        data={
            "matched_documents": docs,
            "explanation": explanation,
        },
        message="RAG documents retrieved",
        metadata={"retrieval_backend": "cached_lexical_hybrid"},
    )


@router.get("/status")
async def rag_status_endpoint() -> dict:
    """Return retriever cache and VectorDB preparation status."""
    return api_response(data=get_retriever_status(), message="RAG status loaded")


@router.post("/vector-payload")
async def rag_vector_payload_endpoint() -> dict:
    """Prepare JSONL payload for a future FAISS or ChromaDB build job."""
    output_path = Path("data/processed/vector_db_payload.jsonl")
    payload = build_vector_db_payload(output_path=output_path)
    return api_response(
        data={
            "output_path": str(output_path),
            "documents": len(payload),
            "backend": "faiss_or_chromadb_ready_payload",
        },
        message="VectorDB preparation payload generated",
    )
