

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.routes_chat import router as chat_router
from app.api.routes_llm import router as llm_router
from app.api.routes_predict import router as predict_router
from app.api.routes_rag import router as rag_router
from app.api.routes_screening import router as screening_router
from app.core.api_response import api_response
from app.core.config import settings
from app.core.logging_config import configure_logging
from app.core.rate_limit import limiter
from app.core.vietnam_time import VIETNAM_TIMEZONE_NAME, now_vietnam_iso
from app.db.mongo import close_mongo_connection, connect_to_mongo, get_database
from app.rag.retriever import get_retriever_status, initialize_retriever_cache
from app.services.ml_disease_prediction_service import get_model_status, load_disease_model
from app.services.vn_disease_text_service import get_vn_text_model_status, load_vn_text_model

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start and stop production dependencies."""
    await connect_to_mongo()
    load_disease_model()
    try:
        load_vn_text_model()
    except Exception:
        logger.exception("Vietnamese text model could not be loaded; structured fallback remains available")
    retriever_status = initialize_retriever_cache()
    logger.info(
        "%s started in %s mode. RAG cache=%s",
        settings.app_name,
        settings.app_env,
        retriever_status,
    )
    yield
    await close_mongo_connection()
    logger.info("%s stopped", settings.app_name)


app = FastAPI(
    title="Healthcare Chatbot API",
    description="Backend for healthcare chatbot with LLM, RAG++, prediction, screening, and triage",
    version="0.3.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
)

app.include_router(chat_router)
app.include_router(predict_router)
app.include_router(llm_router)
app.include_router(rag_router)
app.include_router(screening_router)


@app.get("/")
def read_root() -> dict:
    """Return a minimal API status payload."""
    return api_response(
        data={
            "message": "Healthcare Chatbot API is running",
            "app_name": settings.app_name,
            "environment": settings.app_env,
        },
        message="API is running",
    )


@app.get("/health")
async def health_check() -> dict:
    """Check API, MongoDB and RAG cache health."""
    try:
        db = get_database()
        await db.command("ping")
        return api_response(
            data={
                "status": "ok",
                "app_name": settings.app_name,
                "environment": settings.app_env,
                "database": settings.mongo_db,
                "timezone": VIETNAM_TIMEZONE_NAME,
                "server_time": now_vietnam_iso(),
                "rag": get_retriever_status(),
                "disease_model": get_model_status(),
                "vietnamese_text_model": get_vn_text_model_status(),
            },
            message="Service health check passed",
        )
    except Exception as exc:
        logger.exception("Health check failed")
        return api_response(
            data={
                "status": "error",
                "app_name": settings.app_name,
                "environment": settings.app_env,
                "database": settings.mongo_db,
                "timezone": VIETNAM_TIMEZONE_NAME,
                "server_time": now_vietnam_iso(),
                "detail": str(exc),
            },
            success=False,
            message="Service health check failed",
            errors=[str(exc)],
        )


@app.get("/db-health")
async def db_health_check() -> dict:
    """List MongoDB collections for operational diagnostics."""
    db = get_database()
    collections = await db.list_collection_names()
    return api_response(
        data={
            "status": "ok",
            "database": db.name,
            "collections": collections,
        },
        message="MongoDB is reachable",
    )
