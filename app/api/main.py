"""
FastAPI application — API backend.
Runs on port 8000, consumed by the Dash frontend on port 8050.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.llm_factory import configure_llama_settings
from app.api.chat import router as chat_router
from app.api.documents import router as docs_router
from app.api.health import router as health_router
from app.db.engine import create_all_tables
from app.logging_config import configure_logging, get_logger
from app.retrieval.typesense_client import ensure_collection, get_typesense_client

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log.info("startup_begin")

    # PostgreSQL — create tables
    await create_all_tables()
    log.info("postgres_tables_ready")

    # Typesense — ensure collection exists
    ts = get_typesense_client()
    ensure_collection(ts)
    log.info("typesense_collection_ready")

    # LlamaIndex — configure LLM + embed model
    configure_llama_settings()
    log.info("llama_settings_ready")

    log.info("startup_complete")
    yield
    log.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Chatbot API",
        description="LlamaIndex + Azure OpenAI + Typesense + PostgreSQL",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8050", "http://127.0.0.1:8050"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_router)
    app.include_router(docs_router)
    app.include_router(health_router)

    return app


app = create_app()


from fastapi import Request
from fastapi.responses import JSONResponse
import traceback

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print(f"UNHANDLED ERROR:\n{tb}")
    return JSONResponse(status_code=500, content={"detail": str(exc), "traceback": tb})
