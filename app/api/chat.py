"""
Chat API endpoints.
- POST /api/chat          — standard request/response
- POST /api/chat/stream   — Server-Sent Events streaming
- GET  /api/sessions      — list sessions
- GET  /api/sessions/{id} — session with messages
- POST /api/sessions      — create session
"""
from __future__ import annotations

import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agents.rag_pipeline import get_pipeline
from app.config import get_settings
from app.db.engine import get_db
from app.db.models import ChatMessage, ChatSession
from app.db.schemas import (
    ChatRequest, ChatResponse, MessageSchema,
    SessionCreate, SessionSchema, SessionWithMessages,
)
from app.logging_config import get_logger

log    = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])


# ── Helpers ────────────────────────────────────────────────────────────────

async def _get_or_create_session(
    session_id: uuid.UUID | None,
    db: AsyncSession,
) -> ChatSession:
    if session_id:
        result  = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
        session = result.scalar_one_or_none()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    session = ChatSession()
    db.add(session)
    await db.flush()
    return session


async def _load_history(session: ChatSession, db: AsyncSession) -> list[dict]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at)
        .limit(20)
    )
    msgs = result.scalars().all()
    return [{"role": m.role, "content": m.content} for m in msgs]


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Standard chat — waits for full response before returning."""
    session  = await _get_or_create_session(req.session_id, db)
    history  = await _load_history(session, db)
    settings = get_settings()
    pipeline = get_pipeline()

    # Persist user message
    user_msg = ChatMessage(session_id=session.id, role="user", content=req.message)
    db.add(user_msg)
    await db.flush()

    result = pipeline.query(
        question=req.message,
        history=history,
        top_k=req.top_k or settings.top_k_retrieval,
    )

    # Persist assistant message
    bot_msg = ChatMessage(
        session_id=session.id,
        role="assistant",
        content=result["answer"],
        sources=[s.model_dump() for s in result["sources"]],
        latency_ms=result["latency_ms"],
        token_count=result["token_count"],
    )
    db.add(bot_msg)

    # Auto-title session from first user message
    if session.title == "New conversation":
        session.title = req.message[:60] + ("…" if len(req.message) > 60 else "")

    await db.commit()
    await db.refresh(bot_msg)

    return ChatResponse(
        message_id=bot_msg.id,
        session_id=session.id,
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=result["latency_ms"],
        token_count=result["token_count"],
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    """Streaming chat via Server-Sent Events."""
    session  = await _get_or_create_session(req.session_id, db)
    history  = await _load_history(session, db)
    pipeline = get_pipeline()
    settings = get_settings()

    user_msg = ChatMessage(session_id=session.id, role="user", content=req.message)
    db.add(user_msg)
    await db.flush()
    await db.commit()

    session_id_str = str(session.id)

    async def event_stream() -> AsyncGenerator[str, None]:
        # Send session_id first so client can track
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id_str})}\n\n"

        t0           = time.perf_counter()
        full_answer  = []
        citations    = []

        try:
            stream, citations = pipeline.stream_query(
                question=req.message,
                history=history,
                top_k=req.top_k or settings.top_k_retrieval,
            )
            for chunk in stream:
                token = chunk.delta or ""
                if token:
                    full_answer.append(token)
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

        except Exception as exc:
            log.error("stream_error", error=str(exc))
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
            return

        # Final metadata event
        latency_ms  = int((time.perf_counter() - t0) * 1000)
        answer_text = "".join(full_answer)
        sources_raw = [s.model_dump() for s in citations]

        yield f"data: {json.dumps({'type': 'done', 'sources': sources_raw, 'latency_ms': latency_ms})}\n\n"

        # Persist in background (best-effort)
        try:
            async with db.begin():
                bot_msg = ChatMessage(
                    session_id=session.id,
                    role="assistant",
                    content=answer_text,
                    sources=sources_raw,
                    latency_ms=latency_ms,
                )
                db.add(bot_msg)
        except Exception as exc:
            log.warning("stream_persist_failed", error=str(exc))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Sessions ───────────────────────────────────────────────────────────────

@router.get("/sessions", response_model=list[SessionSchema])
async def list_sessions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.is_active == True)
        .order_by(ChatSession.updated_at.desc())
        .limit(50)
    )
    sessions = result.scalars().all()
    out = []
    for s in sessions:
        count_result = await db.execute(
            select(func.count()).where(ChatMessage.session_id == s.id)
        )
        count = count_result.scalar() or 0
        schema = SessionSchema.model_validate(s)
        schema.message_count = count
        out.append(schema)
    return out


@router.post("/sessions", response_model=SessionSchema)
async def create_session(body: SessionCreate, db: AsyncSession = Depends(get_db)):
    session = ChatSession(title=body.title)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    schema = SessionSchema.model_validate(session)
    schema.message_count = 0
    return schema


@router.get("/sessions/{session_id}", response_model=SessionWithMessages)
async def get_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ChatSession)
        .options(selectinload(ChatSession.messages))
        .where(ChatSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session.is_active = False
    await db.commit()
