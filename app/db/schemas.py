"""
Pydantic v2 schemas — API request/response contracts.
Kept separate from ORM models to maintain a clean boundary.
"""
import uuid
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


# ── Source citation ────────────────────────────────────────────────────────

class SourceCitation(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    content_snippet: str = Field(..., max_length=300)
    relevance_score: float = Field(..., ge=0.0, le=1.0)


# ── Chat ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: Optional[uuid.UUID] = None
    message: str = Field(..., min_length=1, max_length=8192)
    stream: bool = False
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class ChatResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    message_id: uuid.UUID
    session_id: uuid.UUID
    answer: str
    sources: list[SourceCitation] = []
    latency_ms: int
    token_count: Optional[int] = None


class MessageSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    created_at: datetime
    sources: Optional[list[SourceCitation]] = None
    latency_ms: Optional[int] = None


# ── Sessions ───────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    title: str = Field(default="New conversation", max_length=255)


class SessionSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    is_active: bool
    message_count: int = 0


class SessionWithMessages(SessionSchema):
    messages: list[MessageSchema] = []


# ── Documents ──────────────────────────────────────────────────────────────

class DocumentSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    file_type: str
    file_size_bytes: int
    chunk_count: int
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    indexed_at: Optional[datetime] = None


class DocumentUploadResponse(BaseModel):
    document_id: uuid.UUID
    filename: str
    status: str
    message: str


# ── Status / Health ────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    services: dict[str, bool]


class StatsResponse(BaseModel):
    total_sessions: int
    total_messages: int
    total_documents: int
    total_chunks: int
    avg_latency_ms: Optional[float] = None
