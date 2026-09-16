"""
RAG pipeline — orchestrates retrieval, context assembly, and LLM response.

Flow:
  1. TypesenseHybridRetriever fetches top-k chunks (BM25 + vector fusion).
  2. A custom prompt injects retrieved context + conversation history.
  3. Azure OpenAI generates the answer with inline citations.
  4. Citations are parsed and returned as structured SourceCitation objects.
"""
from __future__ import annotations

import time
import uuid
from typing import AsyncGenerator

from llama_index.core import PromptTemplate
from llama_index.core.response_synthesizers import CompactAndRefine

from app.agents.llm_factory import get_embed_model, get_llm
from app.db.schemas import SourceCitation
from app.logging_config import get_logger
from app.retrieval.retriever import TypesenseHybridRetriever

log = get_logger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a knowledgeable assistant with access to a curated document knowledge base.

Guidelines:
- Answer accurately and concisely using the provided context.
- Cite sources using [filename, chunk N] notation inline where relevant.
- If the context does not contain enough information, say so clearly — do not fabricate.
- For follow-up questions, use conversation history to maintain coherence.
- Format structured answers (lists, code) in Markdown.
"""

QA_PROMPT_TMPL = """\
{system_prompt}

--- Conversation history ---
{history}
----------------------------

--- Retrieved context ---
{context_str}
------------------------

User question: {query_str}

Answer:"""

QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)


# ── Helpers ────────────────────────────────────────────────────────────────

def _format_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    lines = []
    for msg in history[-10:]:   # last 10 turns to stay within context window
        role    = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _extract_citations(
    nodes_with_scores,
    threshold: float = 0.0,
) -> list[SourceCitation]:
    citations: list[SourceCitation] = []
    seen: set[str] = set()

    for nws in nodes_with_scores:
        node  = nws.node
        score = float(nws.score or 0)
        key   = f"{node.metadata.get('document_id', '')}:{node.metadata.get('chunk_index', 0)}"
        if key in seen or score < threshold:
            continue
        seen.add(key)
        citations.append(
            SourceCitation(
                document_id=node.metadata.get("document_id", ""),
                filename=node.metadata.get("filename", "unknown"),
                chunk_index=int(node.metadata.get("chunk_index", 0)),
                content_snippet=node.text[:280].replace("\n", " "),
                relevance_score=round(score, 4),
            )
        )
    return sorted(citations, key=lambda c: c.relevance_score, reverse=True)


# ── Main pipeline ──────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Stateless pipeline — one instance per application, reused across requests.
    History is passed in per-call for full session isolation.
    """

    def __init__(self) -> None:
        self._embed_model = get_embed_model()
        self._llm         = get_llm()

    def _make_retriever(
        self,
        top_k: int,
        document_ids: list[str] | None,
    ) -> TypesenseHybridRetriever:
        return TypesenseHybridRetriever(
            embed_model=self._embed_model,
            top_k=top_k,
            document_ids=document_ids,
        )

    def query(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int = 6,
        document_ids: list[str] | None = None,
    ) -> dict:
        """
        Synchronous RAG query.

        Returns:
            {
                "answer":     str,
                "sources":    list[SourceCitation],
                "latency_ms": int,
                "token_count": int | None,
            }
        """
        t0        = time.perf_counter()
        history   = history or []
        retriever = self._make_retriever(top_k, document_ids)

        from llama_index.core.schema import QueryBundle
        query_bundle    = QueryBundle(query_str=question)
        nodes_with_score = retriever.retrieve(query_bundle)

        # Build context string
        context_parts = []
        for i, nws in enumerate(nodes_with_score, 1):
            fname = nws.node.metadata.get("filename", "doc")
            cidx  = nws.node.metadata.get("chunk_index", i)
            context_parts.append(f"[{fname}, chunk {cidx}]\n{nws.node.text}")
        context_str = "\n\n".join(context_parts) if context_parts else "(no relevant context found)"

        prompt = QA_PROMPT.format(
            system_prompt=SYSTEM_PROMPT,
            history=_format_history(history),
            context_str=context_str,
            query_str=question,
        )

        response    = self._llm.complete(prompt)
        answer_text = str(response)
        latency_ms  = int((time.perf_counter() - t0) * 1000)

        # Token count — available on Azure responses
        token_count = None
        if hasattr(response, "raw") and response.raw:
            usage = getattr(response.raw, "usage", None)
            token_count = getattr(usage, "total_tokens", None) if usage else None

        citations = _extract_citations(nodes_with_score)

        log.info(
            "rag_query_complete",
            question=question[:80],
            chunks_retrieved=len(nodes_with_score),
            citations=len(citations),
            latency_ms=latency_ms,
        )

        return {
            "answer":      answer_text,
            "sources":     citations,
            "latency_ms":  latency_ms,
            "token_count": token_count,
        }

    def stream_query(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int = 6,
        document_ids: list[str] | None = None,
    ) -> tuple[any, list[SourceCitation]]:
        """
        Returns (streaming_response, citations).
        Caller is responsible for iterating streaming_response.response_gen.
        """
        history   = history or []
        retriever = self._make_retriever(top_k, document_ids)

        from llama_index.core.schema import QueryBundle
        query_bundle     = QueryBundle(query_str=question)
        nodes_with_score = retriever.retrieve(query_bundle)

        context_parts = []
        for i, nws in enumerate(nodes_with_score, 1):
            fname = nws.node.metadata.get("filename", "doc")
            cidx  = nws.node.metadata.get("chunk_index", i)
            context_parts.append(f"[{fname}, chunk {cidx}]\n{nws.node.text}")
        context_str = "\n\n".join(context_parts) if context_parts else "(no relevant context found)"

        prompt = QA_PROMPT.format(
            system_prompt=SYSTEM_PROMPT,
            history=_format_history(history),
            context_str=context_str,
            query_str=question,
        )

        stream    = self._llm.stream_complete(prompt)
        citations = _extract_citations(nodes_with_score)
        return stream, citations


# Singleton
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
