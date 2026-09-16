"""
Typesense integration — keyword + vector hybrid search.
Falls back to keyword-only if vector search fails.
"""
from __future__ import annotations

import typesense
from typesense.exceptions import ObjectNotFound
from app.config import get_settings
from app.logging_config import get_logger

log = get_logger(__name__)

COLLECTION_NAME = "document_chunks"

COLLECTION_SCHEMA = {
    "name": COLLECTION_NAME,
    "fields": [
        {"name": "id",            "type": "string"},
        {"name": "document_id",   "type": "string", "facet": True},
        {"name": "filename",      "type": "string", "facet": True},
        {"name": "chunk_index",   "type": "int32"},
        {"name": "content",       "type": "string"},
        {"name": "token_count",   "type": "int32"},
        {
            "name":    "embedding",
            "type":    "float[]",
            "num_dim": 3072,
            "index":   True,
        },
    ],
    "default_sorting_field": "chunk_index",
}


def get_typesense_client() -> typesense.Client:
    settings = get_settings()
    return typesense.Client({
        "nodes": [{
            "host":     settings.typesense_host,
            "port":     str(settings.typesense_port),
            "protocol": settings.typesense_protocol,
        }],
        "api_key":                  settings.typesense_api_key,
        "connection_timeout_seconds": 10,
    })


def ensure_collection(client: typesense.Client) -> None:
    try:
        client.collections[COLLECTION_NAME].retrieve()
        log.info("typesense_collection_exists", collection=COLLECTION_NAME)
    except ObjectNotFound:
        client.collections.create(COLLECTION_SCHEMA)
        log.info("typesense_collection_created", collection=COLLECTION_NAME)


def upsert_chunks(client: typesense.Client, chunks: list[dict]) -> None:
    if not chunks:
        return
    result = client.collections[COLLECTION_NAME].documents.import_(
        chunks, {"action": "upsert"}
    )
    errors = [r for r in result if not r.get("success")]
    if errors:
        log.warning("typesense_upsert_errors", count=len(errors), sample=errors[:3])
    log.info("typesense_upsert_ok", count=len(chunks) - len(errors))


def delete_document_chunks(client: typesense.Client, document_id: str) -> int:
    result = client.collections[COLLECTION_NAME].documents.delete(
        {"filter_by": f"document_id:={document_id}"}
    )
    deleted = result.get("num_deleted", 0)
    log.info("typesense_chunks_deleted", document_id=document_id, count=deleted)
    return deleted


def hybrid_search(
    client: typesense.Client,
    query: str,
    embedding: list[float],
    top_k: int = 6,
    document_ids: list[str] | None = None,
) -> list[dict]:
    """
    Hybrid search: tries vector+keyword, falls back to keyword-only.
    """
    filter_by = ""
    if document_ids:
        ids_str   = ",".join(document_ids)
        filter_by = f"document_id:[{ids_str}]"

    # Try vector search first
    try:
        embedding_str = ",".join(str(v) for v in embedding)
        params = {
            "q":              query,
            "query_by":       "content",
            "vector_query":   f"embedding:([{embedding_str}], k:{top_k})",
            "per_page":       top_k,
            "exclude_fields": "embedding",
        }
        if filter_by:
            params["filter_by"] = filter_by

        result = client.collections[COLLECTION_NAME].documents.search(params)
        hits   = result.get("hits", [])
        if hits:
            log.info("typesense_vector_search_ok", hits=len(hits))
            return [
                {**h["document"], "score": h.get("vector_distance", 0.5)}
                for h in hits
            ]
    except Exception as exc:
        log.warning("typesense_vector_search_failed", error=str(exc))

    # Fallback: keyword-only BM25 search
    try:
        params = {
            "q":              query,
            "query_by":       "content",
            "per_page":       top_k,
            "exclude_fields": "embedding",
        }
        if filter_by:
            params["filter_by"] = filter_by

        result = client.collections[COLLECTION_NAME].documents.search(params)
        hits   = result.get("hits", [])
        log.info("typesense_keyword_search_ok", hits=len(hits))
        return [
            {**h["document"], "score": 0.75}
            for h in hits
        ]
    except Exception as exc:
        log.error("typesense_keyword_search_failed", error=str(exc))
        return []
