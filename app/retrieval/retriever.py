"""
TypesenseHybridRetriever — a custom LlamaIndex BaseRetriever.

Overrides _retrieve() to call Typesense hybrid (BM25 + vector) search
instead of LlamaIndex's built-in VectorStoreIndex retrieval.
This makes it a first-class citizen in any LlamaIndex pipeline
(query engines, agents, response synthesisers).
"""
from __future__ import annotations

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from app.config import get_settings
from app.logging_config import get_logger
from app.retrieval.typesense_client import get_typesense_client, hybrid_search

log = get_logger(__name__)


class TypesenseHybridRetriever(BaseRetriever):
    """
    Retrieves document chunks via Typesense hybrid search.

    Args:
        embed_model:   LlamaIndex embedding model (produces query vector).
        top_k:         Number of chunks to retrieve.
        document_ids:  Optional list of document_id strings to scope search.
        score_threshold: Minimum fusion score — hits below this are dropped.
    """

    def __init__(
        self,
        embed_model,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        score_threshold: float | None = None,
    ) -> None:
        self._embed_model     = embed_model
        self._client          = get_typesense_client()
        settings              = get_settings()
        self._top_k           = top_k or settings.top_k_retrieval
        self._document_ids    = document_ids
        self._score_threshold = score_threshold or settings.similarity_threshold
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_text = query_bundle.query_str

        # Embed the query
        embedding = self._embed_model.get_text_embedding(query_text)

        # Hybrid search in Typesense
        hits = hybrid_search(
            client=self._client,
            query=query_text,
            embedding=embedding,
            top_k=self._top_k,
            document_ids=self._document_ids,
        )

        nodes: list[NodeWithScore] = []
        for hit in hits:
            score = float(hit.get("score", 0.0))
            if score < self._score_threshold:
                log.debug(
                    "chunk_below_threshold",
                    chunk_id=hit.get("id"),
                    score=score,
                    threshold=self._score_threshold,
                )
                continue

            node = TextNode(
                text=hit["content"],
                id_=hit["id"],
                metadata={
                    "document_id":  hit.get("document_id", ""),
                    "filename":     hit.get("filename", ""),
                    "chunk_index":  hit.get("chunk_index", 0),
                    "token_count":  hit.get("token_count", 0),
                    "text_score":   hit.get("text_score", 0),
                },
            )
            nodes.append(NodeWithScore(node=node, score=score))

        log.info(
            "retrieval_complete",
            query=query_text[:80],
            hits_total=len(hits),
            hits_above_threshold=len(nodes),
        )
        return nodes
