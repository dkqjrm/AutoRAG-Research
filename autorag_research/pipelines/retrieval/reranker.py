"""Reranker Retrieval Pipeline for AutoRAG-Research.

Implements a two-stage retrieval pipeline:
1. Initial vector search to retrieve a broad set of candidate chunks
2. Cross-encoder reranking to re-score and re-rank the candidates

This approach improves precision by using a more expensive but more accurate
cross-encoder model to rerank results from a cheaper first-stage retriever.
"""

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.embeddings import Embeddings
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.injection import health_check_embedding
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline
from autorag_research.rerankers.base import BaseReranker

logger = logging.getLogger("AutoRAG-Research")


@dataclass(kw_only=True)
class RerankerPipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for two-stage reranker retrieval pipeline.

    Attributes:
        name: Unique name for this pipeline instance.
        embedding: Embedding config name or instance for initial vector search.
        reranker: Reranker class name string or BaseReranker instance.
        reranker_model_name: Model name to use when reranker is a string class name.
        initial_top_k: Number of candidates to fetch in the first stage.
        top_k: Number of results to return after reranking.
        batch_size: Number of queries to process in each batch.

    Example:
        ```python
        config = RerankerPipelineConfig(
            name="reranker",
            embedding="huggingface",
            reranker="SentenceTransformerReranker",
            reranker_model_name="BAAI/bge-reranker-v2-m3",
            initial_top_k=50,
            top_k=10,
        )
        ```
    """

    embedding: str | Embeddings
    """Embedding model for initial vector search. Can be config name or instance."""

    reranker: str | BaseReranker
    """Reranker instance or class name string (e.g., 'SentenceTransformerReranker')."""

    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    """Model name to use when reranker is specified as a class name string."""

    initial_top_k: int = 50
    """Number of candidate documents to fetch in the first-stage vector search."""

    def __setattr__(self, name: str, value: Any) -> None:
        """Auto-convert string config names to model instances."""
        if name == "embedding" and isinstance(value, str):
            from autorag_research.injection import load_embedding_model

            value = load_embedding_model(value)
            health_check_embedding(value)
        elif name == "reranker" and isinstance(value, str):
            import importlib

            reranker_model_name = (
                self.reranker_model_name if hasattr(self, "reranker_model_name") else "BAAI/bge-reranker-v2-m3"
            )
            rerankers_module = importlib.import_module("autorag_research.rerankers")
            reranker_cls = getattr(rerankers_module, value)
            value = reranker_cls(model_name=reranker_model_name)
        super().__setattr__(name, value)

    def get_pipeline_class(self) -> type["RerankerRetrievalPipeline"]:
        """Return the RerankerRetrievalPipeline class."""
        return RerankerRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for RerankerRetrievalPipeline constructor."""
        return {
            "embedding": self.embedding,
            "reranker": self.reranker,
            "initial_top_k": self.initial_top_k,
        }


class RerankerRetrievalPipeline(BaseRetrievalPipeline):
    """Two-stage retrieval pipeline using vector search followed by cross-encoder reranking.

    Stage 1: Embed the query and retrieve initial_top_k candidates via vector search.
    Stage 2: Rerank candidates using a cross-encoder model and return top_k results.

    Example:
        ```python
        from autorag_research.orm.connection import DBConnection
        from autorag_research.injection import load_embedding_model
        from autorag_research.rerankers.sentence_transformer import SentenceTransformerReranker
        from autorag_research.pipelines.retrieval.reranker import RerankerRetrievalPipeline

        db = DBConnection.from_config()
        session_factory = db.get_session_factory()

        pipeline = RerankerRetrievalPipeline(
            session_factory=session_factory,
            name="reranker",
            embedding=load_embedding_model("huggingface"),
            reranker=SentenceTransformerReranker(model_name="BAAI/bge-reranker-v2-m3"),
            initial_top_k=50,
        )

        # Single query retrieval
        results = await pipeline.retrieve("What is machine learning?", top_k=10)

        # Batch processing
        stats = pipeline.run(top_k=10)
        ```
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        name: str,
        embedding: Embeddings,
        reranker: BaseReranker,
        initial_top_k: int = 50,
        schema: Any | None = None,
    ):
        """Initialize reranker retrieval pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            embedding: LangChain embeddings model for initial vector search.
            reranker: BaseReranker instance for cross-encoder reranking.
            initial_top_k: Number of candidates to retrieve in the first stage.
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        # Store parameters BEFORE calling super().__init__
        # because _get_pipeline_config() is called in parent init
        self.embedding = embedding
        self.reranker = reranker
        self.initial_top_k = initial_top_k

        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return reranker pipeline configuration.

        Returns:
            Dictionary containing pipeline configuration for storage.
        """
        return {
            "type": "reranker",
            "reranker_model": getattr(self.reranker, "model_name", str(type(self.reranker).__name__)),
            "initial_top_k": self.initial_top_k,
        }

    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve documents using query ID.

        Fetches query text from DB, then delegates to _retrieve_by_text.

        Args:
            query_id: The query ID to retrieve for.
            top_k: Number of top documents to retrieve.

        Returns:
            List of result dicts with doc_id and score.
        """
        query_texts = self._service.fetch_query_texts([query_id])
        if not query_texts:
            return []

        query_text = query_texts[0]
        return await self._retrieve_by_text(query_text, top_k)

    async def _retrieve_by_text(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve documents using raw query text with two-stage retrieval.

        Stage 1: Embed query and retrieve initial_top_k candidates via vector search.
        Stage 2: Rerank candidates with cross-encoder and return top_k results.

        Args:
            query_text: The query text to retrieve for.
            top_k: Number of top documents to return after reranking.

        Returns:
            List of result dicts with doc_id and score.
        """
        # Stage 1: Embed query and retrieve initial candidates
        embedding = await self.embedding.aembed_query(query_text)
        initial_results = self._service.vector_search_by_embedding(
            embedding=embedding,
            top_k=self.initial_top_k,
        )

        if not initial_results:
            return []

        # Extract doc texts from results (vector_search_by_embedding includes content)
        doc_ids = [r["doc_id"] for r in initial_results]
        doc_texts = [r.get("content") or "" for r in initial_results]

        logger.debug(
            "Reranker stage 1: retrieved %d candidates for query '%s'",
            len(initial_results),
            query_text[:50],
        )

        # Stage 2: Rerank with cross-encoder
        reranked = await self.reranker.arerank(query_text, doc_texts, top_k=top_k)

        logger.debug(
            "Reranker stage 2: reranked to %d results",
            len(reranked),
        )

        # Map reranked results back to original doc_ids
        return [{"doc_id": doc_ids[result.index], "score": result.score} for result in reranked]


__all__ = ["RerankerPipelineConfig", "RerankerRetrievalPipeline"]
