"""Test cases for Reranker Retrieval Pipeline.

Tests the two-stage reranker pipeline: vector search + cross-encoder reranking.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.embeddings import FakeEmbeddings
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.pipelines.retrieval.reranker import (
    RerankerPipelineConfig,
    RerankerRetrievalPipeline,
)
from autorag_research.rerankers.base import RerankResult
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
)

logger = logging.getLogger("AutoRAG-Research")


class TestRerankerPipelineConfig:
    """Tests for RerankerPipelineConfig dataclass."""

    def test_config_get_pipeline_class(self):
        """Test that config returns correct pipeline class."""
        embedding = FakeEmbeddings(size=768)
        reranker = MagicMock()

        config = RerankerPipelineConfig(
            name="test_reranker",
            embedding=embedding,
            reranker=reranker,
        )

        assert config.get_pipeline_class() == RerankerRetrievalPipeline

    def test_config_get_pipeline_kwargs(self):
        """Test that config returns correct pipeline kwargs."""
        embedding = FakeEmbeddings(size=768)
        reranker = MagicMock()

        config = RerankerPipelineConfig(
            name="test_reranker",
            embedding=embedding,
            reranker=reranker,
            initial_top_k=100,
        )

        kwargs = config.get_pipeline_kwargs()
        assert kwargs["embedding"] is embedding
        assert kwargs["reranker"] is reranker
        assert kwargs["initial_top_k"] == 100

    def test_config_default_initial_top_k(self):
        """Test that config uses default initial_top_k of 50."""
        embedding = FakeEmbeddings(size=768)
        reranker = MagicMock()

        config = RerankerPipelineConfig(
            name="test_reranker",
            embedding=embedding,
            reranker=reranker,
        )

        assert config.initial_top_k == 50


class TestRerankerRetrievalPipeline:
    """Tests for RerankerRetrievalPipeline."""

    @pytest.fixture
    def mock_embedding(self):
        return FakeEmbeddings(size=768)

    @pytest.fixture
    def mock_reranker(self):
        reranker = MagicMock()
        reranker.arerank = AsyncMock(
            return_value=[
                RerankResult(index=1, text="Content 2", score=0.95),
                RerankResult(index=0, text="Content 1", score=0.85),
            ]
        )
        return reranker

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory: sessionmaker[Session]):
        created_pipeline_ids: list[int] = []
        yield created_pipeline_ids
        session = session_factory()
        try:
            result_repo = ChunkRetrievedResultRepository(session)
            for pipeline_id in created_pipeline_ids:
                result_repo.delete_by_pipeline(pipeline_id)
            session.commit()
        finally:
            session.close()

    def test_pipeline_config(
        self,
        session_factory: sessionmaker[Session],
        mock_embedding,
        mock_reranker,
        cleanup_pipeline_results: list[int],
    ):
        """Test that pipeline config stores correct type and initial_top_k."""
        pipeline = RerankerRetrievalPipeline(
            session_factory=session_factory,
            name="test_reranker_config",
            embedding=mock_embedding,
            reranker=mock_reranker,
            initial_top_k=30,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()
        assert config["type"] == "reranker"
        assert config["initial_top_k"] == 30

    def test_pipeline_config_default_initial_top_k(
        self,
        session_factory: sessionmaker[Session],
        mock_embedding,
        mock_reranker,
        cleanup_pipeline_results: list[int],
    ):
        """Test that pipeline config defaults to initial_top_k=50."""
        pipeline = RerankerRetrievalPipeline(
            session_factory=session_factory,
            name="test_reranker_default_topk",
            embedding=mock_embedding,
            reranker=mock_reranker,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()
        assert config["initial_top_k"] == 50

    @pytest.mark.asyncio
    async def test_retrieve_single_query(
        self,
        session_factory: sessionmaker[Session],
        mock_embedding,
        mock_reranker,
        cleanup_pipeline_results: list[int],
    ):
        """Test single query retrieval with two-stage reranking."""
        mock_initial_results = [
            {"doc_id": 1, "score": 0.9, "content": "Content 1"},
            {"doc_id": 2, "score": 0.8, "content": "Content 2"},
            {"doc_id": 3, "score": 0.7, "content": "Content 3"},
        ]

        with patch(
            "autorag_research.orm.service.retrieval_pipeline.RetrievalPipelineService.vector_search_by_embedding"
        ) as mock_search:
            mock_search.return_value = mock_initial_results

            pipeline = RerankerRetrievalPipeline(
                session_factory=session_factory,
                name="test_reranker_single",
                embedding=mock_embedding,
                reranker=mock_reranker,
            )
            cleanup_pipeline_results.append(pipeline.pipeline_id)

            results = await pipeline.retrieve("What is deep learning?", top_k=2)

            assert len(results) == 2
            # Reranker reorders: index=1 (doc_id=2) first with score 0.95, index=0 (doc_id=1) second with score 0.85
            assert results[0]["doc_id"] == 2
            assert results[0]["score"] == 0.95
            assert results[1]["doc_id"] == 1
            assert results[1]["score"] == 0.85

    @pytest.mark.asyncio
    async def test_retrieve_empty_initial_results(
        self,
        session_factory: sessionmaker[Session],
        mock_embedding,
        mock_reranker,
        cleanup_pipeline_results: list[int],
    ):
        """Test that retrieve returns empty list when vector search returns nothing."""
        with patch(
            "autorag_research.orm.service.retrieval_pipeline.RetrievalPipelineService.vector_search_by_embedding"
        ) as mock_search:
            mock_search.return_value = []

            pipeline = RerankerRetrievalPipeline(
                session_factory=session_factory,
                name="test_reranker_empty",
                embedding=mock_embedding,
                reranker=mock_reranker,
            )
            cleanup_pipeline_results.append(pipeline.pipeline_id)

            results = await pipeline.retrieve("What is deep learning?", top_k=5)

            assert results == []
            # Reranker should not be called when no initial results
            mock_reranker.arerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrieve_calls_reranker_with_doc_texts(
        self,
        session_factory: sessionmaker[Session],
        mock_embedding,
        mock_reranker,
        cleanup_pipeline_results: list[int],
    ):
        """Test that reranker is called with correct query and document texts."""
        mock_initial_results = [
            {"doc_id": 10, "score": 0.9, "content": "First document text"},
            {"doc_id": 20, "score": 0.8, "content": "Second document text"},
        ]

        with patch(
            "autorag_research.orm.service.retrieval_pipeline.RetrievalPipelineService.vector_search_by_embedding"
        ) as mock_search:
            mock_search.return_value = mock_initial_results

            pipeline = RerankerRetrievalPipeline(
                session_factory=session_factory,
                name="test_reranker_calls",
                embedding=mock_embedding,
                reranker=mock_reranker,
            )
            cleanup_pipeline_results.append(pipeline.pipeline_id)

            query = "What is machine learning?"
            await pipeline.retrieve(query, top_k=2)

            mock_reranker.arerank.assert_called_once_with(
                query,
                ["First document text", "Second document text"],
                top_k=2,
            )

    def test_run_full_pipeline(
        self,
        session_factory: sessionmaker[Session],
        mock_embedding,
        mock_reranker,
        cleanup_pipeline_results: list[int],
    ):
        """Test running the full pipeline batch across all queries."""
        from autorag_research.orm.repository.query import QueryRepository

        session = session_factory()
        try:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()
        finally:
            session.close()

        mock_result = [
            {"doc_id": 1, "score": 0.9, "content": "Content 1"},
            {"doc_id": 2, "score": 0.8, "content": "Content 2"},
        ]

        with patch(
            "autorag_research.orm.service.retrieval_pipeline.RetrievalPipelineService.vector_search_by_embedding"
        ) as mock_search:
            mock_search.return_value = mock_result

            pipeline = RerankerRetrievalPipeline(
                session_factory=session_factory,
                name="test_reranker_full_run",
                embedding=mock_embedding,
                reranker=mock_reranker,
            )
            cleanup_pipeline_results.append(pipeline.pipeline_id)

            result = pipeline.run(top_k=3)

            config = PipelineTestConfig(
                pipeline_type="retrieval",
                expected_total_queries=query_count,
                expected_min_results=0,
                check_persistence=True,
            )
            verifier = PipelineTestVerifier(result, pipeline.pipeline_id, session_factory, config)
            verifier.verify_all()
