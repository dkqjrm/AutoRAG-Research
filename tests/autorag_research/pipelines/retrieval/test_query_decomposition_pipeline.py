"""Test cases for Query Decomposition Retrieval Pipeline.

Tests the Query Decomposition retrieval pipeline logic using mocked LLM and embedding models.
The pipeline decomposes complex queries into sub-queries using an LLM, embeds each sub-query,
performs vector search, and fuses results using Reciprocal Rank Fusion (RRF).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake import FakeListLLM
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.pipelines.retrieval.query_decomposition import (
    DEFAULT_QUERY_DECOMPOSITION_PROMPT_TEMPLATE_KOREAN,
    QueryDecompositionPipelineConfig,
    QueryDecompositionRetrievalPipeline,
)
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
)


class TestQueryDecompositionPipelineConfig:
    """Tests for QueryDecompositionPipelineConfig dataclass."""

    def test_config_get_pipeline_class(self):
        """Test that config returns correct pipeline class."""
        llm = FakeListLLM(responses=["Sub question 1\nSub question 2"])
        embedding = FakeEmbeddings(size=768)

        config = QueryDecompositionPipelineConfig(
            name="test_qd",
            llm=llm,
            embedding=embedding,
        )

        assert config.get_pipeline_class() == QueryDecompositionRetrievalPipeline

    def test_config_get_pipeline_kwargs(self):
        """Test that config returns correct pipeline kwargs."""
        llm = FakeListLLM(responses=["Sub question 1\nSub question 2"])
        embedding = FakeEmbeddings(size=768)
        custom_template = "Decompose: {query} into {max_sub_queries} parts"

        config = QueryDecompositionPipelineConfig(
            name="test_qd",
            llm=llm,
            embedding=embedding,
            prompt_template=custom_template,
            max_sub_queries=3,
            rrf_k=30,
        )

        kwargs = config.get_pipeline_kwargs()

        assert kwargs["llm"] is llm
        assert kwargs["embedding"] is embedding
        assert kwargs["prompt_template"] == custom_template
        assert kwargs["max_sub_queries"] == 3
        assert kwargs["rrf_k"] == 30

    def test_config_default_prompt_template(self):
        """Test that config uses default Korean prompt template."""
        llm = FakeListLLM(responses=["Sub question 1\nSub question 2"])
        embedding = FakeEmbeddings(size=768)

        config = QueryDecompositionPipelineConfig(
            name="test_qd",
            llm=llm,
            embedding=embedding,
        )

        assert config.prompt_template == DEFAULT_QUERY_DECOMPOSITION_PROMPT_TEMPLATE_KOREAN

    def test_config_default_max_sub_queries(self):
        """Test that config uses default max_sub_queries of 4."""
        llm = FakeListLLM(responses=["Sub question 1\nSub question 2"])
        embedding = FakeEmbeddings(size=768)

        config = QueryDecompositionPipelineConfig(
            name="test_qd",
            llm=llm,
            embedding=embedding,
        )

        assert config.max_sub_queries == 4


class TestQueryDecompositionRetrievalPipeline:
    """Tests for QueryDecompositionRetrievalPipeline."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that returns predictable sub-queries."""
        return FakeListLLM(responses=["Sub question 1\nSub question 2\nSub question 3"])

    @pytest.fixture
    def mock_embedding(self):
        """Create a mock embedding model that returns consistent embeddings."""
        return FakeEmbeddings(size=768)

    @pytest.fixture
    def cleanup_pipeline_results(self, session_factory: sessionmaker[Session]):
        """Cleanup fixture that deletes pipeline results after test."""
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
        mock_llm,
        mock_embedding,
        cleanup_pipeline_results: list[int],
    ):
        """Test that pipeline config returns correct dict."""
        custom_template = "Break down: {query} (max {max_sub_queries})"

        pipeline = QueryDecompositionRetrievalPipeline(
            session_factory=session_factory,
            name="test_qd_config",
            llm=mock_llm,
            embedding=mock_embedding,
            prompt_template=custom_template,
            max_sub_queries=3,
            rrf_k=30,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()
        assert config["type"] == "query_decomposition"
        assert config["max_sub_queries"] == 3
        assert config["rrf_k"] == 30
        assert config["prompt_template"] == custom_template

    @pytest.mark.asyncio
    async def test_decompose_query(
        self,
        session_factory: sessionmaker[Session],
        mock_embedding,
        cleanup_pipeline_results: list[int],
    ):
        """Test that LLM generates sub-queries parsed line-by-line."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="Sub question 1\nSub question 2\nSub question 3"))

        pipeline = QueryDecompositionRetrievalPipeline(
            session_factory=session_factory,
            name="test_qd_decompose",
            llm=llm,
            embedding=mock_embedding,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        sub_queries = await pipeline._decompose_query("What is machine learning?")

        assert len(sub_queries) == 3
        assert "Sub question 1" in sub_queries
        assert "Sub question 2" in sub_queries
        assert "Sub question 3" in sub_queries

    @pytest.mark.asyncio
    async def test_retrieve_single_query(
        self,
        session_factory: sessionmaker[Session],
        mock_embedding,
        cleanup_pipeline_results: list[int],
    ):
        """Test single query retrieval via retrieve() method with mocked vector search."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="Sub question 1\nSub question 2"))

        mock_result = [
            {"doc_id": 1, "score": 0.9, "content": "Content 1"},
            {"doc_id": 2, "score": 0.8, "content": "Content 2"},
        ]

        with patch(
            "autorag_research.orm.service.retrieval_pipeline.RetrievalPipelineService.vector_search_by_embedding"
        ) as mock_search:
            mock_search.return_value = mock_result

            pipeline = QueryDecompositionRetrievalPipeline(
                session_factory=session_factory,
                name="test_qd_single_retrieve",
                llm=llm,
                embedding=mock_embedding,
            )
            cleanup_pipeline_results.append(pipeline.pipeline_id)

            results = await pipeline.retrieve("What is deep learning?", top_k=5)

            assert isinstance(results, list)
            assert len(results) > 0
            assert all("doc_id" in r and "score" in r for r in results)

    def test_run_full_pipeline(
        self,
        session_factory: sessionmaker[Session],
        mock_embedding,
        cleanup_pipeline_results: list[int],
    ):
        """Test running the full pipeline with mocked components."""
        from autorag_research.orm.repository.query import QueryRepository

        # Count actual queries in database
        session = session_factory()
        try:
            query_repo = QueryRepository(session)
            query_count = query_repo.count()
        finally:
            session.close()

        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="Sub question 1\nSub question 2\nSub question 3"))

        mock_result = [
            {"doc_id": 1, "score": 0.9, "content": "Content 1"},
            {"doc_id": 2, "score": 0.8, "content": "Content 2"},
        ]

        with patch(
            "autorag_research.orm.service.retrieval_pipeline.RetrievalPipelineService.vector_search_by_embedding"
        ) as mock_search:
            mock_search.return_value = mock_result

            pipeline = QueryDecompositionRetrievalPipeline(
                session_factory=session_factory,
                name="test_qd_full_run",
                llm=llm,
                embedding=mock_embedding,
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
