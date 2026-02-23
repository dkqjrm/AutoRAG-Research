"""Test cases for Step-back Prompting Retrieval Pipeline.

Tests the Step-back Prompting retrieval pipeline logic using mocked LLM and embedding models.
Step-back Prompting generates a more abstract question from the original query using an LLM,
then embeds both queries and fuses vector search results using Reciprocal Rank Fusion (RRF).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake import FakeListLLM
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.orm.repository.chunk_retrieved_result import ChunkRetrievedResultRepository
from autorag_research.pipelines.retrieval.step_back import (
    DEFAULT_STEP_BACK_PROMPT_TEMPLATE,
    StepBackPipelineConfig,
    StepBackRetrievalPipeline,
)
from tests.autorag_research.pipelines.pipeline_test_utils import (
    PipelineTestConfig,
    PipelineTestVerifier,
)


class TestStepBackPipelineConfig:
    """Tests for StepBackPipelineConfig dataclass."""

    def test_config_get_pipeline_class(self):
        """Test that config returns correct pipeline class."""
        llm = FakeListLLM(responses=["What are the general principles of machine learning?"])
        embedding = FakeEmbeddings(size=768)

        config = StepBackPipelineConfig(
            name="test_step_back",
            llm=llm,
            embedding=embedding,
        )

        assert config.get_pipeline_class() == StepBackRetrievalPipeline

    def test_config_get_pipeline_kwargs(self):
        """Test that config returns correct pipeline kwargs containing llm, embedding, prompt_template, rrf_k."""
        llm = FakeListLLM(responses=["What are the general principles of machine learning?"])
        embedding = FakeEmbeddings(size=768)
        custom_template = "Step back from: {query}"

        config = StepBackPipelineConfig(
            name="test_step_back",
            llm=llm,
            embedding=embedding,
            prompt_template=custom_template,
            rrf_k=30,
        )

        kwargs = config.get_pipeline_kwargs()

        assert kwargs["llm"] is llm
        assert kwargs["embedding"] is embedding
        assert kwargs["prompt_template"] == custom_template
        assert kwargs["rrf_k"] == 30

    def test_config_default_prompt_template(self):
        """Test that config uses default English prompt template."""
        llm = FakeListLLM(responses=["What are the general principles of machine learning?"])
        embedding = FakeEmbeddings(size=768)

        config = StepBackPipelineConfig(
            name="test_step_back",
            llm=llm,
            embedding=embedding,
        )

        assert config.prompt_template == DEFAULT_STEP_BACK_PROMPT_TEMPLATE

    def test_config_default_rrf_k(self):
        """Test that config default rrf_k is 60."""
        llm = FakeListLLM(responses=["What are the general principles of machine learning?"])
        embedding = FakeEmbeddings(size=768)

        config = StepBackPipelineConfig(
            name="test_step_back",
            llm=llm,
            embedding=embedding,
        )

        assert config.rrf_k == 60


class TestStepBackRetrievalPipeline:
    """Tests for StepBackRetrievalPipeline."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that returns predictable step-back questions."""
        return FakeListLLM(responses=["What are the general principles of machine learning?"])

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
        """Test that pipeline _get_pipeline_config returns correct dict."""
        custom_template = "Abstract: {query}"

        pipeline = StepBackRetrievalPipeline(
            session_factory=session_factory,
            name="test_step_back_config",
            llm=mock_llm,
            embedding=mock_embedding,
            prompt_template=custom_template,
            rrf_k=30,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        config = pipeline._get_pipeline_config()
        assert config["type"] == "step_back"
        assert config["prompt_template"] == custom_template
        assert config["rrf_k"] == 30

    @pytest.mark.asyncio
    async def test_generate_step_back_query(
        self,
        session_factory: sessionmaker[Session],
        mock_embedding,
        cleanup_pipeline_results: list[int],
    ):
        """Test that LLM generates abstract step-back question from original query."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="What are the general principles of machine learning?"))

        pipeline = StepBackRetrievalPipeline(
            session_factory=session_factory,
            name="test_step_back_generate_query",
            llm=llm,
            embedding=mock_embedding,
        )
        cleanup_pipeline_results.append(pipeline.pipeline_id)

        result = await pipeline._generate_step_back_query("How does gradient descent work in neural networks?")

        assert result == "What are the general principles of machine learning?"
        llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_single_query(
        self,
        session_factory: sessionmaker[Session],
        mock_llm,
        mock_embedding,
        cleanup_pipeline_results: list[int],
    ):
        """Test single query retrieval via retrieve(), verifying RRF fusion of original + step-back results."""
        original_results = [
            {"doc_id": 1, "score": 0.9, "content": "Neural network content"},
            {"doc_id": 2, "score": 0.8, "content": "Gradient descent content"},
        ]
        step_back_results = [
            {"doc_id": 2, "score": 0.85, "content": "Gradient descent content"},
            {"doc_id": 3, "score": 0.75, "content": "Machine learning principles"},
        ]

        with patch(
            "autorag_research.orm.service.retrieval_pipeline.RetrievalPipelineService.vector_search_by_embedding"
        ) as mock_search:
            mock_search.side_effect = [original_results, step_back_results]

            pipeline = StepBackRetrievalPipeline(
                session_factory=session_factory,
                name="test_step_back_single_retrieve",
                llm=mock_llm,
                embedding=mock_embedding,
            )
            cleanup_pipeline_results.append(pipeline.pipeline_id)

            results = await pipeline.retrieve("How does gradient descent work in neural networks?", top_k=3)

            # RRF fusion: all 3 unique doc_ids should appear (1, 2, 3)
            assert len(results) == 3
            result_doc_ids = {r["doc_id"] for r in results}
            assert result_doc_ids == {1, 2, 3}
            # vector_search_by_embedding should be called twice (original + step-back)
            assert mock_search.call_count == 2

    def test_run_full_pipeline(
        self,
        session_factory: sessionmaker[Session],
        mock_llm,
        mock_embedding,
        cleanup_pipeline_results: list[int],
    ):
        """Test full pipeline run with PipelineTestVerifier."""
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

            pipeline = StepBackRetrievalPipeline(
                session_factory=session_factory,
                name="test_step_back_full_run",
                llm=mock_llm,
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
