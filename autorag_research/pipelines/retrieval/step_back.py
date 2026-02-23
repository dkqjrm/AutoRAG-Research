"""Step-back Prompting Retrieval Pipeline for AutoRAG-Research.

This pipeline implements the Step-back Prompting approach from the paper
"Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models"
(Zheng et al., 2023).

Step-back Prompting works by:
1. Using an LLM to generate a more abstract "step-back" question from the original query
2. Embedding both the original query and the step-back query
3. Performing vector search for both embeddings
4. Fusing results using Reciprocal Rank Fusion (RRF)

This helps retrieve background knowledge and principles needed to answer the original question.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.injection import health_check_embedding, health_check_llm
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline

logger = logging.getLogger("AutoRAG-Research")

DEFAULT_STEP_BACK_PROMPT_TEMPLATE_KOREAN = """다음 질문에 대해 한 단계 물러서서, 더 일반적이고 추상적인 질문을 생성하세요.
이 추상적인 질문은 원래 질문에 답하는 데 필요한 배경 지식이나 원리를 찾는 데 도움이 되어야 합니다.
추상적인 질문만 한 줄로 출력하세요.

원래 질문: {query}
추상적 질문:"""

DEFAULT_STEP_BACK_PROMPT_TEMPLATE = """Given the following question, take a step back and generate a more general, abstract question.
This abstract question should help find background knowledge or principles needed to answer the original question.
Output only the abstract question in a single line.

Original question: {query}
Abstract question:"""


@dataclass(kw_only=True)
class StepBackPipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for Step-back Prompting retrieval pipeline.

    Attributes:
        name: Unique name for this pipeline instance.
        llm: LLM config name or instance for generating step-back questions.
        embedding: Embedding config name or instance for embedding queries.
        prompt_template: Template with {query} placeholder for generating step-back questions.
        rrf_k: RRF constant for fusing results (default: 60).
        top_k: Number of results to retrieve per query.
        batch_size: Number of queries to process in each batch.

    Example:
        ```python
        config = StepBackPipelineConfig(
            name="step_back_gemini",
            llm="google-gemini-2.5-flash",
            embedding="huggingface",
            rrf_k=60,
            top_k=10,
        )
        ```
    """

    llm: str | BaseLanguageModel
    """LLM for generating step-back questions. Can be config name or instance."""

    embedding: str | Embeddings
    """Embedding model for queries. Can be config name or instance."""

    prompt_template: str = field(default=DEFAULT_STEP_BACK_PROMPT_TEMPLATE)
    """Template with {query} placeholder for generating step-back questions."""

    rrf_k: int = 60
    """RRF constant. Higher values give more weight to top ranks."""

    def __setattr__(self, name: str, value: Any) -> None:
        """Auto-convert string config names to model instances."""
        if name == "llm" and isinstance(value, str):
            from autorag_research.injection import load_llm

            value = load_llm(value)
            health_check_llm(value)
        elif name == "embedding" and isinstance(value, str):
            from autorag_research.injection import load_embedding_model

            value = load_embedding_model(value)
            health_check_embedding(value)
        super().__setattr__(name, value)

    def get_pipeline_class(self) -> type["StepBackRetrievalPipeline"]:
        """Return the StepBackRetrievalPipeline class."""
        return StepBackRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for StepBackRetrievalPipeline constructor."""
        return {
            "llm": self.llm,
            "embedding": self.embedding,
            "prompt_template": self.prompt_template,
            "rrf_k": self.rrf_k,
        }


class StepBackRetrievalPipeline(BaseRetrievalPipeline):
    """Pipeline for Step-back Prompting retrieval.

    This pipeline generates an abstract step-back question using an LLM, embeds both
    the original and step-back queries, performs vector search for each, and fuses
    results using RRF. This approach retrieves background knowledge and principles
    that help answer the original question.

    Reference: "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models"
    https://arxiv.org/abs/2310.06117

    Example:
        ```python
        from langchain_google_genai import ChatGoogleGenerativeAI
        from autorag_research.orm.connection import DBConnection
        from autorag_research.pipelines.retrieval.step_back import StepBackRetrievalPipeline

        db = DBConnection.from_config()
        session_factory = db.get_session_factory()

        pipeline = StepBackRetrievalPipeline(
            session_factory=session_factory,
            name="step_back_gemini",
            llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
            embedding=load_embedding_model("huggingface"),
            rrf_k=60,
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
        llm: BaseLanguageModel,
        embedding: Embeddings,
        prompt_template: str = DEFAULT_STEP_BACK_PROMPT_TEMPLATE,
        rrf_k: int = 60,
        schema: Any | None = None,
    ):
        """Initialize Step-back Prompting retrieval pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            llm: LangChain LLM for generating step-back questions.
            embedding: LangChain embeddings model for embedding queries.
            prompt_template: Template with {query} placeholder for generating step-back questions.
            rrf_k: RRF constant (default: 60).
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        # Store parameters BEFORE calling super().__init__
        # because _get_pipeline_config() is called in super().__init__
        self.llm = llm
        self.embedding = embedding
        if "{query}" not in prompt_template:
            msg = "prompt_template must contain '{query}' placeholder"
            raise ValueError(msg)
        self.prompt_template = prompt_template
        self.rrf_k = rrf_k

        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return Step-back pipeline configuration.

        Returns:
            Dictionary containing pipeline configuration for storage.
        """
        return {
            "type": "step_back",
            "rrf_k": self.rrf_k,
            "prompt_template": self.prompt_template,
        }

    async def _generate_step_back_query(self, query_text: str) -> str:
        """Generate a more abstract step-back question using the LLM.

        Args:
            query_text: The original query to generate a step-back question for.

        Returns:
            Generated step-back question text.
        """
        prompt = self.prompt_template.format(query=query_text)
        response = await self.llm.ainvoke(prompt)
        step_back = self._extract_response_content(response).strip()
        logger.debug(f"Generated step-back query: {step_back!r} from original: {query_text!r}")
        return step_back

    @staticmethod
    def _extract_response_content(response: Any) -> str:
        """Extract text content from LLM response.

        Args:
            response: LLM response (AIMessage or string).

        Returns:
            Extracted text content.
        """
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)

    def _rrf_fuse(
        self,
        results_a: list[dict[str, Any]],
        results_b: list[dict[str, Any]],
        top_k: int,
        fetch_k: int,
    ) -> list[dict[str, Any]]:
        """Fuse two result lists using Reciprocal Rank Fusion.

        Args:
            results_a: First result list with 'doc_id' and 'score' keys.
            results_b: Second result list with 'doc_id' and 'score' keys.
            top_k: Number of results to return.
            fetch_k: Number of results fetched per query.

        Returns:
            Fused results sorted by RRF score (descending).
        """
        missing_rank = 1.0 / (self.rrf_k + fetch_k + 1)

        all_doc_ids: set[int | str] = set()
        for r in results_a:
            all_doc_ids.add(r["doc_id"])
        for r in results_b:
            all_doc_ids.add(r["doc_id"])

        rrf_scores: dict[int | str, float] = dict.fromkeys(all_doc_ids, 0.0)

        for rank, r in enumerate(results_a, 1):
            rrf_scores[r["doc_id"]] += 1.0 / (self.rrf_k + rank)
        a_ids = {r["doc_id"] for r in results_a}
        for doc_id in all_doc_ids - a_ids:
            rrf_scores[doc_id] += missing_rank

        for rank, r in enumerate(results_b, 1):
            rrf_scores[r["doc_id"]] += 1.0 / (self.rrf_k + rank)
        b_ids = {r["doc_id"] for r in results_b}
        for doc_id in all_doc_ids - b_ids:
            rrf_scores[doc_id] += missing_rank

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"doc_id": doc_id, "score": score} for doc_id, score in sorted_docs[:top_k]]

    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve documents using query ID.

        Fetches query text from DB, generates step-back question,
        embeds both, and performs fused vector search.

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
        """Retrieve documents using raw query text.

        Generates step-back query, embeds both original and step-back queries,
        performs vector search for each, and fuses results with RRF.

        Args:
            query_text: The query text to retrieve for.
            top_k: Number of top documents to retrieve.

        Returns:
            List of result dicts with doc_id and score.
        """
        fetch_k = top_k * 2

        # Step 1: Generate step-back query
        step_back_query = await self._generate_step_back_query(query_text)

        # Step 2: Embed both original query and step-back query
        original_embedding = await self.embedding.aembed_query(query_text)
        step_back_embedding = await self.embedding.aembed_query(step_back_query)

        # Step 3: Vector search for both embeddings
        original_results = self._service.vector_search_by_embedding(
            embedding=original_embedding,
            top_k=fetch_k,
        )
        step_back_results = self._service.vector_search_by_embedding(
            embedding=step_back_embedding,
            top_k=fetch_k,
        )

        # Step 4: Fuse results using RRF
        return self._rrf_fuse(original_results, step_back_results, top_k, fetch_k)


__all__ = [
    "DEFAULT_STEP_BACK_PROMPT_TEMPLATE",
    "DEFAULT_STEP_BACK_PROMPT_TEMPLATE_KOREAN",
    "StepBackPipelineConfig",
    "StepBackRetrievalPipeline",
]
