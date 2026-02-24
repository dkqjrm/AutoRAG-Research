"""Query Decomposition Retrieval Pipeline for AutoRAG-Research.

This pipeline implements a query decomposition approach for retrieval:
1. Using an LLM to break a complex query into simpler sub-queries
2. Embedding each sub-query separately
3. Performing vector similarity search for each sub-query
4. Merging results using Reciprocal Rank Fusion (RRF)

This approach improves recall for complex multi-faceted queries.
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

DEFAULT_QUERY_DECOMPOSITION_PROMPT_TEMPLATE_KOREAN = """다음 질문을 검색에 적합한 더 간단한 하위 질문들로 분해하세요.
각 하위 질문은 원래 질문의 서로 다른 측면을 다뤄야 합니다.
줄바꿈으로 구분하여 하위 질문만 출력하세요. 최대 {max_sub_queries}개까지 생성하세요.

원래 질문: {query}
하위 질문:"""

DEFAULT_QUERY_DECOMPOSITION_PROMPT_TEMPLATE = """Decompose the following question into simpler sub-questions suitable for search.
Each sub-question should address a different aspect of the original question.
Output only the sub-questions, one per line. Generate at most {max_sub_queries} sub-questions.

Original question: {query}
Sub-questions:"""


def _multi_rrf_fuse(
    result_lists: list[list[dict[str, Any]]],
    k: int,
    top_k: int,
    fetch_k: int,
) -> list[dict[str, Any]]:
    """Fuse multiple result lists using Reciprocal Rank Fusion.

    Generalizes RRF to N result lists:
    RRF(d) = Σ_i 1/(k + rank_i(d))

    Documents missing from a result list are treated as having rank fetch_k + 1.

    Args:
        result_lists: List of result lists, each with 'doc_id' and 'score' keys.
        k: RRF constant (typically 60).
        top_k: Number of results to return.
        fetch_k: Number of results fetched per sub-query.

    Returns:
        Fused results sorted by RRF score (descending).
    """
    if not result_lists:
        return []

    missing_rank_contribution = 1.0 / (k + fetch_k + 1)

    # Collect all doc_ids across all result lists
    all_doc_ids: set[int | str] = set()
    for results in result_lists:
        for r in results:
            all_doc_ids.add(r["doc_id"])

    # Calculate RRF scores
    rrf_scores: dict[int | str, float] = dict.fromkeys(all_doc_ids, 0.0)

    for results in result_lists:
        result_doc_ids = set()
        for rank, result in enumerate(results, start=1):
            doc_id = result["doc_id"]
            rrf_scores[doc_id] += 1.0 / (k + rank)
            result_doc_ids.add(doc_id)

        # Add floor contribution for missing documents
        for doc_id in all_doc_ids - result_doc_ids:
            rrf_scores[doc_id] += missing_rank_contribution

    # Sort by RRF score and return top_k
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"doc_id": doc_id, "score": score} for doc_id, score in sorted_docs[:top_k]]


@dataclass(kw_only=True)
class QueryDecompositionPipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for Query Decomposition retrieval pipeline.

    Attributes:
        name: Unique name for this pipeline instance.
        llm: LLM config name or instance for decomposing queries.
        embedding: Embedding config name or instance for embedding sub-queries.
        prompt_template: Template with {query} and {max_sub_queries} placeholders.
        max_sub_queries: Maximum number of sub-queries to generate (default: 4).
        rrf_k: RRF constant (default: 60).
        top_k: Number of results to retrieve per query.
        batch_size: Number of queries to process in each batch.

    Example:
        ```python
        config = QueryDecompositionPipelineConfig(
            name="query_decomposition_gemini",
            llm="google-gemini-2.5-flash",
            embedding="huggingface",
            max_sub_queries=4,
            top_k=10,
        )
        ```
    """

    llm: str | BaseLanguageModel
    """LLM for decomposing queries into sub-queries. Can be config name or instance."""

    embedding: str | Embeddings
    """Embedding model for sub-queries. Can be config name or instance."""

    prompt_template: str = field(default=DEFAULT_QUERY_DECOMPOSITION_PROMPT_TEMPLATE_KOREAN)
    """Template with {query} and {max_sub_queries} placeholders."""

    max_sub_queries: int = 4
    """Maximum number of sub-queries to generate."""

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

    def get_pipeline_class(self) -> type["QueryDecompositionRetrievalPipeline"]:
        """Return the QueryDecompositionRetrievalPipeline class."""
        return QueryDecompositionRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for QueryDecompositionRetrievalPipeline constructor."""
        return {
            "llm": self.llm,
            "embedding": self.embedding,
            "prompt_template": self.prompt_template,
            "max_sub_queries": self.max_sub_queries,
            "rrf_k": self.rrf_k,
        }


class QueryDecompositionRetrievalPipeline(BaseRetrievalPipeline):
    """Pipeline for Query Decomposition retrieval.

    This pipeline decomposes complex queries into simpler sub-queries using an LLM,
    embeds each sub-query, performs vector search for each, and fuses results using RRF.
    This approach improves recall for complex multi-faceted queries.

    Example:
        ```python
        from langchain_google_genai import ChatGoogleGenerativeAI
        from autorag_research.orm.connection import DBConnection
        from autorag_research.pipelines.retrieval.query_decomposition import (
            QueryDecompositionRetrievalPipeline,
        )

        db = DBConnection.from_config()
        session_factory = db.get_session_factory()

        pipeline = QueryDecompositionRetrievalPipeline(
            session_factory=session_factory,
            name="query_decomposition_gemini",
            llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
            embedding=load_embedding_model("huggingface"),
            max_sub_queries=4,
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
        prompt_template: str = DEFAULT_QUERY_DECOMPOSITION_PROMPT_TEMPLATE_KOREAN,
        max_sub_queries: int = 4,
        rrf_k: int = 60,
        schema: Any | None = None,
    ):
        """Initialize Query Decomposition retrieval pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            llm: LangChain LLM for decomposing queries into sub-queries.
            embedding: LangChain embeddings model for embedding sub-queries.
            prompt_template: Template with {query} and {max_sub_queries} placeholders.
            max_sub_queries: Maximum number of sub-queries to generate (default: 4).
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
        self.max_sub_queries = max_sub_queries
        self.rrf_k = rrf_k

        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return Query Decomposition pipeline configuration."""
        return {
            "type": "query_decomposition",
            "max_sub_queries": self.max_sub_queries,
            "rrf_k": self.rrf_k,
            "prompt_template": self.prompt_template,
        }

    async def _decompose_query(self, query_text: str) -> list[str]:
        """Decompose a complex query into simpler sub-queries using the LLM.

        Args:
            query_text: The original query text.

        Returns:
            List of sub-query strings.
        """
        prompt = self.prompt_template.format(query=query_text, max_sub_queries=self.max_sub_queries)
        response = await self.llm.ainvoke(prompt)
        response_text = self._extract_response_content(response)

        # Parse one sub-query per line
        sub_queries = [line.strip() for line in response_text.strip().split("\n") if line.strip()]
        # Remove numbering prefixes like "1. " or "1) "
        cleaned = []
        for q in sub_queries:
            for prefix in ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "1)", "2)", "3)", "4)", "5)"):
                if q.startswith(prefix):
                    q = q[len(prefix) :].strip()
                    break
            if q:
                cleaned.append(q)

        # Limit to max_sub_queries
        cleaned = cleaned[: self.max_sub_queries]

        logger.debug(f"Decomposed query into {len(cleaned)} sub-queries: {cleaned}")
        return cleaned

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

    async def _retrieve_by_id(self, query_id: int | str, top_k: int) -> list[dict[str, Any]]:
        """Retrieve documents using query ID.

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

        Decomposes query into sub-queries, embeds each, searches, and fuses with RRF.

        Args:
            query_text: The query text to retrieve for.
            top_k: Number of top documents to retrieve.

        Returns:
            List of result dicts with doc_id and score.
        """
        # Step 1: Decompose query into sub-queries
        sub_queries = await self._decompose_query(query_text)

        if not sub_queries:
            # Fall back to original query if decomposition yields nothing
            sub_queries = [query_text]

        # Step 2: Embed each sub-query and search
        fetch_k = top_k * 2  # Fetch more to improve RRF fusion quality
        result_lists: list[list[dict[str, Any]]] = []

        for sub_query in sub_queries:
            embedding = await self.embedding.aembed_query(sub_query)
            results = self._service.vector_search_by_embedding(
                embedding=embedding,
                top_k=fetch_k,
            )
            result_lists.append(results)

        # Step 3: Fuse with RRF
        return _multi_rrf_fuse(result_lists, self.rrf_k, top_k, fetch_k)


__all__ = [
    "DEFAULT_QUERY_DECOMPOSITION_PROMPT_TEMPLATE",
    "DEFAULT_QUERY_DECOMPOSITION_PROMPT_TEMPLATE_KOREAN",
    "QueryDecompositionPipelineConfig",
    "QueryDecompositionRetrievalPipeline",
]
