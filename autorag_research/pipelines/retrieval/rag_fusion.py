"""RAG-Fusion Retrieval Pipeline for AutoRAG-Research.

Implements the RAG-Fusion approach which generates multiple query variations
using an LLM, performs vector search for each variation, and combines results
using Reciprocal Rank Fusion (RRF).

RAG-Fusion works by:
1. Using an LLM to generate N diverse query variations from the original query
2. Embedding each query variation
3. Performing vector search for each embedding
4. Fusing all result lists using RRF to produce the final ranking

Reference: "RAG-Fusion: a New Take on Retrieval-Augmented Generation"
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

DEFAULT_RAG_FUSION_PROMPT_TEMPLATE_KOREAN = """다음 질문에 대해 서로 다른 관점의 검색 쿼리를 {num_queries}개 생성하세요.
각 쿼리는 원래 질문의 다른 측면이나 표현을 다뤄야 합니다.
줄바꿈으로 구분하여 쿼리만 출력하세요.

원래 질문: {query}
쿼리:"""

DEFAULT_RAG_FUSION_PROMPT_TEMPLATE = """Generate {num_queries} different search queries related to the following question.
Each query should explore a different aspect or phrasing of the original question.
Output only the queries, one per line.

Original question: {query}
Queries:"""


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
        fetch_k: Number of results fetched per query variation.

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
class RAGFusionPipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for RAG-Fusion retrieval pipeline.

    Attributes:
        name: Unique name for this pipeline instance.
        llm: LLM config name or instance for generating query variations.
        embedding: Embedding config name or instance for embedding queries.
        prompt_template: Template with {query} and {num_queries} placeholders.
        num_queries: Number of query variations to generate (default: 4).
        rrf_k: RRF constant (default: 60).
        top_k: Number of results to retrieve per query.
        batch_size: Number of queries to process in each batch.

    Example:
        ```python
        config = RAGFusionPipelineConfig(
            name="rag_fusion_gemini",
            llm="google-gemini-2.5-flash",
            embedding="bge-m3",
            num_queries=4,
            top_k=10,
        )
        ```
    """

    llm: str | BaseLanguageModel
    """LLM for generating query variations. Can be config name or instance."""

    embedding: str | Embeddings
    """Embedding model for query variations. Can be config name or instance."""

    prompt_template: str = field(default=DEFAULT_RAG_FUSION_PROMPT_TEMPLATE_KOREAN)
    """Template with {query} and {num_queries} placeholders."""

    num_queries: int = 4
    """Number of query variations to generate."""

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

    def get_pipeline_class(self) -> type["RAGFusionRetrievalPipeline"]:
        """Return the RAGFusionRetrievalPipeline class."""
        return RAGFusionRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for RAGFusionRetrievalPipeline constructor."""
        return {
            "llm": self.llm,
            "embedding": self.embedding,
            "prompt_template": self.prompt_template,
            "num_queries": self.num_queries,
            "rrf_k": self.rrf_k,
        }


class RAGFusionRetrievalPipeline(BaseRetrievalPipeline):
    """Pipeline for RAG-Fusion retrieval.

    This pipeline generates multiple query variations using an LLM, embeds each
    variation, performs vector search for each, and fuses results using RRF.
    This approach increases recall by exploring different aspects of the query.

    Reference: "RAG-Fusion: a New Take on Retrieval-Augmented Generation"

    Example:
        ```python
        from langchain_google_genai import ChatGoogleGenerativeAI
        from autorag_research.orm.connection import DBConnection
        from autorag_research.pipelines.retrieval.rag_fusion import RAGFusionRetrievalPipeline

        db = DBConnection.from_config()
        session_factory = db.get_session_factory()

        pipeline = RAGFusionRetrievalPipeline(
            session_factory=session_factory,
            name="rag_fusion_gemini",
            llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
            embedding=load_embedding_model("bge-m3"),
            num_queries=4,
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
        prompt_template: str = DEFAULT_RAG_FUSION_PROMPT_TEMPLATE_KOREAN,
        num_queries: int = 4,
        rrf_k: int = 60,
        schema: Any | None = None,
    ):
        """Initialize RAG-Fusion retrieval pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            llm: LangChain LLM for generating query variations.
            embedding: LangChain embeddings model for embedding query variations.
            prompt_template: Template with {query} and {num_queries} placeholders.
            num_queries: Number of query variations to generate (default: 4).
            rrf_k: RRF constant (default: 60).
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        self.llm = llm
        self.embedding = embedding
        if "{query}" not in prompt_template:
            msg = "prompt_template must contain '{query}' placeholder"
            raise ValueError(msg)
        self.prompt_template = prompt_template
        self.num_queries = num_queries
        self.rrf_k = rrf_k

        super().__init__(session_factory, name, schema)

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return RAG-Fusion pipeline configuration."""
        return {
            "type": "rag_fusion",
            "num_queries": self.num_queries,
            "rrf_k": self.rrf_k,
            "prompt_template": self.prompt_template,
        }

    async def _generate_query_variations(self, query_text: str) -> list[str]:
        """Generate diverse query variations using the LLM.

        Args:
            query_text: The original query text.

        Returns:
            List of query variations (including the original query).
        """
        prompt = self.prompt_template.format(query=query_text, num_queries=self.num_queries)
        response = await self.llm.ainvoke(prompt)
        response_text = self._extract_response_content(response)

        # Parse one query per line
        variations = [line.strip() for line in response_text.strip().split("\n") if line.strip()]
        # Remove numbering prefixes like "1. " or "1) "
        cleaned = []
        for v in variations:
            for prefix in ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "1)", "2)", "3)", "4)", "5)"):
                if v.startswith(prefix):
                    v = v[len(prefix):].strip()
                    break
            if v:
                cleaned.append(v)

        # Limit to num_queries and always include the original
        cleaned = cleaned[: self.num_queries]

        logger.debug(f"Generated {len(cleaned)} query variations: {cleaned}")
        return [query_text, *cleaned]

    @staticmethod
    def _extract_response_content(response: Any) -> str:
        """Extract text content from LLM response."""
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

        Generates query variations, embeds each, searches, and fuses with RRF.

        Args:
            query_text: The query text to retrieve for.
            top_k: Number of top documents to retrieve.

        Returns:
            List of result dicts with doc_id and score.
        """
        # Step 1: Generate query variations
        variations = await self._generate_query_variations(query_text)

        # Step 2: Embed each variation and search
        fetch_k = top_k * 2  # Fetch more to improve RRF fusion quality
        result_lists: list[list[dict[str, Any]]] = []

        for variation in variations:
            embedding = await self.embedding.aembed_query(variation)
            results = self._service.vector_search_by_embedding(
                embedding=embedding,
                top_k=fetch_k,
            )
            result_lists.append(results)

        # Step 3: Fuse with RRF
        return _multi_rrf_fuse(result_lists, self.rrf_k, top_k, fetch_k)


__all__ = [
    "DEFAULT_RAG_FUSION_PROMPT_TEMPLATE",
    "DEFAULT_RAG_FUSION_PROMPT_TEMPLATE_KOREAN",
    "RAGFusionPipelineConfig",
    "RAGFusionRetrievalPipeline",
]
