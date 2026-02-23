"""GrepRAG Retrieval Pipeline for AutoRAG-Research.

Adapts the GrepRAG paper's concept to use actual ripgrep (rg) for text search.
Chunks are exported from PostgreSQL to text files, then ripgrep searches them.

GrepRAG works by:
1. Exporting DB chunks to a file-per-chunk directory (once, at init)
2. Using an LLM to extract search keywords from the query
3. Running ripgrep (rg) to search chunk files for those keywords
4. Scoring chunks by keyword match count and ranking by relevance
"""

import logging
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from sqlalchemy.orm import Session, sessionmaker

from autorag_research.config import BaseRetrievalPipelineConfig
from autorag_research.injection import health_check_llm
from autorag_research.pipelines.retrieval.base import BaseRetrievalPipeline

logger = logging.getLogger("AutoRAG-Research")

RG_BINARY = shutil.which("rg") or "/opt/homebrew/bin/rg"

DEFAULT_GREP_RAG_PROMPT_TEMPLATE_KOREAN = """다음 질문에 대한 답을 찾기 위해 문서에서 검색할 핵심 키워드를 추출하세요.
다양한 키워드를 포함하되, 가장 중요한 것부터 나열하세요.
쉼표로 구분하여 키워드만 출력하세요.

질문: {query}
키워드:"""

DEFAULT_GREP_RAG_PROMPT_TEMPLATE = """Extract key search keywords from the following question to find relevant documents.
Include diverse keywords, listing the most important ones first.
Output only keywords separated by commas.

Question: {query}
Keywords:"""


@dataclass(kw_only=True)
class GrepRAGPipelineConfig(BaseRetrievalPipelineConfig):
    """Configuration for GrepRAG retrieval pipeline.

    Attributes:
        name: Unique name for this pipeline instance.
        llm: LLM config name or instance for generating search keywords.
        prompt_template: Template with {query} placeholder for generating keywords.
        max_keywords: Maximum number of keywords to generate (default: 10).
        top_k: Number of results to retrieve per query.
        batch_size: Number of queries to process in each batch.
    """

    llm: str | BaseLanguageModel
    """LLM for generating search keywords. Can be config name or instance."""

    prompt_template: str = field(default=DEFAULT_GREP_RAG_PROMPT_TEMPLATE_KOREAN)
    """Template with {query} placeholder for generating search keywords."""

    max_keywords: int = 10
    """Maximum number of keywords to generate from query."""

    skip_verification: bool = True
    """Skip verification since keyword search may return 0 results for some queries."""

    def __setattr__(self, name: str, value: Any) -> None:
        """Auto-convert string config names to model instances."""
        if name == "llm" and isinstance(value, str):
            from autorag_research.injection import load_llm

            value = load_llm(value)
            health_check_llm(value)
        super().__setattr__(name, value)

    def get_pipeline_class(self) -> type["GrepRAGRetrievalPipeline"]:
        """Return the GrepRAGRetrievalPipeline class."""
        return GrepRAGRetrievalPipeline

    def get_pipeline_kwargs(self) -> dict[str, Any]:
        """Return kwargs for GrepRAGRetrievalPipeline constructor."""
        return {
            "llm": self.llm,
            "prompt_template": self.prompt_template,
            "max_keywords": self.max_keywords,
        }


class GrepRAGRetrievalPipeline(BaseRetrievalPipeline):
    """Pipeline for GrepRAG retrieval using actual ripgrep.

    This pipeline exports chunks from PostgreSQL to text files, then uses
    ripgrep (rg) for fast keyword-based text search. Chunks are scored by
    the number of matching keywords.

    Reference: Adapts concepts from the GrepRAG paper (arXiv:2601.23254).

    Example:
        ```python
        from langchain_google_genai import ChatGoogleGenerativeAI
        from autorag_research.orm.connection import DBConnection
        from autorag_research.pipelines.retrieval.grep_rag import GrepRAGRetrievalPipeline

        db = DBConnection.from_config()
        session_factory = db.get_session_factory()

        pipeline = GrepRAGRetrievalPipeline(
            session_factory=session_factory,
            name="grep_rag_gemini",
            llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash"),
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
        prompt_template: str = DEFAULT_GREP_RAG_PROMPT_TEMPLATE_KOREAN,
        max_keywords: int = 10,
        schema: Any | None = None,
    ):
        """Initialize GrepRAG retrieval pipeline.

        Args:
            session_factory: SQLAlchemy sessionmaker for database connections.
            name: Name for this pipeline.
            llm: LangChain LLM for generating search keywords.
            prompt_template: Template with {query} placeholder for generating keywords.
            max_keywords: Maximum number of keywords to extract (default: 10).
            schema: Schema namespace from create_schema(). If None, uses default schema.
        """
        self.llm = llm
        if "{query}" not in prompt_template:
            msg = "prompt_template must contain '{query}' placeholder"
            raise ValueError(msg)
        self.prompt_template = prompt_template
        self.max_keywords = max_keywords
        self._chunks_dir: Path | None = None
        self._chunk_id_map: dict[str, str | int] = {}

        super().__init__(session_factory, name, schema)

        # Export chunks to files after DB pipeline is created
        self._export_chunks_to_files()

    def close(self) -> None:
        """Clean up temporary chunk files directory."""
        if self._chunks_dir and self._chunks_dir.exists():
            import shutil as _shutil

            _shutil.rmtree(self._chunks_dir, ignore_errors=True)
            logger.info(f"Cleaned up chunk files at {self._chunks_dir}")

    def _export_chunks_to_files(self) -> None:
        """Export all chunks from DB to individual text files for ripgrep search."""
        self._chunks_dir = Path(tempfile.mkdtemp(prefix="greprag_"))
        logger.info(f"Exporting chunks to {self._chunks_dir}")

        chunks = self._service.get_all_chunks()
        for chunk_id, contents in chunks:
            # Use sanitized filename (replace problematic chars)
            safe_name = str(chunk_id).replace("/", "_").replace(" ", "_")
            file_path = self._chunks_dir / f"{safe_name}.txt"
            file_path.write_text(contents, encoding="utf-8")
            self._chunk_id_map[safe_name] = chunk_id

        logger.info(f"Exported {len(self._chunk_id_map)} chunks to {self._chunks_dir}")

    def _get_pipeline_config(self) -> dict[str, Any]:
        """Return GrepRAG pipeline configuration."""
        return {
            "type": "grep_rag",
            "search_engine": "ripgrep",
            "prompt_template": self.prompt_template,
            "max_keywords": self.max_keywords,
        }

    async def _generate_keywords(self, query_text: str) -> list[str]:
        """Generate search keywords using the LLM.

        Args:
            query_text: The query to generate keywords for.

        Returns:
            List of extracted keywords (up to max_keywords).
        """
        prompt = self.prompt_template.format(query=query_text)
        response = await self.llm.ainvoke(prompt)
        response_text = self._extract_response_content(response)

        keywords = [kw.strip() for kw in response_text.split(",") if kw.strip()]
        keywords = keywords[: self.max_keywords]

        logger.debug(f"Generated {len(keywords)} keywords for query: {keywords}")
        return keywords

    @staticmethod
    def _extract_response_content(response: Any) -> str:
        """Extract text content from LLM response."""
        if hasattr(response, "content"):
            return str(response.content)
        return str(response)

    def _ripgrep_search(self, keywords: list[str], top_k: int) -> list[dict[str, Any]]:
        """Run ripgrep for each keyword and aggregate results.

        Args:
            keywords: List of keywords to search for.
            top_k: Number of top results to return.

        Returns:
            List of result dicts with doc_id, score, and content.
        """
        if not self._chunks_dir or not keywords:
            return []

        # Count keyword matches per file using ripgrep
        match_counts: Counter[str] = Counter()

        for keyword in keywords:
            try:
                result = subprocess.run(  # noqa: S603
                    [RG_BINARY, "--files-with-matches", "--ignore-case", "--fixed-strings", keyword, str(self._chunks_dir)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                # Each matching file path → extract chunk safe_name
                for line in result.stdout.strip().split("\n"):
                    if line:
                        filename = Path(line).stem  # e.g., "chunk_123"
                        match_counts[filename] += 1
            except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
                logger.warning(f"ripgrep failed for keyword '{keyword}': {e}")
                continue

        if not match_counts:
            return []

        # Sort by match count (descending), take top_k
        total_keywords = len(keywords)
        top_matches = match_counts.most_common(top_k)

        # Build results with normalized scores
        results = []
        for safe_name, count in top_matches:
            chunk_id = self._chunk_id_map.get(safe_name)
            if chunk_id is None:
                continue

            score = count / total_keywords  # Normalize to [0, 1]

            # Read chunk content from file
            file_path = self._chunks_dir / f"{safe_name}.txt"
            content = file_path.read_text(encoding="utf-8") if file_path.exists() else ""

            results.append({
                "doc_id": chunk_id,
                "score": score,
                "content": content,
            })

        return results

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

        Args:
            query_text: The query text to retrieve for.
            top_k: Number of top documents to retrieve.

        Returns:
            List of result dicts with doc_id and score.
        """
        # Step 1: Generate keywords using LLM
        keywords = await self._generate_keywords(query_text)

        if not keywords:
            logger.warning(f"No keywords generated for query: {query_text}")
            return []

        # Step 2: Run ripgrep search
        return self._ripgrep_search(keywords, top_k)


__all__ = [
    "DEFAULT_GREP_RAG_PROMPT_TEMPLATE",
    "DEFAULT_GREP_RAG_PROMPT_TEMPLATE_KOREAN",
    "GrepRAGPipelineConfig",
    "GrepRAGRetrievalPipeline",
]
