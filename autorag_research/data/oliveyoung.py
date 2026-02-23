import csv
import json
import logging
from typing import Literal

from langchain_core.embeddings import Embeddings

from autorag_research.data.base import TextEmbeddingDataIngestor
from autorag_research.data.registry import register_ingestor
from autorag_research.exceptions import ServiceNotSetError

logger = logging.getLogger("AutoRAG-Research")

# Query types available in the eval dataset
QUERY_TYPES = Literal[
    "korean",
    "korean_typo",
    "english",
    "english_typo",
    "chinese",
    "chinese_typo",
    "japanese",
    "japanese_typo",
]


@register_ingestor(
    name="oliveyoung",
    description="Oliveyoung product search dataset (CSV products + typo eval queries)",
)
class OliveyoungIngestor(TextEmbeddingDataIngestor):
    """Ingestor for Oliveyoung product search dataset.

    Products are loaded from a CSV file and ingested as "{Brand} {Name}" chunks.
    Eval queries are loaded from a JSON file with multi-language typo variations.

    Each eval entry maps a product (by goodsNo) to 8 query types:
    korean, korean_typo, english, english_typo, chinese, chinese_typo,
    japanese, japanese_typo.
    """

    def __init__(
        self,
        embedding_model: Embeddings,
        products_csv: str,
        eval_json: str,
        query_types: str = "korean,korean_typo",
    ):
        """Initialize Oliveyoung ingestor.

        Args:
            embedding_model: Embedding model for vectorization.
            products_csv: Path to products_unique.csv file.
            eval_json: Path to typo_eval_100.json file.
            query_types: Comma-separated query types to ingest.
                Available: korean, korean_typo, english, english_typo,
                chinese, chinese_typo, japanese, japanese_typo.
                Default: "korean,korean_typo"
        """
        super().__init__(embedding_model)
        self.products_csv = products_csv
        self.eval_json = eval_json
        self.query_types = [qt.strip() for qt in query_types.split(",")]

    def detect_primary_key_type(self) -> Literal["bigint", "string"]:
        """Oliveyoung uses string primary keys (goodsNo like 'A000000130411')."""
        return "string"

    def ingest(
        self,
        subset: Literal["train", "dev", "test"] = "test",
        query_limit: int | None = None,
        min_corpus_cnt: int | None = None,
    ) -> None:
        """Ingest Oliveyoung product data and eval queries.

        Args:
            subset: Ignored for this dataset (single split).
            query_limit: Maximum number of eval entries to ingest.
            min_corpus_cnt: Not used (all products are ingested).
        """
        if self.service is None:
            raise ServiceNotSetError

        # Step 1: Load and ingest products as chunks
        products = self._load_products()
        logger.info(f"Loaded {len(products)} products from CSV")
        self._ingest_products(products)

        # Step 2: Load eval data and ingest queries + ground truth
        eval_data = self._load_eval_data()
        if query_limit is not None:
            eval_data = eval_data[:query_limit]
        logger.info(f"Loaded {len(eval_data)} eval entries with query types: {self.query_types}")

        product_ids = set(products.keys())
        self._ingest_queries_and_gt(eval_data, product_ids)

        self.service.clean()

    def _load_products(self) -> dict[str, dict[str, str]]:
        """Load products from CSV file.

        Returns:
            Dict mapping goodsNo to {brand, name} dict.
        """
        products: dict[str, dict[str, str]] = {}
        with open(self.products_csv, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                goods_no = row["goodsNo"]
                brand = row.get("Brand", "").strip()
                name = row.get("Name", "").strip()
                products[goods_no] = {"brand": brand, "name": name}
        return products

    def _ingest_products(self, products: dict[str, dict[str, str]], batch_size: int = 5000) -> None:
        """Ingest products as text chunks with '{Brand} {Name}' format.

        Chunks are inserted in batches to avoid PostgreSQL parameter limit (65535).
        """
        if self.service is None:
            raise ServiceNotSetError

        chunks = []
        for goods_no, info in products.items():
            contents = f"{info['brand']} {info['name']}".strip()
            chunks.append({"id": goods_no, "contents": contents})

        logger.info(f"Ingesting {len(chunks)} product chunks in batches of {batch_size}...")
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.service.add_chunks(batch)
            logger.info(f"  Ingested batch {i // batch_size + 1}: {len(batch)} chunks")

    def _load_eval_data(self) -> list[dict]:
        """Load eval data from JSON file."""
        with open(self.eval_json, encoding="utf-8") as f:
            return json.load(f)

    def _ingest_queries_and_gt(self, eval_data: list[dict], product_ids: set[str]) -> None:
        """Ingest queries and ground truth from eval data.

        Each eval entry generates one query per selected query_type.
        Query ID format: '{goods_no}_{query_type}' (e.g., 'A000000130411_korean_typo').
        Ground truth: the original product's goodsNo.
        """
        if self.service is None:
            raise ServiceNotSetError

        from autorag_research.orm.models import or_all

        queries = []
        for entry in eval_data:
            goods_no = entry["original_product"]["goods_no"]
            generated_queries = entry["generated_queries"]

            for qt in self.query_types:
                if qt not in generated_queries:
                    logger.warning(f"Query type '{qt}' not found for product {goods_no}, skipping")
                    continue

                query_text = generated_queries[qt]
                query_id = f"{goods_no}_{qt}"
                queries.append({"id": query_id, "contents": query_text})

        logger.info(f"Ingesting {len(queries)} queries...")
        self.service.add_queries(queries)

        # Add ground truth: each query maps to its original product
        gt_count = 0
        for entry in eval_data:
            goods_no = entry["original_product"]["goods_no"]
            if goods_no not in product_ids:
                logger.warning(f"Product {goods_no} not in corpus, skipping GT")
                continue

            for qt in self.query_types:
                if qt not in entry["generated_queries"]:
                    continue
                query_id = f"{goods_no}_{qt}"
                self.service.add_retrieval_gt(query_id, or_all([goods_no]), chunk_type="text")
                gt_count += 1

        logger.info(f"Ingested {gt_count} ground truth relations")
