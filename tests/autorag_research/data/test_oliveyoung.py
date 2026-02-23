import csv
import json
from unittest.mock import MagicMock

import pytest
from langchain_core.embeddings import FakeEmbeddings

from autorag_research.data.oliveyoung import OliveyoungIngestor


@pytest.fixture
def products_csv(tmp_path):
    """Create a temporary products CSV file."""
    csv_path = tmp_path / "products_unique.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "goodsNo", "Brand", "Name", "Sale Price", "Original Price",
                "L1", "L2", "L3_Sample", "Categories", "Category Count",
                "Image URL", "Main Image Path", "Product URL", "Category URL",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "goodsNo": "A000000130411",
            "Brand": "스팀베이스",
            "Name": "스팀베이스 데일리 아이마스크 5매입(라벤더향)",
            "Sale Price": "9,900",
            "Original Price": "12,000",
            "L1": "뷰티소품",
            "L2": "아이케어",
            "L3_Sample": "아이마스크",
            "Categories": "뷰티소품 > 아이케어",
            "Category Count": "1",
            "Image URL": "",
            "Main Image Path": "",
            "Product URL": "",
            "Category URL": "",
        })
        writer.writerow({
            "goodsNo": "A000000229369",
            "Brand": "트루락",
            "Name": "트루락 키즈업 30포",
            "Sale Price": "29,000",
            "Original Price": "",
            "L1": "건강식품",
            "L2": "유산균",
            "L3_Sample": "키즈유산균",
            "Categories": "건강식품 > 유산균",
            "Category Count": "1",
            "Image URL": "",
            "Main Image Path": "",
            "Product URL": "",
            "Category URL": "",
        })
        writer.writerow({
            "goodsNo": "A000000187909",
            "Brand": "필립비",
            "Name": "필립비 디탱글링 토닝 미스트 125ml",
            "Sale Price": "35,000",
            "Original Price": "",
            "L1": "헤어케어",
            "L2": "에센스/세럼",
            "L3_Sample": "헤어미스트",
            "Categories": "헤어케어 > 에센스/세럼",
            "Category Count": "1",
            "Image URL": "",
            "Main Image Path": "",
            "Product URL": "",
            "Category URL": "",
        })
    return str(csv_path)


@pytest.fixture
def eval_json(tmp_path):
    """Create a temporary eval JSON file."""
    eval_data = [
        {
            "original_product": {
                "brand": "스팀베이스",
                "name": "스팀베이스 데일리 아이마스크 5매입(라벤더향)",
                "goods_no": "A000000130411",
            },
            "generated_queries": {
                "korean": "스팀베이스 아이마스크 라벤더",
                "korean_typo": "스팀베이스 아이 마스크 라벤더 향",
                "english": "steam base eye mask lavender",
                "english_typo": "steam bay's eye mask lavendar",
                "chinese": "蒸汽眼罩 薰衣草",
                "chinese_typo": "真气眼罩熏衣草",
                "japanese": "スチームベース アイマスク ラベンダー",
                "japanese_typo": "スチームベース愛マスクラベンダー",
            },
        },
        {
            "original_product": {
                "brand": "트루락",
                "name": "트루락 키즈업 30포",
                "goods_no": "A000000229369",
            },
            "generated_queries": {
                "korean": "트루락 키즈업",
                "korean_typo": "트루락 키즈업 30포",
                "english": "TrueLac Kids Up",
                "english_typo": "TrueLack Kids Up",
                "chinese": "TrueLac 儿童益生菌",
                "chinese_typo": "出路拉克 儿童益生菌",
                "japanese": "トゥルーラック キッズアップ",
                "japanese_typo": "トゥルーラック キズアプ",
            },
        },
    ]
    json_path = tmp_path / "typo_eval_100.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False)
    return str(json_path)


class TestOliveyoungIngestorInit:
    def test_init_default_query_types(self, products_csv, eval_json):
        embedding = FakeEmbeddings(size=1024)
        ingestor = OliveyoungIngestor(
            embedding_model=embedding,
            products_csv=products_csv,
            eval_json=eval_json,
        )
        assert ingestor.query_types == ["korean", "korean_typo"]

    def test_init_custom_query_types(self, products_csv, eval_json):
        embedding = FakeEmbeddings(size=1024)
        ingestor = OliveyoungIngestor(
            embedding_model=embedding,
            products_csv=products_csv,
            eval_json=eval_json,
            query_types="english,english_typo,korean",
        )
        assert ingestor.query_types == ["english", "english_typo", "korean"]

    def test_detect_primary_key_type(self, products_csv, eval_json):
        embedding = FakeEmbeddings(size=1024)
        ingestor = OliveyoungIngestor(
            embedding_model=embedding,
            products_csv=products_csv,
            eval_json=eval_json,
        )
        assert ingestor.detect_primary_key_type() == "string"


class TestOliveyoungIngestorLoadProducts:
    def test_load_products(self, products_csv, eval_json):
        embedding = FakeEmbeddings(size=1024)
        ingestor = OliveyoungIngestor(
            embedding_model=embedding,
            products_csv=products_csv,
            eval_json=eval_json,
        )
        products = ingestor._load_products()
        assert len(products) == 3
        assert "A000000130411" in products
        assert products["A000000130411"]["brand"] == "스팀베이스"
        assert products["A000000130411"]["name"] == "스팀베이스 데일리 아이마스크 5매입(라벤더향)"

    def test_load_eval_data(self, products_csv, eval_json):
        embedding = FakeEmbeddings(size=1024)
        ingestor = OliveyoungIngestor(
            embedding_model=embedding,
            products_csv=products_csv,
            eval_json=eval_json,
        )
        eval_data = ingestor._load_eval_data()
        assert len(eval_data) == 2
        assert eval_data[0]["original_product"]["goods_no"] == "A000000130411"
        assert "korean" in eval_data[0]["generated_queries"]


class TestOliveyoungIngestorIngest:
    def test_ingest_creates_chunks_queries_and_gt(self, products_csv, eval_json):
        """Test that ingest calls service methods with correct data."""
        embedding = FakeEmbeddings(size=1024)
        ingestor = OliveyoungIngestor(
            embedding_model=embedding,
            products_csv=products_csv,
            eval_json=eval_json,
            query_types="korean,korean_typo",
        )

        mock_service = MagicMock()
        mock_service.add_chunks.return_value = []
        mock_service.add_queries.return_value = []
        mock_service.add_retrieval_gt.return_value = []
        ingestor.set_service(mock_service)

        ingestor.ingest()

        # Verify chunks: 3 products ingested
        mock_service.add_chunks.assert_called_once()
        chunks = mock_service.add_chunks.call_args[0][0]
        assert len(chunks) == 3
        assert chunks[0]["id"] == "A000000130411"
        assert chunks[0]["contents"] == "스팀베이스 스팀베이스 데일리 아이마스크 5매입(라벤더향)"

        # Verify queries: 2 products x 2 query types = 4 queries
        mock_service.add_queries.assert_called_once()
        queries = mock_service.add_queries.call_args[0][0]
        assert len(queries) == 4

        # Check query IDs follow '{goods_no}_{query_type}' format
        query_ids = [q["id"] for q in queries]
        assert "A000000130411_korean" in query_ids
        assert "A000000130411_korean_typo" in query_ids
        assert "A000000229369_korean" in query_ids
        assert "A000000229369_korean_typo" in query_ids

        # Verify ground truth: 4 GT relations (2 products x 2 query types)
        assert mock_service.add_retrieval_gt.call_count == 4
        mock_service.clean.assert_called_once()

    def test_ingest_with_query_limit(self, products_csv, eval_json):
        """Test that query_limit limits eval entries."""
        embedding = FakeEmbeddings(size=1024)
        ingestor = OliveyoungIngestor(
            embedding_model=embedding,
            products_csv=products_csv,
            eval_json=eval_json,
            query_types="korean",
        )

        mock_service = MagicMock()
        mock_service.add_chunks.return_value = []
        mock_service.add_queries.return_value = []
        mock_service.add_retrieval_gt.return_value = []
        ingestor.set_service(mock_service)

        ingestor.ingest(query_limit=1)

        # Only 1 eval entry x 1 query type = 1 query
        queries = mock_service.add_queries.call_args[0][0]
        assert len(queries) == 1

    def test_ingest_without_service_raises(self, products_csv, eval_json):
        """Test that ingest raises ServiceNotSetError when service is not set."""
        from autorag_research.exceptions import ServiceNotSetError

        embedding = FakeEmbeddings(size=1024)
        ingestor = OliveyoungIngestor(
            embedding_model=embedding,
            products_csv=products_csv,
            eval_json=eval_json,
        )

        with pytest.raises(ServiceNotSetError):
            ingestor.ingest()

    def test_ingest_all_query_types(self, products_csv, eval_json):
        """Test ingesting all 8 query types."""
        embedding = FakeEmbeddings(size=1024)
        ingestor = OliveyoungIngestor(
            embedding_model=embedding,
            products_csv=products_csv,
            eval_json=eval_json,
            query_types="korean,korean_typo,english,english_typo,chinese,chinese_typo,japanese,japanese_typo",
        )

        mock_service = MagicMock()
        mock_service.add_chunks.return_value = []
        mock_service.add_queries.return_value = []
        mock_service.add_retrieval_gt.return_value = []
        ingestor.set_service(mock_service)

        ingestor.ingest()

        # 2 products x 8 query types = 16 queries
        queries = mock_service.add_queries.call_args[0][0]
        assert len(queries) == 16
        assert mock_service.add_retrieval_gt.call_count == 16
