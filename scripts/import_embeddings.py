"""Import GPU-generated embeddings back into PostgreSQL databases."""

import json
import sys
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, text

BASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432"

DATABASES = [
    "oliveyoung_bge_m3",
    "oliveyoung_finetuned",
    "scifact_bge_m3",
    "scifact_finetuned",
    "mrtydi_korean_bge_m3",
    "mrtydi_korean_finetuned",
    "mrtydi_korean_full",
]


def import_db(db_name: str, input_dir: Path) -> None:
    """Import embeddings for one database."""
    chunk_embs_file = input_dir / f"{db_name}_chunk_embs.npy"
    chunk_ids_file = input_dir / f"{db_name}_chunk_ids.json"
    query_embs_file = input_dir / f"{db_name}_query_embs.npy"
    query_ids_file = input_dir / f"{db_name}_query_ids.json"

    engine = create_engine(f"{BASE_URL}/{db_name}")

    with engine.connect() as conn:
        # Import chunk embeddings
        if chunk_embs_file.exists():
            chunk_embs = np.load(chunk_embs_file)
            with open(chunk_ids_file) as f:
                chunk_ids = json.load(f)
            print(f"  Importing {len(chunk_ids)} chunk embeddings (dim={chunk_embs.shape[1]})...")

            batch_size = 500
            for i in range(0, len(chunk_ids), batch_size):
                batch_ids = chunk_ids[i : i + batch_size]
                batch_embs = chunk_embs[i : i + batch_size]
                for cid, emb in zip(batch_ids, batch_embs, strict=True):
                    emb_str = "[" + ",".join(str(float(v)) for v in emb) + "]"
                    conn.execute(
                        text("UPDATE chunk SET embedding = :emb WHERE id = :id"),
                        {"emb": emb_str, "id": cid},
                    )
                conn.commit()
                if (i + batch_size) % 5000 == 0 or i + batch_size >= len(chunk_ids):
                    print(f"    chunks: {min(i + batch_size, len(chunk_ids))}/{len(chunk_ids)}")

        # Import query embeddings
        if query_embs_file.exists():
            query_embs = np.load(query_embs_file)
            with open(query_ids_file) as f:
                query_ids = json.load(f)
            print(f"  Importing {len(query_ids)} query embeddings (dim={query_embs.shape[1]})...")

            for i in range(0, len(query_ids), batch_size):
                batch_ids = query_ids[i : i + batch_size]
                batch_embs = query_embs[i : i + batch_size]
                for qid, emb in zip(batch_ids, batch_embs, strict=True):
                    emb_str = "[" + ",".join(str(float(v)) for v in emb) + "]"
                    conn.execute(
                        text("UPDATE query SET embedding = :emb WHERE id = :id"),
                        {"emb": emb_str, "id": qid},
                    )
                conn.commit()
                print(f"    queries: {min(i + batch_size, len(query_ids))}/{len(query_ids)}")

    engine.dispose()


def verify_db(db_name: str) -> None:
    """Quick verification: check cosine similarity between first query and its GT chunk."""
    engine = create_engine(f"{BASE_URL}/{db_name}")

    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT q.contents,
                       1 - (q.embedding <=> c.embedding) as cosine,
                       c.contents
                FROM query q
                JOIN retrieval_relation rr ON rr.query_id = q.id
                JOIN chunk c ON c.id = rr.chunk_id
                WHERE q.embedding IS NOT NULL AND c.embedding IS NOT NULL
                LIMIT 3
            """)
        )
        rows = result.fetchall()
        if rows:
            print("  Verification (query↔GT chunk cosine):")
            for row in rows:
                print(f"    {row[1]:.4f} | Q: {row[0][:40]}... → C: {row[2][:40]}...")
        else:
            print("  WARNING: No results for verification")

    engine.dispose()


def main():
    input_dir = Path("scripts/embedding_output")

    if not input_dir.exists():
        print(f"ERROR: {input_dir} not found. SCP embeddings from server first.")
        sys.exit(1)

    dbs = sys.argv[1:] if len(sys.argv) > 1 else DATABASES

    print("Importing embeddings...")
    for db_name in dbs:
        if not (input_dir / f"{db_name}_chunk_embs.npy").exists():
            print(f"  SKIP {db_name}: no embedding files found")
            continue
        print(f"\n{'=' * 60}")
        print(f"DB: {db_name}")
        import_db(db_name, input_dir)
        verify_db(db_name)

    print(f"\n{'=' * 60}")
    print("All done!")


if __name__ == "__main__":
    main()
