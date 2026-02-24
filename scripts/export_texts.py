"""Export chunk/query texts from all databases for re-embedding on GPU server."""

import json
from pathlib import Path

from sqlalchemy import create_engine, text

DATABASES = [
    "oliveyoung_bge_m3",
    "oliveyoung_finetuned",
    "scifact_bge_m3",
    "scifact_finetuned",
    "mrtydi_korean_bge_m3",
    "mrtydi_korean_finetuned",
    "mrtydi_korean_full",
]

BASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432"


def export_db(db_name: str, output_dir: Path) -> dict:
    """Export chunk and query texts from a database."""
    engine = create_engine(f"{BASE_URL}/{db_name}")
    result = {"db_name": db_name, "chunks": [], "queries": []}

    with engine.connect() as conn:
        # Export chunks (id + contents)
        rows = conn.execute(text("SELECT id, contents FROM chunk ORDER BY id"))
        for row in rows:
            result["chunks"].append({"id": row[0], "content": row[1]})

        # Export queries (id + contents)
        rows = conn.execute(text("SELECT id, contents FROM query ORDER BY id"))
        for row in rows:
            result["queries"].append({"id": row[0], "query_text": row[1]})

    engine.dispose()

    out_file = output_dir / f"{db_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    print(f"  {db_name}: {len(result['chunks'])} chunks, {len(result['queries'])} queries -> {out_file}")
    return result


def main():
    output_dir = Path("scripts/embedding_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting texts from databases...")
    for db_name in DATABASES:
        try:
            export_db(db_name, output_dir)
        except Exception as e:
            print(f"  ERROR {db_name}: {e}")

    print("Done!")


if __name__ == "__main__":
    main()
