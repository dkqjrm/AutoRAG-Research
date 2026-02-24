"""GPU embedding script — runs on server with CUDA.

Usage (inside embed_test venv on server):
    cd /workspace/embed_test
    uv run python /workspace/gpu_embed.py --input-dir /workspace/embedding_data --output-dir /workspace/embedding_output

Models:
    - *_bge_m3 / *_full databases: BAAI/bge-m3 (CLS pooling, auto-normalized)
    - *_finetuned databases: dkqjrm/augmented-olive-product-phonetic-sentence-wo-negative-v2-bge-m3 (MEAN pooling, manual normalize)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Model mapping: db_name suffix -> (model_name, normalize_embeddings)
MODEL_MAP = {
    "bge_m3": ("BAAI/bge-m3", False),  # Has Normalize module built-in
    "full": ("BAAI/bge-m3", False),
    "finetuned": (
        "dkqjrm/augmented-olive-product-phonetic-sentence-wo-negative-v2-bge-m3",
        True,  # MEAN pooling, no Normalize module
    ),
}


def get_model_config(db_name: str) -> tuple[str, bool]:
    """Determine model and normalize setting from db name."""
    for suffix, config in MODEL_MAP.items():
        if db_name.endswith(suffix):
            return config
    raise ValueError(f"Unknown db pattern: {db_name}")


def embed_texts(model: SentenceTransformer, texts: list[str], normalize: bool, batch_size: int = 256) -> np.ndarray:
    """Embed a list of texts."""
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=normalize)


def process_db(input_file: Path, output_dir: Path, models_cache: dict, batch_size: int) -> None:
    """Process one database JSON file."""
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    db_name = data["db_name"]
    model_name, normalize = get_model_config(db_name)

    print(f"\n{'='*60}")
    print(f"DB: {db_name}")
    print(f"Model: {model_name}")
    print(f"Normalize: {normalize}")
    print(f"Chunks: {len(data['chunks'])}, Queries: {len(data['queries'])}")

    # Load or reuse model
    if model_name not in models_cache:
        print(f"Loading model: {model_name}")
        models_cache[model_name] = SentenceTransformer(model_name, device="cuda")
    model = models_cache[model_name]

    result = {"db_name": db_name, "model_name": model_name, "normalize": normalize}

    # Embed chunks
    if data["chunks"]:
        chunk_ids = [c["id"] for c in data["chunks"]]
        chunk_texts = [c["content"] for c in data["chunks"]]
        print(f"Embedding {len(chunk_texts)} chunks...")
        t0 = time.time()
        chunk_embs = embed_texts(model, chunk_texts, normalize, batch_size)
        print(f"  Done in {time.time() - t0:.1f}s")
        result["chunk_ids"] = chunk_ids
        result["chunk_embeddings_shape"] = list(chunk_embs.shape)

        # Save as numpy
        np.save(output_dir / f"{db_name}_chunk_embs.npy", chunk_embs)
        with open(output_dir / f"{db_name}_chunk_ids.json", "w") as f:
            json.dump(chunk_ids, f)

    # Embed queries
    if data["queries"]:
        query_ids = [q["id"] for q in data["queries"]]
        query_texts = [q["query_text"] for q in data["queries"]]
        print(f"Embedding {len(query_texts)} queries...")
        t0 = time.time()
        query_embs = embed_texts(model, query_texts, normalize, batch_size)
        print(f"  Done in {time.time() - t0:.1f}s")
        result["query_ids"] = query_ids
        result["query_embeddings_shape"] = list(query_embs.shape)

        np.save(output_dir / f"{db_name}_query_embs.npy", query_embs)
        with open(output_dir / f"{db_name}_query_ids.json", "w") as f:
            json.dump(query_ids, f)

    # Quick sanity check: cosine between first query and first chunk
    if data["chunks"] and data["queries"]:
        cos = np.dot(query_embs[0], chunk_embs[0])
        print(f"  Sanity check — Q0↔C0 cosine: {cos:.4f}")

    # Save metadata
    with open(output_dir / f"{db_name}_meta.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Output saved to {output_dir}/{db_name}_*")


def main():
    parser = argparse.ArgumentParser(description="GPU embedding for AutoRAG databases")
    parser.add_argument("--input-dir", required=True, help="Directory with exported JSON files")
    parser.add_argument("--output-dir", required=True, help="Directory to save embeddings")
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    parser.add_argument("--db", type=str, default=None, help="Process only this database (by name)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_cache: dict[str, SentenceTransformer] = {}

    json_files = sorted(input_dir.glob("*.json"))
    if args.db:
        json_files = [f for f in json_files if f.stem == args.db]

    print(f"Found {len(json_files)} database files to process")

    for jf in json_files:
        try:
            process_db(jf, output_dir, models_cache, args.batch_size)
        except Exception as e:
            print(f"ERROR processing {jf.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("All done!")


if __name__ == "__main__":
    main()
