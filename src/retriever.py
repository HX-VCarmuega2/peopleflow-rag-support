import json
import sys
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAIError

from src.embeddings import create_embedding


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_DIR = PROJECT_ROOT / "data" / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"


def load_faiss_index(
    path: Path = FAISS_INDEX_PATH,
) -> faiss.Index:
    """
    Load the persisted FAISS index.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {path}"
        )

    return faiss.read_index(str(path))


def load_chunks(
    path: Path = CHUNKS_PATH,
) -> list[dict]:
    """
    Load persisted FAQ chunks from JSON.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def search_chunks(
    question: str,
    k: int = 3,
) -> list[dict]:
    """
    Search the FAISS index and return the most relevant chunks.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    index = load_faiss_index()
    chunks = load_chunks()

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    k = min(k, index.ntotal)

    query_embedding = create_embedding(question)

    query_vector = np.array(
        [query_embedding],
        dtype="float32",
    )

    # The stored vectors were normalized when the index was built.
    # The query must also be normalized for cosine similarity.
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(
        query_vector,
        k,
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):
        if index_position == -1:
            continue

        chunk = chunks[index_position]

        results.append(
            {
                "chunk_id": chunk["id"],
                "section": chunk["section"],
                "question": chunk["question"],
                "text": chunk["text"],
                "similarity": float(score),
            }
        )

    return results


def main():
    question = input(
        "Enter your question: "
    ).strip()

    try:
        results = search_chunks(
            question,
            k=3,
        )

    except ValueError as error:
        print(f"Invalid input: {error}")
        sys.exit(1)

    except FileNotFoundError as error:
        print(f"Missing index files: {error}")
        print("Run 'python -m src.build_index' first.")
        sys.exit(1)

    except OpenAIError as error:
        print(f"OpenAI API error: {error}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("TOP MATCHING CHUNKS")
    print("=" * 60)

    for position, result in enumerate(
        results,
        start=1,
    ):
        print(f"\nResult #{position}")
        print("-" * 60)

        print(
            f"Chunk ID: {result['chunk_id']}"
        )

        print(
            f"Section: {result['section']}"
        )

        print(
            f"FAQ: {result['question']}"
        )

        print(
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )

        print("\nAnswer:")
        print(result["text"])


if __name__ == "__main__":
    main()