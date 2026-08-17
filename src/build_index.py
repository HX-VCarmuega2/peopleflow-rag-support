import json
import sys
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAIError

from src.chunking import chunk_by_faq, load_document
from src.embeddings import create_embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_DIR = PROJECT_ROOT / "data" / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"


def build_faiss_index(
    embeddings: list[list[float]],
) -> faiss.Index:
    """
    Create a FAISS index using cosine similarity.

    Embeddings are normalized first and then stored in an
    IndexFlatIP index. For normalized vectors, inner product
    is equivalent to cosine similarity.
    """
    if not embeddings:
        raise ValueError("Embeddings list cannot be empty.")

    vectors = np.array(
        embeddings,
        dtype="float32",
    )

    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    return index


def save_chunks(
    chunks: list[dict],
    path: Path = CHUNKS_PATH,
) -> None:
    """
    Save chunk metadata and content as JSON.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )


def save_faiss_index(
    index: faiss.Index,
    path: Path = FAISS_INDEX_PATH,
) -> None:
    """
    Save the FAISS index to disk.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(path),
    )


def main():
    try:
        print("Loading document...")
        document = load_document()

        print("Creating chunks...")
        chunks = chunk_by_faq(document)

        print(f"Generated chunks: {len(chunks)}")

        print("Generating embeddings...")
        embeddings = create_embeddings(chunks)

    except FileNotFoundError as error:
        print(f"Missing source document: {error}")
        sys.exit(1)

    except ValueError as error:
        print(f"Invalid data: {error}")
        sys.exit(1)

    except OpenAIError as error:
        print(f"OpenAI API error: {error}")
        sys.exit(1)

    print(f"Generated embeddings: {len(embeddings)}")

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Saving FAISS index...")
    save_faiss_index(index)

    print("Saving chunks...")
    save_chunks(chunks)

    print("\n" + "=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)

    print(f"Chunks: {len(chunks)}")
    print(f"Vectors in FAISS: {index.ntotal}")
    print(f"Embedding dimensions: {index.d}")
    print(f"FAISS index: {FAISS_INDEX_PATH}")
    print(f"Chunks file: {CHUNKS_PATH}")


if __name__ == "__main__":
    main()