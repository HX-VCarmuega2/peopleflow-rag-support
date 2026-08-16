import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.retriever import search_chunks


load_dotenv()

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gpt-4o-mini",
)

client = OpenAI()


def build_context(chunks: list[dict]) -> str:
    """
    Build the context that will be provided to the LLM
    from the retrieved FAQ chunks.
    """
    context_parts = []

    for chunk in chunks:
        context_parts.append(
            f"""
Chunk ID: {chunk["chunk_id"]}
Section: {chunk["section"]}
FAQ: {chunk["question"]}
Content:
{chunk["text"]}
""".strip()
        )

    return "\n\n---\n\n".join(context_parts)


def generate_answer(
    question: str,
    chunks: list[dict],
) -> str:
    """
    Generate an answer using only the retrieved FAQ context.
    """
    context = build_context(chunks)

    system_prompt = """
You are a customer support assistant for PeopleFlow HR.

Answer the user's question using ONLY the information contained
in the provided context.

Rules:
- Do not invent policies, procedures, limits, or product behavior.
- If the context does not contain enough information to answer,
  clearly say that the answer is not available in the knowledge base.
- Be concise but complete.
- Prefer direct instructions when the context provides a procedure.
- Do not mention that you are an AI.
""".strip()

    user_prompt = f"""
CONTEXT:

{context}

USER QUESTION:

{question}
""".strip()

    response = client.responses.create(
        model=LLM_MODEL,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        max_output_tokens=300,
    )

    return response.output_text.strip()


def query_rag(
    question: str,
    k: int = 3,
) -> dict:
    """
    Run the complete RAG query pipeline.

    Steps:
    1. Retrieve relevant chunks.
    2. Generate an answer with the LLM.
    3. Return structured output.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    chunks = search_chunks(
        question,
        k=k,
    )

    answer = generate_answer(
        question,
        chunks,
    )

    return {
        "user_question": question,
        "system_answer": answer,
        "chunks_related": chunks,
    }


def main():
    question = input(
        "Enter your question: "
    ).strip()

    result = query_rag(question)

    print("\n" + "=" * 60)
    print("RAG RESPONSE")
    print("=" * 60)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()