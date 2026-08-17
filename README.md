# PeopleFlow RAG Support

PeopleFlow RAG Support is a Retrieval-Augmented Generation (RAG) system for a fictional HR SaaS platform.

The system processes an internal FAQ knowledge base, generates embeddings, stores them in FAISS, retrieves the most relevant chunks for a user question, and generates a grounded answer with an LLM.

## Why RAG

This project uses RAG because the support knowledge lives in internal documentation that changes frequently: the system first retrieves the most relevant chunks and only then generates the answer, so the LLM responds from company documentation instead of its training data.

This two-step approach provides:

* **Updatable knowledge**: editing `faq_document.txt` and rebuilding the index updates the chatbot, with no model fine-tuning required.
* **Transparency and source attribution**: every answer includes the `chunks_related` used to generate it, so each response can be audited.
* **Reduced hallucination**: the prompt restricts the model to the retrieved context, and it abstains when the answer is not in the knowledge base.

## Project Structure

```text
peopleflow-rag-support/
│
├── data/
│   ├── faq_document.txt
│   └── index/
│       ├── faiss.index
│       └── chunks.json
│
├── outputs/
│   └── sample_queries.json
│
├── src/
│   ├── chunking.py
│   ├── embeddings.py
│   ├── build_index.py
│   ├── retriever.py
│   └── query.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Python 3.13 was used for this project.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Git Bash on Windows:

```bash
source .venv/Scripts/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example`:

```env
OPENAI_API_KEY=your-key-here
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

## Build the Vector Index

Run:

```bash
python -m src.build_index
```

This pipeline:

1. Loads `data/faq_document.txt`
2. Splits the document into FAQ-aware chunks
3. Generates embeddings
4. Normalizes the vectors
5. Builds a FAISS index
6. Saves the index and chunk metadata

Generated files:

```text
data/index/faiss.index
data/index/chunks.json
```

The current knowledge base generates 51 chunks.

## Run a Query

Pass the question as a command-line argument:

```bash
python -m src.query "I forgot my password and cannot access my account."
```

Or run it in interactive mode:

```bash
python -m src.query
```

```text
Enter your question: I forgot my password and cannot access my account.
```

The system:

1. Generates an embedding for the question
2. Performs a vector search in FAISS
3. Retrieves the top 3 relevant chunks
4. Sends those chunks to the LLM as context
5. Returns structured JSON

Example output:

```json
{
  "user_question": "I forgot my password and cannot access my account.",
  "system_answer": "To reset your forgotten password, select “Forgot password” on the PeopleFlow HR login page...",
  "chunks_related": [
    {
      "chunk_id": 1,
      "section": "ACCOUNT ACCESS AND SECURITY",
      "question": "How can a user reset a forgotten password?",
      "similarity": 0.457
    }
  ]
}
```

## Technical Decisions

### Chunking

Three strategies were tested:

| Strategy        | Chunks | Average size |
| --------------- | -----: | -----------: |
| Fixed-size      |     43 |    146 words |
| Paragraph-based |     38 |  132.8 words |
| FAQ-aware       |     51 |   91.6 words |

FAQ-aware chunking was selected because each question and answer remains together as one semantic unit.

No overlap was required because the FAQs are already self-contained and the largest chunk is only 118 words.

### Embeddings

The project uses:

```text
text-embedding-3-small
```

Each chunk is embedded using its FAQ question and answer together.

### Vector Search

FAISS is used with:

```python
faiss.IndexFlatIP
```

Vectors are L2-normalized before being stored and queried, allowing inner product to be used as cosine similarity.

The retrieval method is:

```text
Exact k-NN
k = 3
```

Exact search was selected because the knowledge base contains only 51 chunks.

### RAG Generation

The retrieved chunks are provided to the LLM as context.

The prompt instructs the model to answer only from the provided documentation and to avoid inventing information.

If the knowledge base does not contain enough information, the model responds accordingly.

## Sample Outputs

Three complete query examples are available at:

```text
outputs/sample_queries.json
```

They demonstrate:

* Direct FAQ retrieval
* Retrieval across related chunks
* Abstention when the answer is not available in the knowledge base
