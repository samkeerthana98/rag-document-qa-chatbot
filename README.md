# RAG Document Q&A Chatbot

A portfolio project demonstrating **Retrieval-Augmented Generation (RAG)** using
a fully local stack — no OpenAI API key, no paid services, no external LLM API.

Upload a PDF, ask questions about it, and get answers grounded in the document
content — powered by a local LLM (Ollama + llama3.2) and local embeddings
(HuggingFace sentence-transformers).

![CI](https://github.com/samkeerthana98/rag-document-qa-chatbot/actions/workflows/ci.yml/badge.svg)

---

## Key Features

- **Fully local** — LLM, embeddings, and vector database all run on your machine
- **PDF upload** with drag-and-drop, extension validation, and magic-byte checking
- **Semantic search** over document chunks using cosine-similarity embeddings
- **Grounded answers** — the LLM is instructed to use only the uploaded document context
- **Source citations** — every answer includes the filename and page number it came from
- **Re-upload deduplication** — uploading the same PDF twice replaces rather than duplicates
- **Persistent vector store** — ChromaDB data survives restarts via Docker volume
- **Dockerised** — production-ready image with Gunicorn, non-root user, health check
- **AWS EC2 ready** — single-instance deployment guide included
- **329 tests** covering every pipeline layer, all running offline (no live Ollama needed)
- **GitHub Actions CI** — tests and Docker build on every push and PR

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │          RAG Pipeline                │
                        │                                      │
  PDF Upload            │  1. PDF → text extraction (PyPDF)   │
  ──────────►  Flask    │  2. Text → chunks (LangChain)       │
              routes    │  3. Chunks → embeddings (HF model)  │
                        │  4. Embeddings → ChromaDB (store)   │
                        └─────────────────────────────────────┘

                        ┌─────────────────────────────────────┐
                        │          Q&A Pipeline                │
                        │                                      │
  Question              │  1. Question → embedding (HF model) │
  ──────────►  Flask    │  2. Embedding → top-K chunks        │
              routes    │     (ChromaDB cosine similarity)    │
                        │  3. Chunks + question → prompt      │
                        │  4. Prompt → Ollama (llama3.2)      │
                        │  5. Answer + source pages ◄─────────┘
                        └─────────────────────────────────────┘
```

The Flask app and Ollama are kept **separate**: Ollama runs on the host (or EC2
instance directly), and the Flask container connects to it via
`OLLAMA_BASE_URL`. This keeps the Ollama model outside the application image,
so the LLM can be upgraded independently without rebuilding or redeploying the app.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11+, Flask 3 |
| WSGI Server | Gunicorn (production) |
| PDF Processing | PyPDF, LangChain text splitters |
| Embeddings | HuggingFace sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) |
| Vector Database | ChromaDB (local, persistent, cosine similarity) |
| LLM | Ollama (`llama3.2`) |
| LLM Integration | LangChain, langchain-ollama |
| Frontend | HTML, CSS, Vanilla JavaScript (no framework) |
| Testing | pytest, pytest-flask (329 tests, all offline) |
| Containerisation | Docker (non-root user, health check, volume mounts) |
| CI/CD | GitHub Actions |
| Cloud | AWS EC2 |

---

## Project Structure

```
rag-document-qa-chatbot/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions: test + Docker build
├── app/
│   ├── __init__.py             # Flask app factory (create_app)
│   ├── routes.py               # API endpoints
│   ├── config.py               # Environment-based configuration
│   ├── rag/
│   │   ├── document_processor.py   # PDF loading, extraction, chunking
│   │   ├── embeddings.py           # HuggingFace embedding model (cached)
│   │   ├── vector_store.py         # ChromaDB operations (upsert, query)
│   │   ├── retriever.py            # Cosine similarity search, top-K
│   │   └── llm.py                  # Ollama LLM wrapper + error handling
│   └── services/
│       └── qa_service.py           # RAG pipeline orchestrator
├── templates/
│   └── index.html                  # Single-page frontend
├── static/
│   ├── style.css
│   └── script.js                   # Fetch-based API client, XSS-safe
├── tests/                          # 329 pytest tests (all offline)
├── uploads/                        # Uploaded PDFs — git-ignored
├── chroma_db/                      # ChromaDB persistence — git-ignored
├── .dockerignore
├── .env.example                    # Environment variable template
├── .gitignore
├── Dockerfile
├── DEPLOYMENT.md                   # AWS EC2 step-by-step guide
├── requirements.txt                # Pinned dependencies
├── prompts.md                          # Kiro development prompts and workflow
└── run.py                          # Development entry point
```

---

## Prerequisites

- Python 3.11 or higher
- [Ollama](https://ollama.com) installed and running locally
- `llama3.2` model pulled (see [Ollama Setup](#ollama-setup) below)
- Git

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/samkeerthana98/rag-document-qa-chatbot.git
cd rag-document-qa-chatbot

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
# Note: torch + sentence-transformers are ~1.5 GB — first install takes a few minutes
pip install -r requirements.txt

# 4. Copy the environment template
copy .env.example .env      # Windows
cp .env.example .env        # macOS / Linux
```

---

## Ollama Setup

Ollama runs the LLM locally — no API key required. After Ollama and the required model are downloaded, runtime inference runs entirely on your machine with no internet connection needed.

```bash
# 1. Install Ollama from https://ollama.com

# 2. Start the Ollama service (listens on http://localhost:11434)
ollama serve

# 3. Pull the llama3.2 model (~2 GB download, one-time)
ollama pull llama3.2

# 4. Verify it works
ollama run llama3.2 "Hello, are you working?"
```

---

## Running Locally

```bash
# Ollama must be running first (ollama serve)
python run.py
```

Open http://127.0.0.1:5000 in your browser.

**Usage:**
1. Click the upload area or drag a PDF onto it
2. Click **Upload & Process** — the PDF is chunked and indexed into ChromaDB
3. Type a question in the text box and click **Ask**
4. The answer appears with source page references

---

## Environment Variables

Copy `.env.example` to `.env` and adjust as needed. All variables have
sensible defaults for local development.

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `dev-secret-key-...` | Flask session signing key — **change in production** |
| `FLASK_DEBUG` | `true` | Debug mode (set to `false` in Docker/production) |
| `FLASK_PORT` | `5000` | Port the development server listens on |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum PDF upload size in MB |
| `CHUNK_SIZE` | `1000` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap characters between consecutive chunks |
| `TOP_K` | `4` | Number of chunks retrieved per query |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local HuggingFace embedding model |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB persistence directory (use absolute path in Docker) |
| `CHROMA_COLLECTION_NAME` | `documents` | ChromaDB collection name |
| `LLM_PROVIDER` | `ollama` | LLM provider (only `ollama` supported) |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama service URL |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace model cache directory |

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web interface |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/upload` | Upload and index a PDF |
| `POST` | `/api/ask` | Ask a question about indexed documents |

### GET /api/health

```json
{ "status": "healthy" }
```

### POST /api/upload

Request: `multipart/form-data` with a `file` field containing a PDF.

```json
{
  "message": "Document processed successfully",
  "filename": "example.pdf",
  "chunks_processed": 42
}
```

Error responses: `400` (missing/invalid file), `413` (file too large),
`422` (valid PDF but no extractable text), `500` (unexpected error).

### POST /api/ask

```json
{ "question": "What is the main topic of this document?" }
```

Response:

```json
{
  "answer": "The document covers ...",
  "sources": [
    { "filename": "example.pdf", "page": 3 }
  ]
}
```

Error responses: `400` (missing/empty question, question > 2000 chars),
`503` (Ollama not running), `500` (unexpected error).

---

## Running Tests

All 329 tests run fully offline — Ollama does not need to be running.

```bash
pytest

# Verbose output
pytest -v

# Specific test file
pytest tests/test_security.py -v
```

The test suite covers: document processing, embeddings, vector store, retrieval,
Q&A service, LLM wrapper, routes, multi-document scenarios, and security.

---

## Docker

The Docker image bundles Flask, Gunicorn, sentence-transformers, and ChromaDB.
Ollama runs on the host separately — it is **not** included in the image.

```bash
# Build
docker build -t rag-chatbot .

# Run (Docker Desktop — macOS/Windows)
docker run -p 5000:5000 \
  -e SECRET_KEY=your-secret-key \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v hf-cache:/app/.cache/huggingface \
  rag-chatbot

# Run (Linux host — use --add-host instead of host.docker.internal)
docker run -p 5000:5000 \
  -e SECRET_KEY=your-secret-key \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v hf-cache:/app/.cache/huggingface \
  --add-host=host.docker.internal:host-gateway \
  rag-chatbot
```

> **Windows note:** `$(pwd)` is bash syntax. In PowerShell use `${PWD}`;
> in Command Prompt use `%cd%`. Example: `-v ${PWD}/uploads:/app/uploads`

Key Docker design decisions:
- Gunicorn with 2 workers + 2 threads (I/O-concurrency for upload and LLM calls)
- Non-root user (`appuser`, uid 1001) inside the container
- `HF_HOME=/app/.cache/huggingface` — embedding model cached in a named volume
- `FLASK_DEBUG=false` set in the image ENV block
- Docker `HEALTHCHECK` polls `/api/health` every 30 s

---

## AWS Deployment

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for the complete step-by-step guide.

Summary:
- Single EC2 instance (`t3.large` minimum, `t3.xlarge` recommended)
- Ollama + llama3.2 run directly on the host (not in Docker)
- The Flask app runs in a Docker container on the same instance
- Container connects to Ollama via `OLLAMA_BASE_URL=http://host.docker.internal:11434`
- ChromaDB data and uploads persist on the host via Docker volumes
- Security group: port 22 (SSH, your IP only) and port 5000 (app)
- Port 11434 (Ollama) is **not** exposed to the internet

---

## CI/CD

GitHub Actions runs on every push and pull request to `main`/`master`.

**`test` job** — Python 3.11, pip cache keyed on `requirements.txt` hash:
1. Install dependencies
2. Run `pytest -v --tb=short` (329 tests, no Ollama needed)

**`docker` job** — independent of the test job:
1. Set up Docker Buildx
2. Build the image with GitHub Actions layer cache (`type=gha`)
3. Verify the build succeeds (image is not pushed or run)

Both jobs run in parallel; a PR shows both results independently.

---

## Limitations and Future Improvements

**Current limitations:**

- Single LLM provider — only Ollama is supported; adding OpenAI/Anthropic would require changes to `app/rag/llm.py` only
- No authentication — the app is open to anyone who can reach port 5000
- In-process ChromaDB — scales to a single instance; a multi-instance deployment would need a shared vector store
- `llama3.2` quality — answers are only as good as the model; a larger model (llama3.1:70b, etc.) would improve quality at the cost of hardware requirements
- No document deletion endpoint — uploaded documents accumulate; a `DELETE /api/documents/{filename}` endpoint is the natural next step
- PDF only — DOCX, TXT, and HTML are natural extensions (LangChain supports them)
- No streaming — the full LLM response is buffered before being returned; streaming would improve perceived latency

**Straightforward next steps:**

- Add streaming responses via Server-Sent Events
- Add a document management UI (list and delete indexed files)
- Add HTTPS via nginx reverse proxy or AWS ALB
- Store `SECRET_KEY` in AWS Secrets Manager
- Add CloudWatch log shipping from Docker

---

## Development Status

| Phase | Description | Status |
|---|---|---|
| 1 | Project foundation | ✅ Complete |
| 2 | Flask app + health endpoint | ✅ Complete |
| 3 | PDF upload endpoint | ✅ Complete |
| 4 | HuggingFace embeddings + ChromaDB | ✅ Complete |
| 5 | Retrieval | ✅ Complete |
| 6 | Ollama LLM + Q&A endpoint | ✅ Complete |
| 7 | Frontend UI | ✅ Complete |
| 8 | Docker | ✅ Complete |
| 9 | AWS EC2 deployment docs | ✅ Complete |
| 10 | Security review | ✅ Complete |
| 11 | CI/CD (GitHub Actions) | ✅ Complete |
| 12 | Final documentation | ✅ Complete |
| 13 | Final release verification | ✅ Complete |
