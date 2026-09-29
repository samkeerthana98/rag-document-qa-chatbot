# RAG Document Q&A Chatbot — Kiro Development Prompts

This file documents the development workflow and phase-by-phase goals used to
build the RAG Document Q&A Chatbot using Kiro (AI-assisted development).

---

## Development Approach

The project was built incrementally across 13 phases using the following principles:

- **Phase-based increments** — each phase had a single, well-scoped goal
- **Inspect before changing** — read existing code and tests before modifying anything
- **Test after every phase** — verify correctness before moving forward
- **Minimal surface area** — avoid unnecessary abstractions, dependencies, or features
- **Portfolio-ready throughout** — each phase left the project in a clean, explainable state

---

## Phase Prompts

### Phase 1 — Flask Foundation

**Purpose:** Establish the project skeleton.

**Goal:** Create the repository structure, virtual environment, `requirements.txt`,
Flask app factory (`create_app`), a `/api/health` endpoint returning `{"status": "healthy"}`,
`run.py` entry point, `.env.example`, and `.gitignore`. Verify with a smoke test.

---

### Phase 2 — Flask Application Layer

**Purpose:** Set up configuration and routing infrastructure.

**Goal:** Implement environment-based configuration in `app/config.py`, wire the
Flask app factory to load config from environment variables, and confirm the
health endpoint works with the config in place.

---

### Phase 3 — PDF Processing

**Purpose:** Accept and validate PDF uploads.

**Goal:** Implement `POST /api/upload` — multipart file upload with extension
validation, magic-byte checking, size limiting, PyPDF text extraction, and
LangChain chunking. Return chunk count on success; return structured error
responses for missing file, wrong type, oversized file, or unextractable PDF.

---

### Phase 4 — Embeddings + ChromaDB

**Purpose:** Store document chunks as searchable vectors.

**Goal:** Implement `app/rag/embeddings.py` using HuggingFace
`sentence-transformers/all-MiniLM-L6-v2` (cached singleton) and
`app/rag/vector_store.py` using ChromaDB with cosine similarity, persistent
storage, and upsert-on-re-upload deduplication.

---

### Phase 5 — Retriever

**Purpose:** Retrieve the most relevant document chunks for a query.

**Goal:** Implement `app/rag/retriever.py` — embed the question, query
ChromaDB for top-K chunks by cosine similarity, return chunks with metadata
(filename, page number).

---

### Phase 6 — Ollama + Q&A

**Purpose:** Generate grounded answers using a local LLM.

**Goal:** Implement `app/rag/llm.py` wrapping LangChain's Ollama integration
(`llama3.2`), `app/services/qa_service.py` orchestrating the full RAG pipeline
(retrieve → prompt → generate), and `POST /api/ask` returning an answer plus
source citations. Handle Ollama-not-running gracefully with a `503` response.

---

### Phase 7 — Frontend

**Purpose:** Provide a usable single-page UI.

**Goal:** Build `templates/index.html`, `static/style.css`, and
`static/script.js` — drag-and-drop PDF upload, question input, answer display
with source citations. Use vanilla HTML/CSS/JavaScript; no frontend framework.
All DOM manipulation must be XSS-safe (no `innerHTML` with user content).

---

### Phase 8 — Docker

**Purpose:** Package the application for consistent, portable deployment.

**Goal:** Write a production `Dockerfile` (Python 3.11-slim base, Gunicorn WSGI
server, non-root user, `HEALTHCHECK` on `/api/health`) and `.dockerignore`.
Ollama runs on the host separately; the container connects via `OLLAMA_BASE_URL`.
Verify the image builds and the health endpoint responds.

---

### Phase 9 — AWS Deployment Preparation

**Purpose:** Document a reproducible path to deploy on AWS EC2.

**Goal:** Write `DEPLOYMENT.md` covering: EC2 instance selection (`t3.large`
minimum), Ollama installation on the host, Docker run command with volume
mounts for uploads and ChromaDB persistence, security group configuration
(port 22 SSH + port 5000 app; port 11434 Ollama not exposed), and
troubleshooting steps.

---

### Phase 10 — Security + Production Readiness

**Purpose:** Harden the application against common web vulnerabilities.

**Goal:** Audit and fix: file upload validation (extension + magic bytes),
input length limits on the question endpoint, no secret leakage in error
responses, `SECRET_KEY` handled via environment variable, `FLASK_DEBUG=false`
enforced in Docker, and no sensitive data committed to the repository.

---

### Phase 11 — CI/CD

**Purpose:** Automate testing and build validation on every change.

**Goal:** Create `.github/workflows/ci.yml` with two parallel jobs: (1) `test`
— install dependencies and run `pytest -v --tb=short` on Python 3.11 with pip
caching; (2) `docker` — build the image with GitHub Actions layer cache to
catch Dockerfile regressions. No Ollama required; all LLM calls are mocked.

---

### Phase 12 — Documentation + Portfolio Readiness

**Purpose:** Make the repository clear, accurate, and portfolio-ready.

**Goal:** Rewrite `README.md` to be comprehensive and self-contained: project
overview, architecture diagram, technology stack table, installation steps,
environment variable reference, API documentation, Docker instructions, AWS
deployment summary, CI/CD description, and development status table.
Verify all documentation matches the actual implementation.

---

### Phase 13 — Final Release Verification

**Purpose:** Confirm the project is clean and ready for the initial Git push.

**Goal:** Run the full test suite (329 tests, 0 failures), verify the Docker
image builds successfully, confirm `.gitignore` and `.dockerignore` exclude all
generated/sensitive files (`.env`, `uploads/`, `chroma_db/`, `.venv/`,
`__pycache__/`, caches), verify no secrets or large generated files are staged,
and confirm `README.md` and `DEPLOYMENT.md` are accurate. Create the initial
commit and push to GitHub.

---

## Important Design Constraints

These constraints were established at project start and maintained throughout:

| Constraint | Decision |
|---|---|
| LLM | Ollama + `llama3.2` — fully local, no API key |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` — local |
| Vector database | ChromaDB — local, persistent, cosine similarity |
| Backend | Python 3.11 + Flask 3 |
| LLM integration | LangChain + langchain-ollama |
| PDF processing | PyPDF + LangChain text splitters |
| Frontend | Vanilla HTML, CSS, JavaScript — no framework |
| Containerisation | Docker with Gunicorn, non-root user, health check |
| Cloud target | AWS EC2 single-instance |
| CI/CD | GitHub Actions |
| No external LLM API | OpenAI, Anthropic, etc. are explicitly excluded |
| No unnecessary infra | No Kubernetes, no message queues, no managed services |

---

## Final Development Principles

- **Incremental changes** — each phase built on a verified foundation
- **Preserve working functionality** — never break passing tests while adding features
- **Security-first** — uploads validated, secrets in environment variables, non-root container
- **Tests before release** — 329 tests covering every pipeline layer, all running offline
- **Documentation consistency** — README, DEPLOYMENT.md, and code comments kept in sync
- **No unnecessary features** — scope was held to what is demonstrable and explainable
