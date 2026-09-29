"""
app/services/qa_service.py — RAG question-answering orchestrator.

This service wires together all the RAG pipeline components to answer a
user's question:

  question
    → retrieve relevant chunks (retriever.py)
    → build a grounded prompt (here)
    → call the LLM (llm.py)
    → return answer + source references

Interview explanation:
  This is the "Augmented Generation" part of RAG.  The key design decision
  is the prompt template: we explicitly instruct the LLM to answer ONLY from
  the provided context.  Without this instruction, the LLM would blend
  document content with its own general knowledge, which defeats the purpose
  of RAG.

  Why a service layer?
    The Flask route should only handle HTTP parsing and response formatting.
    The RAG pipeline components (retriever, LLM) should be framework-agnostic.
    The service layer sits between them — it has no Flask code and no HTTP
    logic, making it easy to test, reuse, and reason about independently.

Public API:
  answer_question(question, collection, embedding_model, llm, top_k)
      → QAResult

  QAResult:
      answer:  str         — The LLM's grounded answer
      sources: list[dict]  — [{"filename": "...", "page": N}, ...]
      chunks_used: int     — Number of context chunks sent to the LLM
"""

import logging
from dataclasses import dataclass, field
from typing import List

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from app.rag.llm import LLMUnavailableError, invoke_llm
from app.rag.retriever import RetrievedChunk, retrieve, unique_sources

logger = logging.getLogger(__name__)


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class QAResult:
    """The result of a RAG question-answering request.

    Attributes:
        answer:      The LLM's answer, grounded in the retrieved context.
        sources:     List of source references (filename + page number).
        chunks_used: How many document chunks were sent as context.
    """

    answer: str
    sources: List[dict] = field(default_factory=list)
    chunks_used: int = 0


# ── Prompt template ───────────────────────────────────────────────────────────

def build_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    """Assemble the prompt sent to the LLM.

    The prompt has three parts:
      1. A system instruction telling the LLM to answer from context only.
      2. The retrieved document context (chunks joined with separators).
      3. The user's question.

    Why this structure?
      LLMs are instruction-following models.  By explicitly saying "use only
      the context below" and "if not found, say so", we ground the answer in
      the uploaded document rather than the model's training data.

    Args:
        question: The user's natural-language question.
        chunks:   Retrieved document chunks from the retriever.

    Returns:
        A single string to pass to llm.invoke().
    """
    # Format each chunk with its source reference so the LLM can see
    # where the information comes from.
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Source {i}: {chunk.source}, page {chunk.page_number}]\n{chunk.text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant that answers questions based strictly on the provided document context.

INSTRUCTIONS:
- Answer the question using ONLY the information in the context below.
- Do not use your general knowledge or information outside of the context.
- If the answer is not found in the context, respond with: "I could not find the answer to that question in the uploaded document."
- Be concise and precise.
- Cite the source and page number when relevant.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    return prompt


# ── Core Q&A function ─────────────────────────────────────────────────────────

def answer_question(
    question: str,
    collection: chromadb.Collection,
    embedding_model: HuggingFaceEmbeddings,
    llm: OllamaLLM,
    top_k: int = 4,
) -> QAResult:
    """Answer a question using the full RAG pipeline.

    Steps:
      1. Retrieve the most relevant document chunks for the question.
      2. Handle the no-documents case early (return a helpful message).
      3. Build a grounded prompt from the chunks and question.
      4. Call the LLM and get the answer.
      5. Return the answer with deduplicated source references.

    Args:
        question:        The user's natural-language question.
        collection:      An open ChromaDB Collection with stored document chunks.
        embedding_model: The loaded HuggingFace embedding model.
        llm:             A configured OllamaLLM instance.
        top_k:           Number of chunks to retrieve (from app config).

    Returns:
        A QAResult with the answer, source references, and chunk count.

    Raises:
        ValueError:          If the question is empty.
        LLMUnavailableError: If Ollama is not running.
        Exception:           Any other unexpected error.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    logger.info("Processing question: '%s'", question[:80])

    # ── Step 1: Retrieve relevant chunks ──────────────────────────────────────
    chunks = retrieve(
        collection=collection,
        question=question,
        embedding_model=embedding_model,
        top_k=top_k,
    )

    # ── Step 2: Handle no documents case ──────────────────────────────────────
    # retrieve() returns [] when the collection is empty.  We return a clear
    # message rather than calling the LLM with an empty context, which would
    # cause it to hallucinate.
    if not chunks:
        logger.warning("No document chunks found — collection is empty")
        return QAResult(
            answer=(
                "No documents have been uploaded yet. "
                "Please upload a PDF document before asking questions."
            ),
            sources=[],
            chunks_used=0,
        )

    logger.info("Retrieved %d chunk(s) for context", len(chunks))

    # ── Step 3: Build the grounded prompt ─────────────────────────────────────
    prompt = build_prompt(question, chunks)
    logger.debug("Prompt length: %d chars", len(prompt))

    # ── Step 4: Call the LLM ─────────────────────────────────────────────────
    # invoke_llm raises LLMUnavailableError if Ollama is down.
    # We let it propagate — the route handles it with a 503 response.
    answer = invoke_llm(llm, prompt)

    # ── Step 5: Build source references ──────────────────────────────────────
    sources = unique_sources(chunks)

    logger.info(
        "Answer generated (%d chars) from %d chunk(s), %d source page(s)",
        len(answer),
        len(chunks),
        len(sources),
    )

    return QAResult(answer=answer, sources=sources, chunks_used=len(chunks))
