/**
 * script.js — RAG Document Q&A Chatbot frontend
 *
 * Responsibilities:
 *   - Handle PDF drag-and-drop and file input selection
 *   - POST the selected file to /api/upload
 *   - Show upload progress, success, and error feedback
 *   - Track which documents have been indexed
 *   - POST a question to /api/ask
 *   - Display the answer and source references
 *   - Show loading states and user-friendly error messages
 *   - Client-side validation (empty file, empty question)
 *
 * No frameworks. No build step. Pure ES6+ in a single file.
 */

"use strict";

// ── DOM references ───────────────────────────────────────────

const dropZone        = document.getElementById("drop-zone");
const fileInput       = document.getElementById("file-input");
const fileSelected    = document.getElementById("file-selected");
const selectedFilename = document.getElementById("selected-filename");
const clearFileBtn    = document.getElementById("clear-file");
const uploadBtn       = document.getElementById("upload-btn");
const uploadStatus    = document.getElementById("upload-status");

const docsSection     = document.getElementById("docs-section");
const docList         = document.getElementById("doc-list");

const questionInput   = document.getElementById("question-input");
const charCount       = document.getElementById("char-count");
const askBtn          = document.getElementById("ask-btn");

const answerContainer = document.getElementById("answer-container");
const answerLoading   = document.getElementById("answer-loading");
const answerResult    = document.getElementById("answer-result");
const answerText      = document.getElementById("answer-text");
const sourcesBlock    = document.getElementById("sources-block");
const sourcesList     = document.getElementById("sources-list");
const answerError     = document.getElementById("answer-error");


// ── State ────────────────────────────────────────────────────

/** Currently selected File object, or null. */
let selectedFile = null;

/** Set of filenames that have been successfully indexed. */
const indexedDocs = new Set();


// ── Helpers ──────────────────────────────────────────────────

/**
 * Show a status message below an element.
 * @param {HTMLElement} el    - The status container element.
 * @param {string}      text  - Message to display.
 * @param {"success"|"error"|"info"} type
 */
function showStatus(el, text, type) {
    el.textContent = text;
    el.className = `status-message ${type}`;
    el.hidden = false;
}

/** Hide a status message element. */
function hideStatus(el) {
    el.hidden = true;
    el.textContent = "";
    el.className = "status-message";
}

/**
 * Set a button into loading state (shows spinner, hides label, disables).
 * @param {HTMLElement} btn
 */
function setLoading(btn, loading) {
    const label   = btn.querySelector(".btn-label");
    const spinner = btn.querySelector(".btn-spinner");
    if (loading) {
        btn.disabled = true;
        label.hidden  = true;
        spinner.hidden = false;
    } else {
        label.hidden  = false;
        spinner.hidden = true;
        // Re-enable is handled by the caller based on application state.
    }
}

/** Update the char counter for the question textarea. */
function updateCharCount() {
    const len = questionInput.value.length;
    const max = parseInt(questionInput.getAttribute("maxlength"), 10);
    charCount.textContent = `${len} / ${max}`;
    charCount.className = "char-count";
    if (len >= max)           charCount.className += " at-limit";
    else if (len >= max * 0.9) charCount.className += " near-limit";
}

/** Show or hide the indexed-documents card. */
function refreshDocsSection() {
    if (indexedDocs.size === 0) {
        docsSection.hidden = true;
        return;
    }
    docsSection.hidden = false;
    docList.innerHTML = "";
    indexedDocs.forEach((name) => {
        const li = document.createElement("li");
        li.className = "doc-item";
        li.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                <polyline points="14 2 14 8 20 8"/>
            </svg>
            <span title="${escapeHtml(name)}">${escapeHtml(name)}</span>`;
        docList.appendChild(li);
    });
}

/** Escape HTML special characters to prevent XSS. */
function escapeHtml(str) {
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


// ── File selection ───────────────────────────────────────────

/**
 * Apply the chosen file to the UI state.
 * Called by both the file input change handler and drag-and-drop.
 */
function applySelectedFile(file) {
    if (!file) return;

    // Client-side type check before uploading
    if (!file.name.toLowerCase().endsWith(".pdf")) {
        showStatus(uploadStatus, "Only PDF files are supported. Please choose a .pdf file.", "error");
        clearSelection();
        return;
    }

    selectedFile = file;
    selectedFilename.textContent = file.name;
    fileSelected.hidden = false;
    uploadBtn.disabled  = false;
    hideStatus(uploadStatus);
}

/** Reset file selection back to empty state. */
function clearSelection() {
    selectedFile = null;
    fileInput.value = "";
    fileSelected.hidden = true;
    selectedFilename.textContent = "No file selected";
    uploadBtn.disabled = true;
}

// File input change (click to browse)
fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        applySelectedFile(fileInput.files[0]);
    }
});

// Clear selected file
clearFileBtn.addEventListener("click", (e) => {
    e.stopPropagation(); // don't re-open file picker
    clearSelection();
    hideStatus(uploadStatus);
});

// Drag-and-drop
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        applySelectedFile(files[0]);
    }
});


// ── Upload ───────────────────────────────────────────────────

uploadBtn.addEventListener("click", handleUpload);

async function handleUpload() {
    if (!selectedFile) {
        showStatus(uploadStatus, "Please select a PDF file first.", "error");
        return;
    }

    setLoading(uploadBtn, true);
    hideStatus(uploadStatus);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch("/api/upload", {
            method: "POST",
            body: formData,
            // Don't set Content-Type manually — browser sets it with boundary.
        });

        const data = await response.json();

        if (response.ok) {
            const chunks = data.chunks_processed ?? "?";
            // showStatus uses el.textContent, so the browser handles escaping
            // automatically. Do NOT call escapeHtml here — it would cause
            // characters like & to render as &amp; in the status message.
            showStatus(
                uploadStatus,
                `✓ "${data.filename}" processed successfully — ${chunks} chunk(s) indexed.`,
                "success"
            );
            indexedDocs.add(data.filename);
            refreshDocsSection();
            clearSelection();
        } else {
            // Server returned an error (400, 413, 422, 500)
            const msg = data.error || "Upload failed. Please try again.";
            showStatus(uploadStatus, msg, "error");
        }
    } catch (err) {
        // Network error or server unreachable
        showStatus(
            uploadStatus,
            "Could not reach the server. Please check your connection and try again.",
            "error"
        );
        console.error("Upload error:", err);
    } finally {
        setLoading(uploadBtn, false);
        // Re-enable only if a file is still selected
        uploadBtn.disabled = (selectedFile === null);
    }
}


// ── Question input ───────────────────────────────────────────

questionInput.addEventListener("input", updateCharCount);

// Allow Ctrl+Enter / Cmd+Enter to submit
questionInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        askBtn.click();
    }
});

updateCharCount(); // initialise counter on load


// ── Ask ──────────────────────────────────────────────────────

askBtn.addEventListener("click", handleAsk);

async function handleAsk() {
    const question = questionInput.value.trim();

    if (!question) {
        showAnswerError("Please enter a question before clicking Ask.");
        questionInput.focus();
        return;
    }

    // Show container and loading skeleton, hide previous results
    answerContainer.hidden  = false;
    answerLoading.hidden    = false;
    answerResult.hidden     = true;
    answerError.hidden      = true;

    setLoading(askBtn, true);

    try {
        const response = await fetch("/api/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });

        const data = await response.json();

        if (response.ok) {
            showAnswer(data.answer, data.sources || []);
        } else if (response.status === 503) {
            showAnswerError(
                "The Ollama LLM service is not available. " +
                "Please make sure Ollama is running (ollama serve) and try again."
            );
        } else {
            showAnswerError(data.error || "An error occurred. Please try again.");
        }
    } catch (err) {
        showAnswerError("Could not reach the server. Please check your connection and try again.");
        console.error("Ask error:", err);
    } finally {
        answerLoading.hidden = true;
        setLoading(askBtn, false);
        askBtn.disabled = false;
    }
}

/**
 * Render a successful answer with optional source references.
 * @param {string}  answer  - The LLM's answer text.
 * @param {Array}   sources - Array of {filename, page} objects.
 */
function showAnswer(answer, sources) {
    answerText.textContent = answer;

    if (sources.length > 0) {
        sourcesList.innerHTML = "";
        sources.forEach(({ filename, page }) => {
            const li = document.createElement("li");
            li.innerHTML = `
                <span class="source-tag">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    ${escapeHtml(filename)} · p.${escapeHtml(String(page))}
                </span>`;
            sourcesList.appendChild(li);
        });
        sourcesBlock.hidden = false;
    } else {
        sourcesBlock.hidden = true;
    }

    answerResult.hidden = false;
    answerError.hidden  = true;
}

/**
 * Render an error message in the answer area.
 * @param {string} message - User-facing error text.
 */
function showAnswerError(message) {
    answerError.textContent = message;
    answerError.hidden      = false;
    answerResult.hidden     = true;
    answerContainer.hidden  = false;
    answerLoading.hidden    = true;
}
