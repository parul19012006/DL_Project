# LegalDocAI GenAI Service

An independent Python microservice that answers questions about legal documents using **Retrieval-Augmented Generation (RAG)**. Every answer is built only from passages retrieved from the caller's own documents, and every citation is checked against those passages.

It is built with FastAPI and is designed to be called over HTTP by a separate backend (for example a MERN stack). It contains no frontend, authentication or database code.

## Features

- **Ingestion** of PDF and DOCX files, with OCR fallback for scanned pages and removal of repeated headers and footers
- **Chunking** by token count with overlap and section awareness
- **Embeddings** with Sentence-Transformers (default `all-MiniLM-L6-v2`)
- **Vector storage** in ChromaDB, with a pluggable backend registry
- **Hybrid retrieval** combining semantic and keyword search, fused with RRF or weighted scoring, then de-duplicated and reranked with a cross-encoder
- **Context construction** with a token budget, per-source limits and redundancy filtering
- **Grounded generation** with citation validation and quote verification
- **Multi-tenant isolation**: every stored-data endpoint requires an explicit `tenant_id`
- **Operations**: one error envelope, a request ID on every response, batch ingestion as background jobs, and liveness/readiness probes

## How it works

```
 file / text ──► ingestion ──► chunking ──► embeddings ──► vector store
                                                              │
 question ──► hybrid retrieval ──► rerank ──► context ──► generation ──► answer + validated citations
              (semantic + keyword)                       (LLM or extractive)
```

## Project structure

```
app/
├── main.py          FastAPI app factory, middleware, router mounting
├── config.py        All settings (environment-driven, validated)
├── exceptions.py    Error types and the shared error envelope
├── logging_config.py
├── api/             HTTP routers (pipeline, health, documents, index, retrieval, context, generation)
├── ingestion/       PDF/DOCX parsing, OCR, cleaning, section detection, validation
├── chunking/        Chunker, splitters, tokenizer
├── embeddings/      Sentence-Transformers and deterministic encoders, factory
├── vectorstore/     ChromaDB and in-memory backends, factory
├── retrieval/       Semantic, keyword, fusion, dedup, rerank, pipeline
├── context/         Source selection, ordering, token budget, formatting
├── generation/      Providers (extractive, OpenAI-compatible, Anthropic), prompts, citation checks
├── services/        Indexing and batch job orchestration
└── models/          Pydantic schemas and domain models
```

## Getting started

### Requirements

- Python 3.11 (the version used by the repo's dev container)
- Tesseract OCR, only if you need to read scanned PDFs

### Install

The repository does not ship a `requirements.txt` for this service. The imports in `app/` need these packages:

```bash
pip install fastapi uvicorn python-multipart pydantic pydantic-settings httpx \
            chromadb sentence-transformers pymupdf python-docx
# Optional
pip install pytesseract pillow   # OCR for scanned PDFs
pip install tiktoken             # exact token counting (TOKENIZER=tiktoken)
```

### Run

From the repository root:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive API docs are at `http://localhost:8000/docs` (disable with `ENABLE_DOCS=false`).

### Configure

Settings are read from environment variables or a `.env` file (path overridable with `ENV_FILE`). Names are case-insensitive.

| Variable | Default | Purpose |
|---|---|---|
| `ENVIRONMENT` | `development` | `development`, `staging` or `production` |
| `PORT` | `8000` | Listen port |
| `SERVICE_TOKEN` | *(empty)* | Shared secret sent as `X-Service-Token`. Empty disables auth (development only) |
| `DOCUMENT_ROOT` | `./data/documents` | The only directory documents may be read from |
| `MAX_FILE_SIZE_MB` | `50` | Upload size limit |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | Chunking in tokens |
| `EMBEDDING_PROVIDER` | `sentence_transformers` | Or `deterministic` for dev and CI |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Changing it requires re-indexing |
| `VECTOR_BACKEND` | `chroma` | Or `memory` (tests only) |
| `VECTOR_PERSIST_DIR` | `./data/vectors` | ChromaDB storage |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid`, `semantic` or `keyword` |
| `RERANKER_PROVIDER` | `cross_encoder` | Or `lexical` |
| `LLM_PROVIDER` | `extractive` | `extractive`, `openai` or `anthropic` |
| `LLM_MODEL` | *(provider default)* | Model name |
| `LLM_BASE_URL` | *(none)* | Custom endpoint, e.g. Azure, Groq, vLLM or Ollama via the OpenAI format |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | *(empty)* | Credentials, read from the environment only |
| `LLM_STRICT` / `EMBEDDING_STRICT` | `false` | Fail at startup instead of falling back (recommended in production) |

Example `.env` for an Anthropic-backed setup:

```env
ENVIRONMENT=production
SERVICE_TOKEN=change-me
LLM_PROVIDER=anthropic
LLM_MODEL=<model-name>
ANTHROPIC_API_KEY=<your-key>
LLM_STRICT=true
EMBEDDING_STRICT=true
ENABLE_DOCS=false
```

> **Note on the default provider.** With `LLM_PROVIDER=extractive` the service selects sentences from retrieved passages and uses no language model. It works offline and needs no API key, but it cannot reason or synthesise, and responses say so.

## API

All endpoints except `/health*` and `/` require the `X-Service-Token` header when `SERVICE_TOKEN` is set.

### Integration endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Ingest one document from `file_path` (under `DOCUMENT_ROOT`) or inline `text` |
| `POST` | `/ingest/batch` | Ingest many documents as a background job (HTTP 202) |
| `GET` | `/ingest/batch/{job_id}` | Poll a batch job's progress |
| `POST` | `/query` | Ask a question and get an answer with citations |
| `DELETE` | `/documents/{document_id}` | Remove a document's vector data |
| `GET` | `/health` | Health, plus `/health/live` and `/health/ready` probes |

Re-ingesting the same `document_id` replaces its chunks, so retries are safe.

### Example

Ingest a document:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -H "X-Service-Token: change-me" \
  -d '{
        "document_id": "doc-1",
        "tenant_id": "tenant-a",
        "file_path": "tenant-a/vendor_contract_2024.pdf",
        "document_type": "contract"
      }'
```

Ask a question:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-Service-Token: change-me" \
  -d '{
        "query": "How much notice is required to terminate for convenience?",
        "tenant_id": "tenant-a",
        "top_k": 6,
        "include_sources": true
      }'
```

`/query` also accepts optional filters (`document_ids`, `document_type`, `section`, `page_range`, `metadata`) and tuning (`temperature`, `include_context`).

### Debugging endpoints

These expose individual pipeline stages and are not part of the integration contract: `/documents/extract`, `/documents/chunk`, `/documents/formats`, `/index`, `/search`, `/retrieve`, `/context`, `/answer`, `/index/stats`.

### Errors

Every non-2xx response uses one envelope, so clients can branch on a stable `error.type` instead of parsing messages:

```json
{ "error": { "type": "document_not_found", "message": "...", "details": {} }, "request_id": "..." }
```

Send an `X-Request-ID` header to correlate logs; if you don't, one is generated and returned in the response header and body.

## Design notes

- **Tenant isolation is required, never inferred.** There is no default tenant and no way to search across all of them.
- **Deterministic by default.** Temperature is `0.0`, so identical questions give identical answers.
- **No silent degradation in production.** The `*_STRICT` flags make a broken model or provider fatal rather than quietly serving poor output.
- **Secrets stay in the environment.** API keys are never accepted from requests, logged or returned.

## Known gaps

- No dependency file or `.env.example` is included in the repository (the code refers to one).
- No automated tests are included under `app/`.
