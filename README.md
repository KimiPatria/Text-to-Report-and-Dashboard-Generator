# LLM Text-to-SQL

A natural-language interface for PostgreSQL databases. Ask questions in plain English; the system generates SQL, executes it, and explains the results — all through a browser-based chat UI.

Built as an internship project to explore practical LLM application development: retrieval-augmented generation, multi-step LLM pipelines, and cost-efficient inference on free-tier APIs.

---

## How It Works

The pipeline has four stages, each with a distinct responsibility:

```
User question
     │
     ▼
1. Retrieval (no LLM)       — ChromaDB + BM25 hybrid search finds relevant tables
     │
     ▼
2. SQL Generation (LLM)     — Prompt with schema context → SQL query
     │
     ▼
3. Execute + Self-Correct   — Run query; on error, feed error back to LLM for one retry
     │
     ▼
4. Interpretation (LLM)     — Explain results in natural language (skipped for simple results)
```

**Routing (pre-step):** A lightweight LLM (Llama 3.1 8B) first narrows the table candidates from the full catalog down to ≤12 tables, so the main SQL-generation prompt stays within token budget regardless of database size.

---

## Features

- **Hybrid table retrieval** — combines dense vector search (ChromaDB + fastembed) with BM25 keyword matching for robust schema lookup
- **Dynamic routing** — small model pre-filters the table catalog; reduces prompt size and improves SQL accuracy
- **Self-correction loop** — on SQL execution error, the error message is fed back to the LLM for one corrected retry
- **SQL safety validation** — blocks DDL, DML, and multi-statement queries before execution; database connection is read-only with a configurable statement timeout
- **Few-shot examples from history** — successful queries are stored and retrieved as in-context examples for future similar questions
- **Switchable model providers** — toggle between Llama 3.3 70B, GPT OSS 120B, and GPT OSS 20B at runtime via the UI or API
- **Schema bootstrapping** — `bootstrap_metadata.py` uses Gemini to auto-generate table descriptions, synonyms, and example questions from DDL
- **Hot-reload** — schema metadata and business glossary can be reloaded without restarting the server

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI + Uvicorn |
| LLM inference | Groq API (Llama 3.3 70B, Llama 3.1 8B, GPT OSS 120B/20B) |
| Schema bootstrapping | Google Gemini 2.5 Flash |
| Vector store | ChromaDB |
| Embeddings | fastembed |
| Keyword search | BM25 (rank-bm25) |
| Database | PostgreSQL via SQLAlchemy |
| Frontend | Vanilla HTML/JS (single static file) |

---

## Project Structure

```
├── main.py               # FastAPI app and 4-step pipeline
├── llm.py                # Groq client, provider routing, token logging
├── retrieval.py          # Hybrid ChromaDB + BM25 retrieval
├── prompt_builder.py     # Prompt assembly and SQL extraction
├── metadata_loader.py    # Schema catalog loader (schema_metadata.json)
├── sql_validator.py      # SQL safety checks
├── query_history.py      # SQLite store for few-shot examples
├── schema_router.py      # Lightweight LLM routing for table pre-filtering
├── config.py             # Env-var config and SQLAlchemy engine
├── bootstrap_metadata.py # One-time Gemini-powered schema annotation script
├── static/index.html     # Chat UI
├── glossary.example.yaml       # Template: domain terms → table/column mappings
├── business_rules.example.txt  # Template: query rules for the LLM
├── .env.example          # Template: environment variables
└── requirements.txt
```

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/KimiPatria/Text-to-SQL-Chatbot.git
cd llm-database
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://username:password@host:5432/dbname
DB_SCHEMA=public
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Set up domain context files

```bash
cp glossary.example.yaml glossary.yaml
cp business_rules.example.txt business_rules.txt
```

Edit these files to describe your database's domain terms, table preferences, and business rules. The LLM uses them as grounding context in every prompt.

### 4. Bootstrap schema metadata

Run this once to generate `schema_metadata.json` — natural-language descriptions for every table in your database:

```bash
# Requires GEMINI_API_KEY in .env
python bootstrap_metadata.py
```

Or create `schema_metadata.json` manually (see the format in `metadata_loader.py`).

### 5. Run

```bash
uvicorn main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) and start asking questions.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `POST` | `/chat` | Submit a question |
| `GET` | `/get-provider` | Get active LLM provider |
| `POST` | `/set-provider` | Switch LLM provider |
| `GET` | `/health` | Health check |
| `GET` | `/admin/catalog` | Inspect loaded table catalog |
| `POST` | `/admin/reload-schema` | Hot-reload schema metadata |
| `POST` | `/admin/reload-glossary` | Hot-reload business rules |

### `/chat` example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How many records were created last month?"}'
```

```json
{
  "answer": "There were 1,284 records created in April 2025.",
  "sql": "SELECT COUNT(*) FROM t_example WHERE ...",
  "rows": [{"count": 1284}],
  "columns": ["count"],
  "tables_used": ["t_example"],
  "provider": "groq",
  "provider_label": "Llama 3.3 70B"
}
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `DATABASE_URL` | — | PostgreSQL connection string (required) |
| `DB_SCHEMA` | `public` | PostgreSQL schema to introspect |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Main SQL-generation model |
| `ROUTING_MODEL` | `llama-3.1-8b-instant` | Lightweight table-routing model |
| `MAX_RESULT_ROWS` | `100` | Row cap injected into every query |
| `STATEMENT_TIMEOUT_MS` | `15000` | PostgreSQL statement timeout (ms) |
| `MAX_ROUTED_TABLES` | `12` | Max tables passed to SQL-generation prompt |
| `CATALOG_TTL_SECONDS` | `1800` | How long the routing catalog is cached |

---

## License

MIT
