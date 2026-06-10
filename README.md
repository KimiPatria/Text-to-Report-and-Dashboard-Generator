# AutoDashboard

A natural-language AI interface for a PostgreSQL plantation management (EPMS) database. Ask questions in plain English across three browser-based interfaces — the system generates SQL, executes it, and presents results as dashboards, reports, or conversational answers.

Built as an internship project to explore practical LLM application development: retrieval-augmented generation, multi-step LLM pipelines, and cost-efficient inference on free-tier APIs.

---

## Three Main Features

### 1. Dashboard
Ask a goal like *"show me revenue by division"* and the app generates a full dashboard layout — 3 to 12 widgets (KPI cards, bar charts, line charts, tables) — each backed by its own SQL query. The layout is structured JSON produced by the LLM, then executed and returned to the browser as a live dashboard.

### 2. Report
Two modes:
- **Preset reports** — fixed SQL templates for common summaries (general, employee, harvest). Zero LLM cost.
- **Generated reports** — ask a freeform question; the full RAG → SQL pipeline runs, potentially generating multiple queries and composing the results into a multi-section report. Supports saved query templates with date parameterization for repeatable business queries.

### 3. Chat
Conversational Q&A. Ask a question in plain English, get a natural-language answer. The pipeline retrieves relevant tables and few-shot examples, generates SQL, executes it with one self-correction retry on error, then uses an LLM to compose an answer adapted to the shape of the result (count, comparison, trend, list, etc.). Returns the SQL, row count, tables used, and the answer.

---

## How It Works

All three features share the same underlying pipeline:

```
User question
     │
     ▼
1. Retrieval (no LLM)       — ChromaDB + BM25 hybrid search finds relevant tables
     │
     ▼
2. SQL Generation (LLM)     — Prompt with schema context, business rules → SQL query
     │
     ▼
3. Execute + Self-Correct   — Run query; on error, feed error back to LLM for one retry
     │
     ▼
4. Presentation (LLM)       — Dashboard layout / report sections / conversational answer
```

**Business grounding:** Every prompt is augmented with `business_rules.txt` and `glossary.yaml` so the LLM understands domain-specific terms, table preferences, and query constraints.

**SQL safety:** All queries are validated before execution — DDL, DML, and multi-statement queries are blocked. The database connection is read-only with a configurable statement timeout.

---

## Features

- **Hybrid table retrieval** — combines dense vector search (ChromaDB + fastembed) with BM25 keyword matching for robust schema lookup
- **Self-correction loop** — on SQL execution error, the error message is fed back to the LLM for one corrected retry
- **SQL safety validation** — blocks DDL, DML, and multi-statement queries; read-only DB connection with configurable timeout
- **Few-shot examples from history** — successful Chat queries are stored and retrieved as in-context examples for future similar questions
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
| Embeddings | fastembed (BAAI/bge-small-en-v1.5) |
| Keyword search | BM25 (rank-bm25) |
| Database | PostgreSQL via SQLAlchemy |
| Frontend | Vanilla HTML/CSS/JS (three static files) |

---

## Project Structure

```
├── dashboard_server.py     # Main entry point (port 8001) — mounts all three apps
├── chat_router.py          # Chat endpoint: RAG → SQL → conversational answer
├── report_router.py        # Report endpoints: preset and generated reports
├── main.py                 # Original standalone text-to-SQL server (port 8000)
├── llm.py                  # LLM provider routing (Groq, GPT OSS), token logging
├── retrieval.py            # Hybrid ChromaDB + BM25 retrieval
├── prompt_builder.py       # Prompt assembly and SQL extraction
├── metadata_loader.py      # Schema catalog loader (schema_metadata.json)
├── sql_validator.py        # SQL safety checks
├── query_history.py        # Few-shot example storage
├── layout_prompt.py        # Dashboard layout prompt builder
├── layout_schema.py        # Dashboard widget JSON schema
├── error_handler.py        # Error classification
├── query_normalizer.py     # Query standardization
├── config.py               # Env-var config and SQLAlchemy engine
├── bootstrap_metadata.py   # One-time Gemini-powered schema annotation script
├── dashboard_static/
│   └── index.html          # Dashboard UI
├── report_static/
│   └── report.html         # Report UI
├── chat_static/
│   └── chat.html           # Chat UI
├── business_rules.txt      # Domain rules and query constraints for the LLM
├── glossary.yaml           # Domain term → table/column mappings
├── schema_metadata.json    # Table descriptions and embeddings
└── requirements.txt
```

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/KimiPatria/Text-to-SQL-Chatbot.git
cd AutoDashboard
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://username:password@host:5432/dbname
DB_SCHEMA=public
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Set up domain context files

Edit `business_rules.txt` to describe query constraints, and `glossary.yaml` to map domain terms to table/column names. The LLM uses both as grounding context in every prompt.

### 4. Bootstrap schema metadata

Run this once to generate `schema_metadata.json` — natural-language descriptions for every table in your database:

```bash
# Requires GEMINI_API_KEY in .env
python bootstrap_metadata.py
```

### 5. Run

```bash
uvicorn dashboard_server:app --reload --port 8001
```

Then open the three interfaces:

| Interface | URL |
|---|---|
| Dashboard | [http://localhost:8001](http://localhost:8001) |
| Report | [http://localhost:8001/report](http://localhost:8001/report) |
| Chat | [http://localhost:8001/chat](http://localhost:8001/chat) |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Dashboard UI |
| `POST` | `/api/generate` | Generate a dashboard layout |
| `GET` | `/report` | Report UI |
| `POST` | `/report/generate` | Generate a freeform report |
| `GET` | `/chat` | Chat UI |
| `POST` | `/chat/message` | Send a chat message |
| `GET` | `/health` | Health check |
| `POST` | `/admin/reload-schema` | Hot-reload schema metadata |
| `POST` | `/admin/reload-glossary` | Hot-reload business rules |

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `DATABASE_URL` | — | PostgreSQL connection string (required) |
| `DB_SCHEMA` | `public` | PostgreSQL schema to introspect |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Main SQL-generation model |
| `MAX_RESULT_ROWS` | `100` | Row cap injected into every query |
| `STATEMENT_TIMEOUT_MS` | `15000` | PostgreSQL statement timeout (ms) |

---

## License

MIT
