import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MODEL_120B = os.getenv("GROQ_MODEL_120B", "openai/gpt-oss-120b")
GROQ_MODEL_20B = os.getenv("GROQ_MODEL_20B", "openai/gpt-oss-20b")
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "groq").lower()
MAX_RESULT_ROWS = int(os.getenv("MAX_RESULT_ROWS", "5000"))
LARGE_RESULT_THRESHOLD = int(os.getenv("LARGE_RESULT_THRESHOLD", "50"))
STATEMENT_TIMEOUT_MS = int(os.getenv("STATEMENT_TIMEOUT_MS", "15000"))
DB_SCHEMA = os.getenv("DB_SCHEMA", "public")

# --- dynamic routing (iteration 2) ---
ROUTING_MODEL = os.getenv("ROUTING_MODEL", "llama-3.1-8b-instant")
CATALOG_TTL_SECONDS = int(os.getenv("CATALOG_TTL_SECONDS", "1800"))
MAX_ROUTED_TABLES = int(os.getenv("MAX_ROUTED_TABLES", "12"))
GLOSSARY_PATH = os.getenv("GLOSSARY_PATH", "./glossary.yaml")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")

_pg_options = (
    f"-c default_transaction_read_only=on "
    f"-c statement_timeout={STATEMENT_TIMEOUT_MS}"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"options": _pg_options},
)
