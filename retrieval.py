import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

import metadata_loader
from metadata_loader import TableCard, embedding_text, load_all_cards
from query_history import get_successful_pairs
from llm import call_routing_llm
from error_handler import is_off_domain as _is_off_domain

log = logging.getLogger(__name__)

_EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
_CHROMA_PATH   = "./chroma_store"
_TABLE_COLL    = "tables"
_HISTORY_COLL  = "query_history"

# Module-level singletons — set by initialize()
_embedder:          Any = None
_chroma_client:     Any = None
_table_collection:  Any = None
_history_collection: Any = None
_bm25_index:        Any = None   # rank_bm25.BM25Okapi
_bm25_table_names:  list[str] = []


# ── tokenizer ──────────────────────────────────────────────────────────────

# Table prefix letters and common noise tokens that appear in every document
_STOPWORDS = {"t", "m", "tr", "mc", "id", "by", "the", "a", "of", "is", "in", "and", "or"}

# Tables whose names start with these prefixes are approval/audit logs — penalise them
_AUDIT_PREFIXES = ("log_", "audit_", "login_")


def _tokenize(text: str) -> list[str]:
    """Split text for BM25: handles underscores, hyphens, camelCase, and stopwords."""
    # Split camelCase boundaries
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Replace underscores, hyphens, dots with spaces
    text = re.sub(r'[_\-\.]', ' ', text)
    tokens = text.lower().split()
    return [tok for tok in tokens if tok not in _STOPWORDS and len(tok) > 1]


# ── embedding ──────────────────────────────────────────────────────────────

def _embed(texts: list[str]) -> list[list[float]]:
    global _embedder
    if _embedder is None:
        try:
            from fastembed import TextEmbedding
            _embedder = TextEmbedding(model_name=_EMBED_MODEL)
            log.info("fastembed model '%s' loaded", _EMBED_MODEL)
        except Exception as e:
            raise RuntimeError(
                f"fastembed model download/init failed: {e}. "
                "Check network access and that 'fastembed' is installed."
            ) from e
    embeddings = list(_embedder.embed(texts))
    return [e.tolist() for e in embeddings]


# ── ChromaDB ───────────────────────────────────────────────────────────────

def _init_chroma() -> Any:
    global _chroma_client
    import chromadb

    def _make_client() -> Any:
        return chromadb.PersistentClient(path=_CHROMA_PATH)

    try:
        client = _make_client()
        client.list_collections()  # smoke-test
        _chroma_client = client
        return client
    except Exception as e:
        log.warning("ChromaDB init failed (%s) — resetting chroma_store/ and retrying.", e)
        p = Path(_CHROMA_PATH)
        if p.exists():
            shutil.rmtree(p)
        client = _make_client()
        _chroma_client = client
        return client


# ── public initializer ─────────────────────────────────────────────────────

def initialize(cards: dict[str, TableCard] | None = None) -> None:
    """Build dense + BM25 indexes. Called once at server startup."""
    global _bm25_index, _bm25_table_names, _table_collection

    from rank_bm25 import BM25Okapi

    if cards is None:
        cards = metadata_loader._CARDS_CACHE if metadata_loader._CARDS_CACHE else load_all_cards()

    log.info("Initializing retrieval indexes for %d tables...", len(cards))
    t0 = time.monotonic()

    client = _init_chroma()

    # ── table dense index ────────────────────────────────────────────────
    table_names = sorted(cards.keys())
    texts = [embedding_text(cards[name]) for name in table_names]

    embeddings = _embed(texts)   # raises clearly on network failure

    _table_collection = client.get_or_create_collection(
        name=_TABLE_COLL,
        metadata={"hnsw:space": "cosine"},
    )
    _table_collection.upsert(ids=table_names, embeddings=embeddings, documents=texts)

    # ── BM25 index ───────────────────────────────────────────────────────
    _bm25_index       = BM25Okapi([_tokenize(t) for t in texts])
    _bm25_table_names = table_names

    # ── query-history dense index ────────────────────────────────────────
    _refresh_history_collection(client)

    elapsed = time.monotonic() - t0
    log.info(
        "Loaded %d table cards, built dense+BM25 indexes in %.2fs",
        len(cards), elapsed,
    )


# ── query history index ────────────────────────────────────────────────────

def _refresh_history_collection(client: Any | None = None) -> None:
    global _history_collection

    c = client or _chroma_client
    if c is None:
        return

    _history_collection = c.get_or_create_collection(
        name=_HISTORY_COLL,
        metadata={"hnsw:space": "cosine"},
    )

    pairs = get_successful_pairs(limit=500)
    if not pairs:
        return

    ids        = [str(row_id) for row_id, _, _  in pairs]
    questions  = [q           for _, q, _        in pairs]
    sqls       = [sql         for _, _, sql       in pairs]

    try:
        embeddings = _embed(questions)
    except Exception as e:
        log.warning("Failed to embed query history: %s", e)
        return

    _history_collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=questions,
        metadatas=[{"sql": sql, "success": True} for sql in sqls],
    )
    log.info("Indexed %d query history entries", len(ids))


def refresh_query_history_index() -> None:
    """Incrementally upsert the newest successful query. Called by main.py after record_query()."""
    if _history_collection is None or _chroma_client is None:
        return

    pairs = get_successful_pairs(limit=1)
    if not pairs:
        return

    row_id, question, sql = pairs[0]
    str_id = str(row_id)

    try:
        existing = _history_collection.get(ids=[str_id])
        if existing and existing.get("ids"):
            return
    except Exception:
        pass

    try:
        embedding = _embed([question])
        _history_collection.upsert(
            ids=[str_id],
            embeddings=embedding,
            documents=[question],
            metadatas=[{"sql": sql, "success": True}],
        )
        log.debug("Indexed new query history entry id=%s", str_id)
    except Exception as e:
        log.warning("Failed to index query history entry id=%s: %s", str_id, e)


# ── public retrieval API ───────────────────────────────────────────────────

def retrieve_tables(question: str, k: int = 5) -> tuple[list[TableCard], bool]:
    """Hybrid BM25 + dense retrieval with reciprocal rank fusion + 1-hop FK closure.

    Returns (selected_tables, used_llm_fallback). The fallback flag is True only when
    primary retrieval found 0 tables and the routing LLM was invoked to recover.
    """
    cards = metadata_loader._CARDS_CACHE

    if _table_collection is None or _bm25_index is None:
        log.warning("Indexes not ready — using BM25-only fallback (top-10)")
        return _bm25_fallback(question, 10, cards), False

    n = len(_bm25_table_names)

    # Dense
    min_dense_dist = 1.0
    try:
        q_emb        = _embed([question])[0]
        dense_res    = _table_collection.query(query_embeddings=[q_emb], n_results=min(20, n))
        dense_ids    = dense_res["ids"][0] if dense_res["ids"] else []
        dense_dists  = dense_res.get("distances", [[]])[0] if dense_res else []
        min_dense_dist = min(dense_dists) if dense_dists else 1.0
    except Exception as e:
        log.warning("Dense retrieval failed: %s", e)
        dense_ids = []

    # BM25
    scores = _bm25_index.get_scores(_tokenize(question))
    scores = [
        s * 0.3 if any(_bm25_table_names[i].startswith(p) for p in _AUDIT_PREFIXES) else s
        for i, s in enumerate(scores)
    ]
    max_bm25_score = max(scores) if n > 0 else 0.0
    bm25_ranked    = sorted(range(n), key=lambda i: scores[i], reverse=True)[:20]
    bm25_ids       = [_bm25_table_names[i] for i in bm25_ranked]

    log.info("[retrieval] dense: %s", ", ".join(dense_ids[:8]))
    log.info("[retrieval] bm25:  %s", ", ".join(bm25_ids[:8]))
    log.info("[retrieval] confidence: max_bm25=%.4f  min_dense_dist=%.4f", max_bm25_score, min_dense_dist)

    # Reciprocal rank fusion
    fused: dict[str, float] = {}
    for rank, tid in enumerate(dense_ids):
        fused[tid] = fused.get(tid, 0.0) + 1.0 / (60 + rank + 1)
    for rank, tid in enumerate(bm25_ids):
        fused[tid] = fused.get(tid, 0.0) + 1.0 / (60 + rank + 1)

    top_k = sorted(fused, key=lambda t: fused[t], reverse=True)[:k]

    # BM25 anchor: when BM25 is highly confident (strong token match, e.g. Indonesian synonym
    # hit on m_destination "tujuan"), guarantee its top result is in the final set even if
    # the English-only dense model disagrees completely.
    if max_bm25_score >= 5.0:
        bm25_winner = _bm25_table_names[bm25_ranked[0]]
        if bm25_winner not in top_k:
            top_k = [bm25_winner] + top_k[: k - 1]

    # 1-hop FK closure, capped at +3
    top_set = set(top_k)
    extra: list[str] = []
    for name in top_k:
        card = cards.get(name)
        if not card:
            continue
        for col in card.columns:
            if col.fk_ref:
                ref_table, _ = col.fk_ref
                if ref_table not in top_set and ref_table not in extra:
                    extra.append(ref_table)
                    if len(extra) >= 3:
                        break
        if len(extra) >= 3:
            break

    top_k = top_k + extra[:3]
    log.info("[retrieval] fused top-%d: %s", k, ", ".join(top_k))

    selected_tables = [cards[name] for name in top_k if name in cards]

    # Off-domain check: fast keyword match for known plantation terms; LLM call only for edge cases.
    _off_domain = _is_off_domain(question)

    if len(selected_tables) == 0 or _off_domain:
        log.warning(
            "LLM fallback triggered: off_domain=%s max_bm25=%.4f min_dist=%.4f query=%r",
            _off_domain, max_bm25_score, min_dense_dist, question[:80],
        )
        table_list = "\n".join(
            f"{name}: {card.description}"
            for name, card in sorted(metadata_loader._CARDS_CACHE.items())
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a database table selector for a palm oil plantation management system.\n"
                    "Given the user query below, return a JSON object with key \"tables\" containing "
                    "an array of the most relevant table names (maximum 5).\n"
                    "Only include tables that could plausibly contain data needed to answer the query.\n"
                    "If no tables are relevant, return {\"tables\": []}.\n"
                    "Respond ONLY with the JSON object, no explanation.\n\n"
                    f"Available tables:\n{table_list}"
                ),
            },
            {
                "role": "user",
                "content": f"User query: {question}",
            },
        ]
        llm_tables = call_routing_llm(messages)
        valid_names = [name for name in llm_tables if name in metadata_loader._CARDS_CACHE][:5]
        selected_tables = [metadata_loader._CARDS_CACHE[name] for name in valid_names]
        return selected_tables, True

    return selected_tables, False


def _bm25_fallback(question: str, k: int, cards: dict[str, TableCard]) -> list[TableCard]:
    if not _bm25_index:
        return []
    scores = _bm25_index.get_scores(_tokenize(question))
    scores = [
        s * 0.3 if any(_bm25_table_names[i].startswith(p) for p in _AUDIT_PREFIXES) else s
        for i, s in enumerate(scores)
    ]
    ranked  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [cards[_bm25_table_names[i]] for i in ranked if _bm25_table_names[i] in cards]


def retrieve_examples(question: str, k: int = 3) -> list[tuple[str, str]]:
    """Return (past_question, past_sql) from query history, skipping silently if < 3 exist."""
    if _history_collection is None:
        return []

    try:
        total = _history_collection.count()
    except Exception:
        return []

    if total < 3:
        return []

    try:
        q_emb   = _embed([question])[0]
        results = _history_collection.query(
            query_embeddings=[q_emb],
            n_results=min(k, total),
            where={"success": True},
        )
        if not results["ids"] or not results["ids"][0]:
            return []

        past_questions = results["documents"][0]
        metadatas      = results["metadatas"][0]
        pairs = [(pq, m["sql"]) for pq, m in zip(past_questions, metadatas)]
        log.info("[retrieval] examples: %d found", len(pairs))
        return pairs
    except Exception as e:
        log.warning("Example retrieval failed: %s", e)
        return []
