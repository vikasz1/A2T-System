"""
OptumRx Billing Agent — True A2T Implementation
=================================================
Architecture:
  • The LLM is given a TOOL REGISTRY and decides autonomously which tools to call.
  • Python is only an executor — it never hard-codes routing logic.
  • The agentic loop runs until the LLM emits `final_answer`.

A2T Tool Registry:
  1. recall_memory(query)        — retrieve relevant past conversation turns
  2. discover_schema()           — inspect the SQLite database structure
  3. query_database(sql)         — execute a SQL query
  4. search_documents(query)     — semantic search over indexed PDFs
  5. final_answer(answer, suggestions) — terminate the loop & respond

Install:
    pip install streamlit chromadb sentence-transformers pypdf requests pandas
    pip install "sentence-transformers[cross-encoder]"
Run:
    ollama pull llama3.1:8b && ollama serve
    streamlit run pdf_chat.py
"""

import io, json, re, sqlite3, time, hashlib, uuid
from datetime import datetime
from pathlib import Path

import chromadb, requests, streamlit as st, pandas as pd
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="OptumRx Billing Agent", page_icon="💊", layout="wide")

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
DB_PATH      = "healthcare.db"
CHROMA_DIR   = "./chroma_store"
DOC_COL      = "optumrx_docs"
MEM_COL      = "optumrx_memory"
EMBED_MODEL  = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SENT_CHUNK_SIZE = 5
SENT_OVERLAP    = 2
RETRIEVE_K      = 12
TOP_K_FINAL     = 4
MEMORY_TURNS    = 4
MAX_AGENT_STEPS  = 8     # safety cap on tool calls per turn
TOOL_MAX_TOKENS  = 256   # small — tool calls are JSON only, not prose
ANSWER_MAX_TOKENS = 900  # large — separate answer synthesis call
MAX_SAMPLE_ROWS = 1
MAX_OTHER_TABLES = 3
MAX_EXPLAIN_ROWS = 5

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
SESSION_ID = st.session_state.session_id

# ══════════════════════════════════════════════════════════════════════════════
# Cached resources
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource(show_spinner="Loading reranker…")
def get_reranker():
    try:
        return CrossEncoder(RERANK_MODEL)
    except Exception:
        return None

@st.cache_resource(show_spinner="Connecting to ChromaDB…")
def get_chroma_client():
    return chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )

def get_doc_collection(fresh=False):
    client = get_chroma_client()
    if fresh:
        try: client.delete_collection(DOC_COL)
        except Exception: pass
    return client.get_or_create_collection(DOC_COL, metadata={"hnsw:space": "cosine"})

def get_mem_collection():
    return get_chroma_client().get_or_create_collection(MEM_COL, metadata={"hnsw:space": "cosine"})

# ══════════════════════════════════════════════════════════════════════════════
# PDF ingestion — sentence-aware chunking
# ══════════════════════════════════════════════════════════════════════════════
SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

def split_sentences(text):
    sents, result = SENT_RE.split(text.strip()), []
    for s in sents:
        lines = [l.strip() for l in s.split('\n') if l.strip()]
        result.extend(lines if len(lines) > 1 else [s])
    return [s for s in result if len(s) > 20]

def sentence_chunks(text, size=SENT_CHUNK_SIZE, overlap=SENT_OVERLAP):
    sents = split_sentences(text)
    if not sents:
        return [text.strip()] if text.strip() else []
    chunks, i = [], 0
    while i < len(sents):
        chunks.append(" ".join(sents[i:i+size]))
        i += size - overlap
    return [c for c in chunks if c.strip()]

def extract_pages(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return [{"page_num": i+1, "text": (p.extract_text() or "").strip()}
            for i, p in enumerate(reader.pages) if (p.extract_text() or "").strip()]

def ingest_pdf(pdf_bytes, filename, doc_col):
    embedder    = get_embedder()
    pages       = extract_pages(pdf_bytes)
    ts          = datetime.utcnow().isoformat()
    ids, docs, embeds, metas = [], [], [], []
    for page in pages:
        for ci, chunk in enumerate(sentence_chunks(page["text"])):
            uid = hashlib.md5(f"{filename}|{page['page_num']}|{ci}".encode()).hexdigest()
            ids.append(uid)
            docs.append(chunk)
            embeds.append(embedder.encode(chunk).tolist())
            metas.append({"filename": filename, "page_num": page["page_num"],
                           "chunk_index": ci, "total_pages": len(pages), "uploaded_at": ts})
    for i in range(0, len(ids), 100):
        doc_col.upsert(ids=ids[i:i+100], documents=docs[i:i+100],
                       embeddings=embeds[i:i+100], metadatas=metas[i:i+100])
    return {"filename": filename, "pages": len(pages), "chunks": len(ids)}

def collection_stats(col):
    count = col.count()
    sources = set()
    if count > 0:
        for m in col.get(limit=min(count, 1000), include=["metadatas"])["metadatas"]:
            sources.add(m.get("filename", "unknown"))
    return {"count": count, "sources": sorted(sources)}

# ══════════════════════════════════════════════════════════════════════════════
# A2T TOOL IMPLEMENTATIONS
# Each function is a pure tool — no LLM calls inside, just data operations.
# ══════════════════════════════════════════════════════════════════════════════

def tool_recall_memory(query: str) -> dict:
    """Retrieve the most relevant past conversation turns from ChromaDB memory."""
    mem_col  = get_mem_collection()
    if mem_col.count() == 0:
        return {"turns": [], "message": "No memory found for this session."}
    embedder = get_embedder()
    q_embed  = embedder.encode(query).tolist()
    n        = min(MEMORY_TURNS * 2, mem_col.count())
    try:
        results = mem_col.query(
            query_embeddings=[q_embed], n_results=n,
            include=["documents", "metadatas"],
            where={"session_id": SESSION_ID},
        )
        turns = sorted(
            [{"role": m.get("role","user"), "content": d, "timestamp": m.get("timestamp","")}
             for d, m in zip(results["documents"][0], results["metadatas"][0])],
            key=lambda x: x["timestamp"]
        )[-MEMORY_TURNS:]
        return {"turns": turns}
    except Exception as e:
        return {"turns": [], "error": str(e)}


# Module-level cache so the SQL validator can access real column names
_schema_cache: dict = {}   # {table_name: [col_name, ...]}

def tool_discover_schema() -> dict:
    """Inspect all tables in the SQLite database and return schema + sample rows."""
    global _schema_cache
    if not Path(DB_PATH).exists():
        return {"error": f"Database {DB_PATH} not found.", "schema_text": ""}
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    all_tables = [r["name"] for r in cur.fetchall()]
    priority   = [t for t in all_tables if t == "billing"]
    others     = [t for t in all_tables if t != "billing"][:MAX_OTHER_TABLES]
    schema     = {}
    for table in priority + others:
        cur.execute(f"PRAGMA table_info({table})")
        cols = [{"name": r["name"], "type": r["type"],
                 "notnull": bool(r["notnull"]), "pk": bool(r["pk"])} for r in cur.fetchall()]
        cur.execute(f"SELECT COUNT(*) AS cnt FROM {table}")
        rc = cur.fetchone()["cnt"]
        sample = []
        if table == "billing":
            cur.execute(f"SELECT * FROM {table} LIMIT {MAX_SAMPLE_ROWS}")
            sample = [dict(r) for r in cur.fetchall()]
        schema[table] = {"columns": cols, "sample_rows": sample, "row_count": rc}
        # Populate cache: lowercase lookup → real name for validation
        _schema_cache[table] = [c["name"] for c in cols]
    conn.close()
    # Compact text for the LLM — show exact names, backtick-quoted
    lines = []
    for table, info in schema.items():
        parts = []
        for c in info["columns"]:
            flags = ("PK " if c["pk"] else "") + ("NN" if c["notnull"] else "")
            parts.append(f"`{c['name']}` {c['type']}{(' ' + flags.strip()) if flags.strip() else ''}")
        lines.append(f"{table}({', '.join(parts)}) — {info['row_count']} rows")
        if info["sample_rows"]:
            lines.append(f"  e.g. {info['sample_rows'][0]}")
    # Append a CRITICAL reminder so the LLM reads the exact names
    reminder = (
        "\nCRITICAL: Use ONLY the EXACT column names shown above (case-sensitive, backtick-quoted). "
        "Do NOT invent names like patient_name, billing_amount, etc. "
        "The real names are listed above — copy them exactly."
    )
    schema_text = "\n".join(lines) + reminder
    return {"schema_text": schema_text, "tables": list(schema.keys())}


def _validate_sql_columns(sql: str) -> str | None:
    """
    Check SQL against _schema_cache for obviously wrong column names.
    Returns an error string if bad names are found, else None.
    Strip backticks/quotes for comparison; check against real column names
    (case-insensitive to be lenient, but report the real casing).
    """
    if not _schema_cache:
        return None  # schema not loaded yet — let the DB error naturally

    import re as _re
    # Collect all backtick-quoted or bare identifiers in the SQL
    candidates = _re.findall(r"`([^`]+)`|\b([A-Za-z_][A-Za-z0-9_ ]*)\b", sql)
    tokens = [a or b for a, b in candidates]

    sql_keywords = {
        "select","from","where","and","or","not","in","like","order","by","group",
        "limit","offset","join","on","as","having","distinct","count","sum","avg",
        "min","max","is","null","between","case","when","then","else","end","inner",
        "left","right","outer","true","false","asc","desc","insert","update","delete"
    }

    bad = []
    for table, real_cols in _schema_cache.items():
        real_lower = {c.lower(): c for c in real_cols}
        for tok in tokens:
            tok_lower = tok.lower().strip()
            if tok_lower in sql_keywords or tok_lower == table.lower():
                continue
            if len(tok_lower) < 2:
                continue
            # Flag if the token looks like a column reference but isn't in the schema
            if tok_lower not in real_lower and "_" in tok_lower:
                # snake_case token that doesn't exist → likely hallucinated
                bad.append(tok)

    if bad:
        all_cols = []
        for t, cols in _schema_cache.items():
            all_cols.append(f"{t}: " + ", ".join(f"`{c}`" for c in cols))
        return (
            f"SQL contains column names that do not exist in the database: {bad}. "
            f"Use ONLY these exact names (backtick-quoted): {'; '.join(all_cols)}. "
            f"Rewrite the SQL with the correct column names."
        )
    return None


def tool_query_database(sql: str) -> dict:
    """Execute a SQL SELECT query on the healthcare database."""
    if not Path(DB_PATH).exists():
        return {"success": False, "error": f"{DB_PATH} not found.", "data": []}

    # Pre-flight column validation
    col_error = _validate_sql_columns(sql)
    if col_error:
        return {"success": False, "error": col_error, "data": []}

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur  = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return {"success": True, "message": "Query succeeded but returned no rows.",
                    "data": [], "row_count": 0}
        return {"success": True, "data": [dict(r) for r in rows], "row_count": len(rows)}
    except Exception as e:
        # Also include real column names in DB errors to help the LLM fix itself
        hint = ""
        if _schema_cache:
            hint = " Real columns: " + "; ".join(
                f"{t}({', '.join(f'`{c}`' for c in cols)})"
                for t, cols in _schema_cache.items()
            )
        return {"success": False, "error": str(e) + hint, "data": []}


def tool_search_documents(query: str) -> dict:
    """
    Semantic search over indexed PDFs.
    Uses multi-query expansion + cross-encoder reranking for best accuracy.
    """
    doc_col  = get_doc_collection()
    if doc_col.count() == 0:
        return {"chunks": [], "message": "No documents indexed yet. Please upload PDFs first."}

    embedder = get_embedder()
    reranker = get_reranker()

    # Multi-query expansion via LLM
    expand_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"Output ONLY a JSON array of 2 alternative search queries. No markdown.\n"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Original: {query}\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n["
    )
    queries = [query]
    try:
        raw = call_llm(expand_prompt, max_tokens=80).strip()
        raw = ("[" + raw) if not raw.startswith("[") else raw
        raw = raw if raw.endswith("]") else raw.rstrip('", ') + '"]'
        queries += [str(a) for a in json.loads(raw)[:2]]
    except Exception:
        pass

    # Retrieve candidates from all query variants
    seen, candidates = set(), []
    for q in queries:
        q_emb = embedder.encode(q).tolist()
        n     = min(RETRIEVE_K, doc_col.count())
        res   = doc_col.query(query_embeddings=[q_emb], n_results=n,
                              include=["documents", "metadatas", "distances"])
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            uid = f"{meta.get('filename')}|{meta.get('page_num')}|{meta.get('chunk_index')}"
            if uid not in seen:
                seen.add(uid)
                candidates.append({
                    "text":        doc,
                    "filename":    meta.get("filename", "unknown"),
                    "page_num":    meta.get("page_num", "?"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "score":       round(1 - dist, 4),
                })

    # Cross-encoder rerank
    if reranker and len(candidates) > TOP_K_FINAL:
        scores = reranker.predict([(query, c["text"]) for c in candidates])
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    else:
        candidates.sort(key=lambda x: x["score"], reverse=True)

    return {"chunks": candidates[:TOP_K_FINAL], "total_candidates": len(candidates)}


# ── Tool registry — what the LLM sees ─────────────────────────────────────────
TOOL_REGISTRY = {
    "recall_memory": {
        "fn":          tool_recall_memory,
        "description": "Retrieve relevant past conversation turns from memory. Call this FIRST to understand prior context.",
        "parameters":  {"query": "string — the current user question, used to find relevant past turns"},
        "returns":     "List of {role, content} turns",
        "emoji":       "🧠",
    },
    "discover_schema": {
        "fn":          tool_discover_schema,
        "description": "Inspect the healthcare SQLite database — returns table names, column types, row counts, and sample data. Call this BEFORE writing any SQL.",
        "parameters":  {},
        "returns":     "schema_text string and list of table names",
        "emoji":       "🗄️",
    },
    "query_database": {
        "fn":          tool_query_database,
        "description": "Execute a SQL SELECT query on the healthcare database. Always wrap column/table names in backticks. Only call AFTER discover_schema.",
        "parameters":  {"sql": "string — a valid SQLite SELECT statement"},
        "returns":     "{success, data, row_count} or {success:false, error}",
        "emoji":       "⚙️",
    },
    "search_documents": {
        "fn":          tool_search_documents,
        "description": "Semantic search over indexed PDF documents. Returns the most relevant text chunks with source citations (filename, page number).",
        "parameters":  {"query": "string — the search query, rephrase for best retrieval"},
        "returns":     "List of {text, filename, page_num, score} chunks",
        "emoji":       "🔍",
    },
    "final_answer": {
        "fn":          None,   # handled by the loop, not executed as a tool
        "description": "Call this when you have gathered enough information to answer the user. "
                       "This ENDS the agentic loop. Keep the JSON SHORT — the actual answer prose "
                       "will be generated separately. Just signal readiness and list 3 follow-up questions.",
        "parameters":  {
            "ready":       "boolean true — signals you have enough information",
            "suggestions": "array of exactly 3 short follow-up question strings",
        },
        "returns":     "Terminates the agent loop",
        "emoji":       "✅",
    },
}

def tools_description() -> str:
    """Render the tool registry as a concise string for the system prompt."""
    lines = []
    for name, t in TOOL_REGISTRY.items():
        params = ", ".join(f'"{k}": {v}' for k, v in t["parameters"].items()) if t["parameters"] else "none"
        lines.append(f'  Tool: "{name}"\n  Description: {t["description"]}\n  Parameters: {{{params}}}\n  Returns: {t["returns"]}')
    return "\n\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# LLM helpers
# ══════════════════════════════════════════════════════════════════════════════
def call_llm(prompt: str, max_tokens: int = 700) -> str:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
               "options": {"temperature": 0.15, "num_predict": max_tokens}}
    for attempt in range(3):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=180)
            r.raise_for_status()
            return r.json()["response"]
        except requests.exceptions.ConnectionError:
            if attempt < 2: time.sleep(2 ** attempt)
            else: raise RuntimeError("Cannot connect to Ollama — run `ollama serve`")
        except Exception:
            if attempt < 2: time.sleep(2 ** attempt)
            else: raise


def parse_tool_call(raw: str) -> dict:
    """
    Extract the tool call JSON from LLM output.
    Handles markdown fences, leading/trailing text, and partial JSON.
    """
    text = raw.strip()

    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Find first complete {...} object
    if not text.startswith("{"):
        # Try to find { in the text
        idx = text.find("{")
        text = text[idx:] if idx != -1 else "{" + text

    depth, end = 0, 0
    for i, ch in enumerate(text):
        if ch == "{":   depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    try:
        return json.loads(text[:end] if end else text)
    except Exception:
        # If JSON is truncated but clearly a final_answer, treat it as a signal
        if '"tool"' in text and '"final_answer"' in text:
            return {"tool": "final_answer", "parameters": {"ready": True, "suggestions": []},
                    "reasoning": "Detected truncated final_answer — treating as ready signal"}
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# A2T SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════
def build_agent_system_prompt() -> str:
    return f"""You are OptumRx Billing Agent — an intelligent AI assistant for OptumRx pharmacy billing.
You reason step-by-step and use tools to answer user questions accurately.

AVAILABLE TOOLS:
{tools_description()}

STRICT OUTPUT FORMAT:
Every response MUST be a single JSON object — no prose, no markdown outside of JSON.
Format:
  {{"tool": "<tool_name>", "parameters": {{...}}, "reasoning": "<why you chose this tool>"}}

AGENT RULES:
1. Always call `recall_memory` FIRST to understand prior conversation context.
2. For database questions: call `discover_schema` BEFORE `query_database`.
3. For document questions: call `search_documents` with a well-phrased query.
4. For hybrid questions: gather BOTH document chunks AND database rows, then call `final_answer`.
5. For greetings, thanks, or small talk: call `final_answer` immediately with ready=true. Do NOT call any data tools and do NOT fabricate any information — the answer will be generated from conversation context only.
6. After gathering enough information, call `final_answer` with {{"ready": true, "suggestions": ["...", "...", "..."]}}. Keep this JSON SHORT — the answer prose is generated separately.
7. `final_answer.suggestions` must be exactly 3 short follow-up questions. Do NOT put the answer inside the JSON.
8. NEVER call the same tool twice with identical parameters.
9. NEVER hallucinate data — only use what tools return.
10. If a query_database call fails, fix the SQL (wrap names in backticks) and retry ONCE.

REASONING STYLE:
- Think: "What does the user need? What tool gives me that? What do I already know from tool results?"
- Synthesise multiple tool results into one coherent answer.
- Cite document sources as (filename, page N) inline in your answer.
"""


def build_agent_prompt(system: str, history: list[dict]) -> str:
    """
    Build the full prompt from system instruction + tool call history.
    history = [{"role": "tool_call"|"tool_result", "content": str}, ...]
    """
    parts = [
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system}\n<|eot_id|>"
    ]
    for h in history:
        if h["role"] == "user":
            parts.append(f"<|start_header_id|>user<|end_header_id|>\n{h['content']}\n<|eot_id|>")
        elif h["role"] == "tool_call":
            parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n{h['content']}\n<|eot_id|>")
        elif h["role"] == "tool_result":
            parts.append(f"<|start_header_id|>tool<|end_header_id|>\n{h['content']}\n<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n{")
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Memory persistence
# ══════════════════════════════════════════════════════════════════════════════
def save_turn(role: str, content: str):
    mem_col  = get_mem_collection()
    embedder = get_embedder()
    uid      = hashlib.md5(f"{SESSION_ID}|{time.time()}|{role}".encode()).hexdigest()
    mem_col.upsert(
        ids=[uid], documents=[content],
        embeddings=[embedder.encode(content).tolist()],
        metadatas=[{"session_id": SESSION_ID, "role": role,
                    "timestamp": datetime.utcnow().isoformat()}],
    )


# ══════════════════════════════════════════════════════════════════════════════
# A2T AGENT LOOP  ← the heart of the system
# ══════════════════════════════════════════════════════════════════════════════
def run_agent(user_question: str):
    """
    True A2T agentic loop. Yields step dicts for the UI.

    Protocol:
      1. Build system prompt with full tool registry
      2. LLM emits: {"tool": "...", "parameters": {...}, "reasoning": "..."}
      3. Python looks up & executes the tool
      4. Tool result appended to history as a tool_result message
      5. Repeat until LLM calls `final_answer` or MAX_AGENT_STEPS reached
    """
    system_prompt = build_agent_system_prompt()

    # Conversation history fed to LLM each step
    history = [{"role": "user", "content": user_question}]

    used_tools      = []    # track calls for display
    schema_retrieved = False  # gate: must call discover_schema before query_database
    final_answer = None
    sql_results  = []   # collected DB results for UI rendering
    doc_chunks   = []   # collected PDF chunks for UI rendering

    for step_num in range(MAX_AGENT_STEPS):

        # ── Ask LLM which tool to call ────────────────────────────────────────
        prompt   = build_agent_prompt(system_prompt, history)
        raw      = call_llm(prompt, max_tokens=TOOL_MAX_TOKENS)
        tool_obj = parse_tool_call(raw)

        tool_name   = tool_obj.get("tool", "")
        params      = tool_obj.get("parameters", {})
        reasoning   = tool_obj.get("reasoning", "")

        # Append LLM's tool call to history
        history.append({"role": "tool_call", "content": json.dumps(tool_obj)})

        # ── Validate tool name ────────────────────────────────────────────────
        if tool_name not in TOOL_REGISTRY:
            # Check if the LLM responded with plain prose instead of JSON.
            # This happens when it "answers" directly after recall_memory.
            # Treat the raw text as the final answer immediately — no nudging.
            raw_stripped = raw.strip()
            looks_like_prose = (
                not raw_stripped.startswith("{")
                and len(raw_stripped) > 20
                and not raw_stripped.startswith("[")
            )
            if looks_like_prose:
                # Use the prose directly as the ready signal
                final_answer = {"ready": True, "suggestions": []}
                # Stash the prose so _synthesise_answer can use it
                doc_chunks  = doc_chunks  # keep whatever was collected
                sql_results = sql_results
                # Override synthesis with the prose directly
                yield {
                    "final":       True,
                    "answer":      raw_stripped,
                    "suggestions": [],
                    "sql_results": sql_results,
                    "doc_chunks":  doc_chunks,
                    "used_tools":  used_tools,
                }
                return

            # Actual unknown tool name in JSON — nudge once, then give up
            nudge_count = sum(1 for h in history if h.get("role") == "tool_result"
                              and "Unknown tool" in h.get("content", ""))
            if nudge_count >= 2:
                # Too many nudges — bail out and synthesise from what we have
                final_answer = {"ready": True, "suggestions": []}
                break

            error_msg = (
                f"You must respond with a JSON tool call. "
                f"Unknown tool '{tool_name}'. "
                f"Valid tools: {list(TOOL_REGISTRY.keys())}. "
                f'Format: {{"tool": "<name>", "parameters": {{...}}, "reasoning": "..."}}'
            )
            history.append({"role": "tool_result", "content": json.dumps({"error": error_msg})})
            continue

        t_info = TOOL_REGISTRY[tool_name]
        yield {
            "step":      f"{t_info['emoji']} Tool Call {step_num+1}: `{tool_name}`",
            "detail":    f"**Reasoning:** {reasoning}",
            "tool_name": tool_name,
            "params":    params,
            "sql":       params.get("sql") if tool_name == "query_database" else None,
        }

        # ── final_answer terminates the loop ─────────────────────────────────
        if tool_name == "final_answer":
            final_answer = params
            break

        # ── Guard: block query_database until discover_schema has run ────────
        if tool_name == "query_database" and not schema_retrieved:
            force_msg = (
                "ERROR: You called query_database before discover_schema. "
                "You MUST call discover_schema first to learn the real table "
                "and column names. Never guess table names. Call discover_schema now."
            )
            history.append({"role": "tool_result",
                            "content": json.dumps({"error": force_msg})})
            yield {"step": "🚫 Blocked: query_database before discover_schema",
                   "detail": "Forcing schema discovery first to prevent guessed table/column names."}
            continue

        # ── Execute the tool ──────────────────────────────────────────────────
        tool_fn = t_info["fn"]
        try:
            result = tool_fn(**params)
        except Exception as e:
            result = {"error": str(e)}

        # Store results for UI
        if tool_name == "discover_schema" and "tables" in result:
            schema_retrieved = True   # unlock query_database
        if tool_name == "query_database" and result.get("success") and result.get("data"):
            sql_results = result["data"]
        if tool_name == "search_documents" and result.get("chunks"):
            doc_chunks = result["chunks"]

        # Append tool result to history (compact — cap large payloads)
        result_str = json.dumps(result)
        if len(result_str) > 3000:
            # Truncate to avoid blowing context window
            if "data" in result:
                compact = {**result, "data": result["data"][:MAX_EXPLAIN_ROWS],
                           "note": f"Showing {MAX_EXPLAIN_ROWS} of {result.get('row_count',len(result['data']))} rows"}
                result_str = json.dumps(compact)
            elif "chunks" in result:
                compact = {**result, "chunks": result["chunks"][:TOP_K_FINAL]}
                result_str = json.dumps(compact)

        history.append({"role": "tool_result", "content": result_str})
        used_tools.append(tool_name)

        # Yield result summary for UI
        yield {
            "step":        f"✅ Result from `{tool_name}`",
            "detail":      _summarise_result(tool_name, result),
            "tool_name":   tool_name,
            "raw_result":  result,
            "sql_results": sql_results,
            "doc_chunks":  doc_chunks,
        }

    # ── Produce final output ──────────────────────────────────────────────────
    # Always synthesise the answer in a SEPARATE high-token LLM call.
    # This decouples the short tool-call JSON from the long prose answer,
    # completely eliminating truncation errors.
    suggestions = []
    if final_answer:
        raw_sug = final_answer.get("suggestions", [])
        if isinstance(raw_sug, str):
            try:   raw_sug = json.loads(raw_sug)
            except: raw_sug = []
        suggestions = [str(s) for s in raw_sug if str(s).strip()][:3]

    answer = _synthesise_answer(user_question, history, doc_chunks, sql_results)

    yield {
        "final":       True,
        "answer":      answer,
        "suggestions": suggestions,
        "sql_results": sql_results,
        "doc_chunks":  doc_chunks,
        "used_tools":  used_tools,
    }


def _synthesise_answer(question: str, history: list[dict],
                       doc_chunks: list[dict], sql_results: list[dict]) -> str:
    """
    Separate high-token LLM call that turns collected tool results into prose.
    Runs AFTER the tool loop — never inside it — so token budget is never an issue.

    STRICT GROUNDING RULE: if no data was retrieved, the LLM must say so and
    ask a clarifying question. It must NEVER invent content, cite phantom sources,
    or draw on general knowledge.
    """
    has_docs = bool(doc_chunks)
    has_db   = bool(sql_results)

    # ── No data retrieved at all → conversational reply only, no fabrication ──
    if not has_docs and not has_db:
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"You are OptumRx Billing Agent, a friendly assistant.\n"
            f"STRICT RULES:\n"
            f"- You have NO documents and NO database records for this question.\n"
            f"- Do NOT invent facts, policies, page numbers, or citations.\n"
            f"- Do NOT pretend to know things you haven't been shown.\n"
            f"- If it is a greeting or small talk, respond warmly and briefly.\n"
            f"- If it is a real question, say you don't have that information yet "
            f"and ask the user to upload relevant documents or check the database.\n"
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{question}\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        return call_llm(prompt, max_tokens=200)

    # ── Build grounded context block from actual retrieved data ───────────────
    parts = []

    if has_docs:
        doc_ctx = "\n\n---\n".join(
            f"[{c['filename']} | Page {c['page_num']}]\n{c['text']}"
            for c in doc_chunks
        )
        parts.append(f"DOCUMENT EXCERPTS (retrieved from indexed PDFs):\n{doc_ctx}")

    if has_db:
        parts.append(
            f"DATABASE RECORDS (first {MAX_EXPLAIN_ROWS} rows):\n"
            + json.dumps(sql_results[:MAX_EXPLAIN_ROWS], indent=2)
        )

    context_block = "\n\n".join(parts)

    # Show the LLM what data types are present so it knows what to use
    source_note = []
    if has_docs: source_note.append(f"{len(doc_chunks)} document chunk(s)")
    if has_db:   source_note.append(f"{len(sql_results)} database row(s)")
    sources_present = " and ".join(source_note)

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"You are OptumRx Billing Agent.\n"
        f"You have been given REAL retrieved data ({sources_present}). "
        f"You MUST use it to answer the question.\n\n"
        f"RULES:\n"
        f"1. Base your answer ENTIRELY on the CONTEXT section below.\n"
        f"2. The context contains real data — you MUST use it. "
           f"Do NOT say you lack information when context is provided.\n"
        f"3. For document chunks: cite inline as (filename, page N).\n"
        f"4. For database rows: summarise the key facts clearly.\n"
        f"5. Do NOT add disclaimers or hedging sentences after answering.\n"
        f"6. Stop as soon as the question is fully answered.\n"
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"Question: {question}\n\n"
        f"CONTEXT ({sources_present}):\n{context_block}\n"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    return call_llm(prompt, max_tokens=ANSWER_MAX_TOKENS)


def _summarise_result(tool_name: str, result: dict) -> str:
    """Human-readable one-liner summary of a tool result for the step expander."""
    if tool_name == "recall_memory":
        n = len(result.get("turns", []))
        return f"Retrieved **{n}** relevant memory turn(s)."
    if tool_name == "discover_schema":
        tables = result.get("tables", [])
        return f"Found tables: `{'`, `'.join(tables)}`" if tables else result.get("error","")
    if tool_name == "query_database":
        if result.get("success"):
            return f"Returned **{result.get('row_count',0)}** row(s)."
        return f"Error: `{result.get('error','unknown')}`"
    if tool_name == "search_documents":
        n = len(result.get("chunks", []))
        if n:
            sources = list({c["filename"] for c in result["chunks"]})
            return f"Retrieved **{n}** chunk(s) from: {', '.join(f'`{s}`' for s in sources)}"
        return result.get("message", "No chunks found.")
    return json.dumps(result)[:200]


# ── Helper: render one assistant turn (steps + answer + tables) ──────────────
def render_assistant_turn(final: dict, msg_index: int):
    """Render all tool steps and the final answer for one assistant turn."""
    # Tool steps — always shown, collapsed by default
    for step in final.get("steps", []):
        with st.expander(step["step"], expanded=False):
            if step.get("detail"):
                st.markdown(step["detail"])
            if step.get("sql"):
                st.code(step["sql"], language="sql")

            raw = step.get("raw_result", {})

            if step.get("tool_name") == "recall_memory" and raw.get("turns"):
                for t in raw["turns"]:
                    prefix = "🧑 User" if t["role"] == "user" else "🤖 Assistant"
                    st.caption(f"{prefix}: {t['content'][:200]}")

            if step.get("tool_name") == "discover_schema" and raw.get("schema_text"):
                st.code(raw["schema_text"], language="text")

            if step.get("tool_name") == "query_database" and raw.get("data"):
                st.dataframe(pd.DataFrame(raw["data"][:5]),
                             use_container_width=True, hide_index=True)

            if step.get("tool_name") == "search_documents" and raw.get("chunks"):
                for c in raw["chunks"]:
                    score = c.get("rerank_score", c.get("score", "?"))
                    st.markdown(f"**📄 {c['filename']} — p.{c['page_num']}** (score: {score})")
                    st.caption(c["text"][:250] + ("…" if len(c["text"]) > 250 else ""))

    # Tools-used badge
    used = final.get("used_tools", [])
    if used:
        badges = " ".join(
            f'`{TOOL_REGISTRY[t]["emoji"]} {t}`'
            for t in dict.fromkeys(used)
            if t in TOOL_REGISTRY
        )
        st.caption(f"Tools used: {badges}")

    # Answer
    answer = final.get("answer", "")
    st.subheader("💬 Answer")
    st.markdown(answer)

    # Document citations
    chunks = final.get("doc_chunks", [])
    if chunks:
        st.divider()
        st.subheader("📎 Document Sources")
        for i, c in enumerate(chunks, 1):
            score = c.get("rerank_score", c.get("score", "?"))
            with st.expander(
                f"{i}. {c['filename']} — Page {c['page_num']}  (relevance: {score})",
                expanded=False,
            ):
                st.markdown(c["text"])

    # Database results
    rows = final.get("sql_results", [])
    if rows:
        st.divider()
        st.subheader("📋 Database Results")
        df = pd.DataFrame(rows)
        m = st.columns(4)
        m[0].metric("Rows", len(rows))
        if "billed" in df.columns:
            m[1].metric("Total Billed", f"${df['billed'].sum():,.2f}")
        if "administered" in df.columns:
            m[2].metric("Total Administered", f"${df['administered'].sum():,.2f}")
        if {"billed","administered"}.issubset(df.columns):
            gap = df["billed"].sum() - df["administered"].sum()
            m[3].metric("Billing Gap", f"${gap:,.2f}",
                        delta=f"${gap:,.2f}", delta_color="inverse")
        col_cfg = {}
        for col in df.columns:
            if col.lower() in ("billed","administered"):
                col_cfg[col] = st.column_config.NumberColumn(
                    col.replace("_"," ").title(), format="$%.2f")
            elif "date" in col.lower():
                col_cfg[col] = st.column_config.DateColumn(col.replace("_"," ").title())
            else:
                col_cfg[col] = st.column_config.TextColumn(col.replace("_"," ").title())
        st.dataframe(df, use_container_width=True, hide_index=True, column_config=col_cfg)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols and len(df) > 1:
            cc = st.selectbox("📈 Visualise:", num_cols, key=f"chart_{msg_index}")
            lc = next((c for c in df.columns if "name" in c.lower()), None)
            st.bar_chart(df.set_index(lc)[cc] if lc else df[cc])



# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  .title    {font-size:2rem;font-weight:700;color:#FF612B;}
  .subtitle {color:#888;margin-top:-0.5rem;margin-bottom:1rem;font-size:0.9rem;}
  .tool-badge {
    background:#FF612B22;border:1px solid #FF612B55;border-radius:6px;
    padding:2px 8px;font-size:0.8rem;color:#FF612B;font-weight:600;
  }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">💊 OptumRx Billing Agent</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">True A2T Protocol · LLM drives tool selection · '
    f'Powered by {OLLAMA_MODEL} + ChromaDB · Fully local & private</p>',
    unsafe_allow_html=True,
)

# ── Init collections ──────────────────────────────────────────────────────────
try:
    doc_col = get_doc_collection()
    mem_col = get_mem_collection()
except Exception as e:
    st.error(f"ChromaDB error: {e}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💊 OptumRx Billing Agent")
    st.caption("A2T Protocol — LLM selects tools autonomously")
    st.divider()

    # A2T tools overview
    with st.expander("🔧 Registered A2T Tools", expanded=False):
        for name, t in TOOL_REGISTRY.items():
            st.markdown(f"**{t['emoji']} `{name}`**")
            st.caption(t["description"])

    st.divider()
    st.subheader("📚 Document Management")
    fresh = st.toggle("🗑️ Start fresh vector store", value=False)
    if fresh:
        st.warning("Fresh mode ON — clears on next upload.")

    stats = collection_stats(doc_col)
    c1, c2 = st.columns(2)
    c1.metric("Chunks", stats["count"])
    c2.metric("Docs",   len(stats["sources"]))
    if stats["sources"]:
        st.caption("Indexed:")
        for s in stats["sources"]:
            st.markdown(f"  - 📄 `{s}`")

    st.divider()
    uploaded = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if uploaded and st.button("📥 Index PDFs", type="primary"):
        if fresh:
            doc_col = get_doc_collection(fresh=True)
            st.info("Store cleared.")
        prog = st.progress(0, "Starting…")
        for i, f in enumerate(uploaded):
            prog.progress(i / len(uploaded), f"Indexing {f.name}…")
            try:
                r = ingest_pdf(f.read(), f.name, doc_col)
                st.success(f"✅ **{r['filename']}** — {r['pages']}p, {r['chunks']} chunks")
            except Exception as e:
                st.error(f"`{f.name}`: {e}")
        prog.progress(1.0, "Done!")
        st.rerun()

    st.divider()
    st.subheader("🧠 Memory")
    st.metric("Turns stored", mem_col.count())
    st.caption(f"Session: `{SESSION_ID[:12]}…`")
    if st.button("🗑️ Clear Memory"):
        try:
            ids = mem_col.get(where={"session_id": SESSION_ID})["ids"]
            if ids: mem_col.delete(ids=ids)
            st.session_state.messages    = []
            st.session_state.suggestions = []
            st.success("Cleared.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.subheader("🔌 Ollama")
    try:
        requests.get("http://localhost:11434", timeout=2)
        st.success("Running ✅")
        avail = [m["name"] for m in requests.get("http://localhost:11434/api/tags",
                                                   timeout=2).json().get("models",[])]
        if any(OLLAMA_MODEL in m for m in avail):
            st.success(f"`{OLLAMA_MODEL}` ready ✅")
        else:
            st.warning(f"`ollama pull {OLLAMA_MODEL}`")
    except Exception:
        st.error("Not reachable ❌\nRun: `ollama serve`")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "suggestions" not in st.session_state: st.session_state.suggestions = []
if "pending"     not in st.session_state: st.session_state.pending     = None

# ── Suggestion buttons ────────────────────────────────────────────────────────
if st.session_state.suggestions:
    st.markdown("**💡 Suggested follow-ups:**")
    cols = st.columns(len(st.session_state.suggestions))
    for i, (col, sug) in enumerate(zip(cols, st.session_state.suggestions)):
        if col.button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state.pending     = sug
            st.session_state.suggestions = []
            st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("final"):
            # Replay full turn: tool steps + answer + tables
            render_assistant_turn(msg["final"], idx)
        else:
            st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
typed      = st.chat_input("Ask anything about billing, documents, or say hello…")
user_input = st.session_state.pending or typed
if st.session_state.pending:
    st.session_state.pending = None

# ── Process ───────────────────────────────────────────────────────────────────
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_turn("user", user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        collected_steps = []
        final = None

        # ── Stream & render steps live ────────────────────────────────────────
        for step in run_agent(user_input):
            if step.get("final"):
                final = step
                break

            # Render live
            with st.expander(step["step"], expanded=False):
                if step.get("detail"):
                    st.markdown(step["detail"])
                if step.get("sql"):
                    st.code(step["sql"], language="sql")
                raw = step.get("raw_result", {})
                if step.get("tool_name") == "recall_memory" and raw.get("turns"):
                    for t in raw["turns"]:
                        prefix = "🧑 User" if t["role"] == "user" else "🤖 Assistant"
                        st.caption(f"{prefix}: {t['content'][:200]}")
                if step.get("tool_name") == "discover_schema" and raw.get("schema_text"):
                    st.code(raw["schema_text"], language="text")
                if step.get("tool_name") == "query_database" and raw.get("data"):
                    st.dataframe(pd.DataFrame(raw["data"][:5]),
                                 use_container_width=True, hide_index=True)
                if step.get("tool_name") == "search_documents" and raw.get("chunks"):
                    for c in raw["chunks"]:
                        score = c.get("rerank_score", c.get("score", "?"))
                        st.markdown(f"**📄 {c['filename']} — p.{c['page_num']}** (score: {score})")
                        st.caption(c["text"][:250] + ("…" if len(c["text"]) > 250 else ""))

            # Save step for persistence
            collected_steps.append(step)

        # ── Render final answer ───────────────────────────────────────────────
        if final:
            final["steps"] = collected_steps   # attach steps to the final payload

            # Tools-used badge
            used = final.get("used_tools", [])
            if used:
                badges = " ".join(
                    f'`{TOOL_REGISTRY[t]["emoji"]} {t}`'
                    for t in dict.fromkeys(used)
                    if t in TOOL_REGISTRY
                )
                st.caption(f"Tools used: {badges}")

            answer = final.get("answer", "")
            st.subheader("💬 Answer")
            st.markdown(answer)

            # Document citations
            chunks = final.get("doc_chunks", [])
            if chunks:
                st.divider()
                st.subheader("📎 Document Sources")
                for i, c in enumerate(chunks, 1):
                    score = c.get("rerank_score", c.get("score", "?"))
                    with st.expander(
                        f"{i}. {c['filename']} — Page {c['page_num']}  (relevance: {score})",
                        expanded=False,
                    ):
                        st.markdown(c["text"])

            # Database results
            rows = final.get("sql_results", [])
            if rows:
                st.divider()
                st.subheader("📋 Database Results")
                df = pd.DataFrame(rows)
                m = st.columns(4)
                m[0].metric("Rows", len(rows))
                if "billed" in df.columns:
                    m[1].metric("Total Billed", f"${df['billed'].sum():,.2f}")
                if "administered" in df.columns:
                    m[2].metric("Total Administered", f"${df['administered'].sum():,.2f}")
                if {"billed","administered"}.issubset(df.columns):
                    gap = df["billed"].sum() - df["administered"].sum()
                    m[3].metric("Billing Gap", f"${gap:,.2f}",
                                delta=f"${gap:,.2f}", delta_color="inverse")
                col_cfg = {}
                for col in df.columns:
                    if col.lower() in ("billed","administered"):
                        col_cfg[col] = st.column_config.NumberColumn(
                            col.replace("_"," ").title(), format="$%.2f")
                    elif "date" in col.lower():
                        col_cfg[col] = st.column_config.DateColumn(col.replace("_"," ").title())
                    else:
                        col_cfg[col] = st.column_config.TextColumn(col.replace("_"," ").title())
                st.dataframe(df, use_container_width=True, hide_index=True, column_config=col_cfg)
                num_cols = df.select_dtypes(include="number").columns.tolist()
                if num_cols and len(df) > 1:
                    cc = st.selectbox("📈 Visualise:", num_cols,
                                      key=f"chart_{len(st.session_state.messages)}")
                    lc = next((c for c in df.columns if "name" in c.lower()), None)
                    st.bar_chart(df.set_index(lc)[cc] if lc else df[cc])

            # ── Persist to session state ──────────────────────────────────────
            save_turn("assistant", answer)
            # Store full final payload (with steps) so reruns can replay it
            st.session_state.messages.append({
                "role":    "assistant",
                "content": answer,
                "final":   final,          # ← steps + chunks + sql_results all here
            })
            st.session_state.suggestions = final.get("suggestions", [])
            if st.session_state.suggestions:
                st.rerun()