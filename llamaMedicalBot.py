"""
OptumRx Billing Agent
======================
• Upload PDFs → sentence-aware chunked → embedded → ChromaDB (persistent)
• Persistent conversation memory stored in ChromaDB (separate collection)
• Query sources: PDF docs | Healthcare SQLite DB | Both | General chat
• Suggestive follow-up prompts after every answer
• Better retrieval: sentence-aware chunking + cross-encoder re-ranking

Dependencies:
    pip install streamlit chromadb sentence-transformers pypdf requests pandas
    pip install "sentence-transformers[cross-encoder]"
"""

import io, json, re, sqlite3, time, hashlib, uuid
from datetime import datetime
from pathlib import Path

import chromadb
import requests
import streamlit as st
import pandas as pd
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder

# Page config
st.set_page_config(page_title="OptumRx Billing Agent", page_icon="\U0001f48a", layout="wide")

OPTUM_ORANGE = "#FF612B"
st.markdown(f"""
<style>
  .stButton>button {{border-color:{OPTUM_ORANGE};color:{OPTUM_ORANGE};}}
  .stButton>button:hover {{background:{OPTUM_ORANGE};color:white;}}
</style>
""", unsafe_allow_html=True)

# Config
OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "llama3.1:8b"
DB_PATH        = "healthcare.db"
CHROMA_DIR     = "./chroma_store"
DOC_COLLECTION = "optumrx_docs"
MEM_COLLECTION = "optumrx_memory"
EMBED_MODEL    = "all-MiniLM-L6-v2"
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SENT_CHUNK_SIZE = 6
SENT_OVERLAP    = 2
MIN_CHUNK_CHARS = 80
TOP_K_FETCH     = 12
TOP_K_RERANK    = 4
MEM_TURNS       = 6
MAX_SAMPLE_ROWS  = 1
MAX_OTHER_TABLES = 3
MAX_EXPLAIN_ROWS = 5


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Loading re-ranker...")
def get_reranker():
    try:
        return CrossEncoder(RERANK_MODEL)
    except Exception:
        return None


@st.cache_resource(show_spinner="Connecting to vector store...")
def get_chroma_client():
    return chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def get_doc_collection(fresh=False):
    client = get_chroma_client()
    if fresh:
        try: client.delete_collection(DOC_COLLECTION)
        except: pass
    return client.get_or_create_collection(DOC_COLLECTION, metadata={"hnsw:space": "cosine"})


def get_mem_collection():
    return get_chroma_client().get_or_create_collection(
        MEM_COLLECTION, metadata={"hnsw:space": "cosine"}
    )


# Sentence-aware chunking
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

def split_sentences(text):
    raw = _SENT_RE.split(text.strip())
    sents = []
    for s in raw:
        for line in s.split('\n'):
            line = line.strip()
            if len(line) > 20:
                sents.append(line)
    return sents


def chunk_sentences(sentences, size=SENT_CHUNK_SIZE, overlap=SENT_OVERLAP):
    if not sentences:
        return []
    chunks, step = [], max(1, size - overlap)
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i+size])
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
    return chunks


def extract_pages(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return [
        {"page_num": i+1, "text": (p.extract_text() or "").strip()}
        for i, p in enumerate(reader.pages)
        if (p.extract_text() or "").strip()
    ]


def ingest_pdf(pdf_bytes, filename, collection):
    embedder    = get_embedder()
    pages       = extract_pages(pdf_bytes)
    upload_ts   = datetime.utcnow().isoformat()
    total_pages = len(pages)
    all_ids, all_docs, all_embeds, all_metas = [], [], [], []
    for page in pages:
        for ci, chunk in enumerate(chunk_sentences(split_sentences(page["text"]))):
            uid = hashlib.md5(f"{filename}|{page['page_num']}|{ci}".encode()).hexdigest()
            all_ids.append(uid)
            all_docs.append(chunk)
            all_embeds.append(embedder.encode(chunk).tolist())
            all_metas.append({
                "filename": filename, "page_num": page["page_num"],
                "chunk_index": ci, "total_pages": total_pages, "uploaded_at": upload_ts,
            })
    for i in range(0, len(all_ids), 100):
        collection.upsert(
            ids=all_ids[i:i+100], documents=all_docs[i:i+100],
            embeddings=all_embeds[i:i+100], metadatas=all_metas[i:i+100],
        )
    return {"filename": filename, "pages": total_pages, "chunks": len(all_ids)}


def vector_search(query, collection, top_k_fetch=TOP_K_FETCH, top_k_final=TOP_K_RERANK):
    if collection.count() == 0:
        return []
    embedder = get_embedder()
    n        = min(top_k_fetch, collection.count())
    results  = collection.query(
        query_embeddings=[embedder.encode(query).tolist()], n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    candidates = [{
        "text": doc, "filename": meta.get("filename", "?"),
        "page_num": meta.get("page_num", "?"), "chunk_index": meta.get("chunk_index", 0),
        "score": round(1 - dist, 4),
    } for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    )]
    reranker = get_reranker()
    if reranker and len(candidates) > top_k_final:
        scores = reranker.predict([(query, c["text"]) for c in candidates])
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
    return candidates[:top_k_final]


def save_turn(session_id, role, content, mem_col):
    embedder = get_embedder()
    mem_col.upsert(
        ids=[str(uuid.uuid4())],
        documents=[content],
        embeddings=[embedder.encode(content).tolist()],
        metadatas=[{"session_id": session_id, "role": role, "timestamp": datetime.utcnow().isoformat()}],
    )


def recall_memory(session_id, query, mem_col, n=MEM_TURNS):
    if mem_col.count() == 0:
        return []
    n_fetch = min(n * 3, mem_col.count())
    results = mem_col.query(
        query_embeddings=[get_embedder().encode(query).tolist()],
        n_results=n_fetch, include=["documents", "metadatas"],
    )
    turns = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        if meta.get("session_id") == session_id:
            turns.append({"role": meta["role"], "content": doc, "timestamp": meta.get("timestamp","")})
        if len(turns) >= n:
            break
    turns.sort(key=lambda x: x["timestamp"])
    return turns


def format_memory(turns):
    if not turns:
        return ""
    lines = ["Recent conversation:"]
    for t in turns:
        lines.append(f"{'User' if t['role']=='user' else 'Assistant'}: {t['content'][:300]}")
    return "\n".join(lines)


def discover_schema():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    all_tables = [r["name"] for r in cur.fetchall()]
    priority = [t for t in all_tables if t == "billing"]
    others   = [t for t in all_tables if t != "billing"][:MAX_OTHER_TABLES]
    schema   = {}
    for table in priority + others:
        cur.execute(f"PRAGMA table_info({table})")
        cols = [{"name": r["name"], "type": r["type"], "notnull": bool(r["notnull"]), "pk": bool(r["pk"])} for r in cur.fetchall()]
        cur.execute(f"SELECT COUNT(*) AS cnt FROM {table}")
        rc = cur.fetchone()["cnt"]
        sample = []
        if table == "billing":
            cur.execute(f"SELECT * FROM {table} LIMIT {MAX_SAMPLE_ROWS}")
            sample = [dict(r) for r in cur.fetchall()]
        schema[table] = {"columns": cols, "sample_rows": sample, "row_count": rc}
    conn.close()
    return schema


def schema_to_text(schema):
    lines = []
    for table, info in schema.items():
        parts = []
        for c in info["columns"]:
            flags = ("PK " if c["pk"] else "") + ("NN" if c["notnull"] else "")
            parts.append(f"`{c['name']}` {c['type']}{(' ' + flags.strip()) if flags.strip() else ''}")
        lines.append(f"{table}({', '.join(parts)}) — {info['row_count']} rows")
        if info["sample_rows"]:
            lines.append(f"  e.g. {info['sample_rows'][0]}")
    return "\n".join(lines)


def run_sql(query):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return {"success": True, "message": "No rows returned.", "data": [], "row_count": 0}
        return {"success": True, "data": [dict(r) for r in rows], "row_count": len(rows)}
    except Exception as e:
        return {"success": False, "error": str(e), "data": []}


def call_llm(prompt, max_tokens=700, temperature=0.2):
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
               "options": {"temperature": temperature, "num_predict": max_tokens}}
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


def parse_json(raw):
    text = raw.strip()
    if not text.startswith("{"): text = "{" + text
    text = text.lstrip("```json").lstrip("```").strip()
    depth = end = 0
    for i, ch in enumerate(text):
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: end = i+1; break
    return json.loads(text[:end] if end else text)


# ── Prompts ──────────────────────────────────────────────────────────────────

def build_intent_prompt(question, memory_ctx, has_docs, has_db):
    sources = []
    if has_docs: sources.append('"' + "pdf" + '" — questions about uploaded policy/clinical documents')
    if has_db:   sources.append('"' + "db"  + '" — billing records, invoices, patient data')
    src = "\n".join(f"  \u2022 {s}" for s in sources) or "  \u2022 none"
    mem = f"\n{memory_ctx}\n" if memory_ctx else ""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an intent classifier for OptumRx Billing Agent. Output ONLY raw JSON.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{mem}Available sources:
{src}

Message: "{question}"

Classify:
- "chat"  -> greetings, general knowledge, thanks, questions about the assistant, small talk, anything that does NOT need documents or database records
- "pdf"   -> needs content from uploaded PDFs only
- "db"    -> needs structured records from database only
- "both"  -> needs both PDFs and database

CRITICAL: Use "chat" for ANY greeting or general question even if sources exist.
For "chat" write a warm helpful reply in "reply". For others leave "reply" empty.

Output: {{"intent":"chat|pdf|db|both","reasoning":"one sentence","reply":"..."}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{"""


def build_chat_prompt(question, memory_ctx):
    mem = f"\n{memory_ctx}\n" if memory_ctx else ""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are OptumRx Billing Agent, a friendly and knowledgeable AI assistant for OptumRx pharmacy billing.
You help with billing, policy lookups, healthcare analytics, and general conversation.
Be warm, concise, and professional. If you don't know something, say so honestly.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{mem}User: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def build_sql_prompt(schema_text, question, memory_ctx):
    mem = f"\nConversation context:\n{memory_ctx}\n" if memory_ctx else ""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a SQLite expert. Output ONLY raw JSON.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Schema:
{schema_text}
{mem}
Question: {question}

Rules:
- Prefer 'billing' table unless another is clearly needed.
- Never SELECT * — only needed columns.
- ALWAYS wrap column and table names in backticks.
- Think: which table? which columns? which filters? which ORDER/LIMIT?

Output: {{"reasoning":"step-by-step reasoning","sql":"SELECT ..."}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{"""


def build_fix_prompt(schema_text, bad_sql, error):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
SQLite expert. Output ONLY raw JSON.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Schema: {schema_text}
Failed: {bad_sql}
Error: {error}
Output: {{"sql":"fixed SELECT ..."}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{"""


def build_pdf_prompt(question, chunks, memory_ctx):
    context = "\n\n---\n\n".join(
        f"[SOURCE: {c['filename']}, Page {c['page_num']}]\n{c['text']}" for c in chunks
    )
    mem = f"\n{memory_ctx}\n" if memory_ctx else ""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are OptumRx Billing Agent, an expert healthcare document analyst.
Instructions:
1. Read ALL provided context carefully before answering.
2. Answer ONLY from the context — never hallucinate facts.
3. Cite every fact with (filename, page N).
4. If multiple sources cover the topic, synthesise them.
5. If the answer is not in the context, clearly say so.
6. Use bullet points for multi-part answers. Be thorough yet concise.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{mem}Context:
{context}

Question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def build_db_prompt(question, sql, rows, memory_ctx):
    mem = f"\n{memory_ctx}\n" if memory_ctx else ""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are OptumRx Billing Agent, an expert in pharmacy billing analytics.
Instructions:
1. Analyse the query results carefully.
2. Provide a structured answer with 3-5 bullet points.
3. Highlight discrepancies (billed vs administered), anomalies, totals, trends.
4. Use $ for monetary values. Reference specific records where helpful.
5. Do NOT repeat raw data verbatim — interpret and summarise it.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{mem}Question: {question}
SQL: {sql}
Results ({len(rows)} rows): {json.dumps(rows[:MAX_EXPLAIN_ROWS])}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def build_hybrid_prompt(question, chunks, sql_data, memory_ctx):
    doc_ctx = "\n\n---\n\n".join(
        f"[SOURCE: {c['filename']}, Page {c['page_num']}]\n{c['text']}" for c in chunks
    )
    mem = f"\n{memory_ctx}\n" if memory_ctx else ""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are OptumRx Billing Agent with access to both clinical documents and billing records.
Instructions:
1. Synthesise information from BOTH sources in your answer.
2. Clearly label what comes from documents vs database.
3. Cite documents as (filename, page N).
4. Format monetary values with dollar signs.
5. Use bullet points. Be concise but complete.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{mem}Document excerpts:
{doc_ctx}

Database records: {json.dumps(sql_data[:MAX_EXPLAIN_ROWS])}

Question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def build_suggestions_prompt(question, answer, intent):
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Generate follow-up suggestions. Output ONLY a JSON array, no markdown.
<|eot_id|><|start_header_id|>user<|end_header_id|>
User asked: "{question}"
Source type: {intent}
Answer summary: {answer[:250]}

Generate exactly 3 short, specific follow-up questions.
Make them varied: one about details, one about patterns/trends, one broader.
Output: ["question 1","question 2","question 3"]
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
["""


def get_suggestions(question, answer, intent):
    try:
        raw  = call_llm(build_suggestions_prompt(question, answer, intent),
                        max_tokens=150, temperature=0.7)
        text = raw.strip()
        if not text.startswith("["): text = "[" + text
        if not text.rstrip().endswith("]"): text = text.rstrip().rstrip(",") + "]"
        return [s for s in json.loads(text) if isinstance(s, str)][:3]
    except Exception:
        return []


# ── Agent ─────────────────────────────────────────────────────────────────────

def run_agent(question, doc_col, mem_col, session_id):
    has_docs = doc_col.count() > 0
    has_db   = Path(DB_PATH).exists()

    recent_turns = recall_memory(session_id, question, mem_col)
    memory_ctx   = format_memory(recent_turns)

    yield {"step": "\U0001f914 Understanding Your Request", "detail": "Classifying intent..."}

    try:
        intent_raw = call_llm(build_intent_prompt(question, memory_ctx, has_docs, has_db), max_tokens=120)
        intent_obj = parse_json(intent_raw)
    except Exception as e:
        intent_obj = {"intent": "chat", "reasoning": f"Parse error: {e}", "reply": ""}

    intent    = intent_obj.get("intent", "chat")
    reasoning = intent_obj.get("reasoning", "")
    llm_reply = intent_obj.get("reply", "").strip()

    ICONS = {"chat": "\U0001f4ac", "pdf": "\U0001f4c4", "db": "\U0001f5c4", "both": "\U0001f500"}
    yield {"step": f"{ICONS.get(intent,'?')}: {intent.upper()}", "detail": f"**Reasoning:** {reasoning}"}

    # CHAT
    if intent == "chat":
        if llm_reply and len(llm_reply) > 10:
            explanation = llm_reply
        else:
            yield {"step": "\U0001f4ac Generating Reply", "detail": ""}
            explanation = call_llm(build_chat_prompt(question, memory_ctx), max_tokens=400, temperature=0.4)
        suggestions = get_suggestions(question, explanation, "general")
        yield {"final": True, "intent": "chat", "explanation": explanation,
               "result": None, "sql": None, "chunks": [], "suggestions": suggestions}
        return

    # PDF
    pdf_chunks = []
    if intent in ("pdf", "both"):
        if not has_docs:
            yield {"step": "\u26a0\ufe0f No Documents", "detail": "Upload PDFs first."}
            if intent == "pdf":
                yield {"final": True, "intent": "pdf",
                       "explanation": "No documents indexed yet. Please upload a PDF first.",
                       "result": None, "sql": None, "chunks": [], "suggestions": []}
                return
        else:
            yield {"step": "\U0001f50d Semantic Search + Re-ranking",
                   "detail": f"Fetching {TOP_K_FETCH} candidates, re-ranking to {TOP_K_RERANK}..."}
            pdf_chunks = vector_search(question, doc_col, TOP_K_FETCH, TOP_K_RERANK)
            yield {"step": f"\u2705 {len(pdf_chunks)} Chunks Selected",
                   "detail": "Top chunks after re-ranking:", "chunks": pdf_chunks}

    # DB
    sql_result  = None
    sql_query   = None
    schema_text = ""
    if intent in ("db", "both"):
        if not has_db:
            yield {"step": "\u26a0\ufe0f DB Not Found", "detail": f"`{DB_PATH}` missing."}
        else:
            n = "2" if intent == "db" else "3"
            yield {"step": f"\U0001f5c4\ufe0f Step {n}: Discovering Schema", "detail": ""}
            schema      = discover_schema()
            schema_text = schema_to_text(schema)
            yield {"step": f"\U0001f9e0 Step {n}b: Planning SQL", "detail": ""}
            raw_sql = call_llm(build_sql_prompt(schema_text, question, memory_ctx))
            try:
                plan       = parse_json(raw_sql)
                sql_query  = plan["sql"]
                sql_reason = plan.get("reasoning","")
            except Exception:
                sql_query  = next((l.strip() for l in raw_sql.splitlines() if "SELECT" in l.upper()), None)
                sql_reason = raw_sql
            yield {"step": "\U0001f4a1 Query Plan", "detail": f"**Why:** {sql_reason}", "sql": sql_query}

            if sql_query:
                yield {"step": f"\u2699\ufe0f Step {n}c: Executing SQL",
                       "detail": f"Running on `{DB_PATH}`...", "sql": sql_query}
                sql_result = run_sql(sql_query)
                if not sql_result["success"]:
                    yield {"step": "\U0001f527 Self-Correcting",
                           "detail": f"Error: `{sql_result.get('error')}` — fixing..."}
                    try:
                        fix_raw   = call_llm(build_fix_prompt(schema_text, sql_query, sql_result.get("error","")))
                        sql_query = parse_json(fix_raw)["sql"]
                        yield {"step": "\U0001f504 Retrying", "sql": sql_query, "detail": ""}
                        sql_result = run_sql(sql_query)
                    except Exception as fe:
                        sql_result = {"success": False, "error": str(fe), "data": []}
                if sql_result.get("success"):
                    yield {"step": f"\u2705 DB: {sql_result['row_count']} Row(s)", "detail": ""}

    yield {"step": "\U0001f4dd Generating Answer", "detail": "Synthesising with reasoning..."}

    if intent == "pdf":
        explanation = call_llm(build_pdf_prompt(question, pdf_chunks, memory_ctx), max_tokens=800)
    elif intent == "db":
        rows = sql_result.get("data",[]) if sql_result else []
        explanation = call_llm(build_db_prompt(question, sql_query or "", rows, memory_ctx), max_tokens=600)
    elif intent == "both":
        db_rows = sql_result.get("data",[]) if (sql_result and sql_result.get("success")) else []
        explanation = call_llm(build_hybrid_prompt(question, pdf_chunks, db_rows, memory_ctx), max_tokens=900)
    else:
        explanation = "I could not determine how to answer this question."

    suggestions = get_suggestions(question, explanation, intent)
    yield {"final": True, "intent": intent, "explanation": explanation,
           "result": sql_result, "sql": sql_query, "chunks": pdf_chunks, "suggestions": suggestions}


# ── Helper ────────────────────────────────────────────────────────────────────
def collection_stats(col):
    count   = col.count()
    sources = set()
    if count > 0:
        r = col.get(limit=min(count, 500), include=["metadatas"])
        for m in r["metadatas"]:
            sources.add(m.get("filename","unknown"))
    return {"count": count, "sources": sorted(sources)}


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

# Header
col_logo, col_title = st.columns([1, 9])
with col_logo:
    st.markdown(
        f'<div style="background:{OPTUM_ORANGE};border-radius:10px;width:56px;height:56px;'        f'display:flex;align-items:center;justify-content:center;font-size:30px;margin-top:6px">\U0001f48a</div>',
        unsafe_allow_html=True,
    )
with col_title:
    st.markdown(
        f'<h1 style="margin:0">OptumRx <span style="color:{OPTUM_ORANGE}">Billing Agent</span></h1>'        f'<p style="color:#888;margin:0;font-size:13px">Powered by {OLLAMA_MODEL} \u00b7 ChromaDB \u00b7 Local AI</p>',
        unsafe_allow_html=True,
    )
st.divider()

# Session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

doc_col = get_doc_collection(fresh=False)
mem_col = get_mem_collection()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<h2 style="color:{OPTUM_ORANGE}">\u2699\ufe0f Controls</h2>', unsafe_allow_html=True)

    # Documents
    st.subheader("\U0001f4da Document Library")
    stats = collection_stats(doc_col)
    c1, c2 = st.columns(2)
    c1.metric("Chunks", stats["count"])
    c2.metric("Files", len(stats["sources"]))
    if stats["sources"]:
        with st.expander("Indexed files"):
            for src in stats["sources"]:
                st.markdown(f"\U0001f4c4 `{src}`")

    st.divider()
    fresh_toggle = st.toggle("\U0001f5d1\ufe0f Replace store on next upload",
                             help="Wipes all existing vectors before indexing.")
    uploaded = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if uploaded and st.button("\U0001f4e5 Index Documents", type="primary"):
        if fresh_toggle:
            doc_col = get_doc_collection(fresh=True)
            st.info("Vector store cleared.")
        progress = st.progress(0)
        for i, f in enumerate(uploaded):
            progress.progress(i / len(uploaded), text=f"Indexing {f.name}...")
            try:
                r = ingest_pdf(f.read(), f.name, doc_col)
                st.success(f"\u2705 {r['filename']}: {r['pages']}p, {r['chunks']} chunks")
            except Exception as e:
                st.error(f"\u274c {f.name}: {e}")
        progress.progress(1.0, text="Done!")
        st.rerun()

    st.divider()

    # Memory
    st.subheader("\U0001f9e0 Conversation Memory")
    st.caption(f"{mem_col.count()} turns stored in ChromaDB")
    if st.button("\U0001f5d1\ufe0f Clear My Memory"):
        try:
            if mem_col.count() > 0:
                res = mem_col.get(limit=min(mem_col.count(), 1000), include=["metadatas"])
                ids_del = [res["ids"][i] for i, m in enumerate(res["metadatas"])
                           if m.get("session_id") == st.session_state.session_id]
                if ids_del:
                    mem_col.delete(ids=ids_del)
            st.session_state.messages = []
            st.success("Memory cleared.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.divider()

    # Ollama status
    st.subheader("\U0001f50c Ollama")
    try:
        requests.get("http://localhost:11434", timeout=2)
        models_r  = requests.get("http://localhost:11434/api/tags", timeout=2).json()
        available = [m["name"] for m in models_r.get("models", [])]
        if any(OLLAMA_MODEL in m for m in available):
            st.success(f"`{OLLAMA_MODEL}` ready \u2705")
        else:
            st.warning(f"Run: `ollama pull {OLLAMA_MODEL}`")
    except Exception:
        st.error("Ollama not reachable \u274c\n`ollama serve`")

    st.divider()
    st.caption("**Source routing:**")
    st.markdown("- \U0001f4ac **Chat** — greetings & general\n- \U0001f4c4 **PDF** — document content\n- \U0001f5c4\ufe0f **DB** — billing records\n- \U0001f500 **Both** — cross-source")

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if (msg["role"] == "assistant"
                and msg is st.session_state.messages[-1]
                and msg.get("suggestions")):
            st.markdown(f'<p style="color:{OPTUM_ORANGE};font-weight:600">\U0001f4a1 You might want to ask:</p>',
                        unsafe_allow_html=True)
            btn_cols = st.columns(len(msg["suggestions"]))
            for bi, (col, sug) in enumerate(zip(btn_cols, msg["suggestions"])):
                if col.button(sug, key=f"sug_last_{bi}"):
                    st.session_state.pending_input = sug
                    st.rerun()

# ── Input ─────────────────────────────────────────────────────────────────────
if st.session_state.pending_input:
    user_input = st.session_state.pending_input
    st.session_state.pending_input = None
else:
    user_input = st.chat_input("Ask about your documents, billing data, or anything...")

if user_input:
    save_turn(st.session_state.session_id, "user", user_input, mem_col)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        final_payload = None

        for step in run_agent(user_input, doc_col, mem_col, st.session_state.session_id):
            if step.get("final"):
                final_payload = step
                break
            with st.expander(step["step"], expanded=False):
                if step.get("detail"): st.markdown(step["detail"])
                if step.get("sql"):    st.code(step["sql"], language="sql")
                if step.get("chunks"):
                    for c in step["chunks"]:
                        sk = "rerank_score" if "rerank_score" in c else "score"
                        st.markdown(f"**\U0001f4c4 {c['filename']} — Page {c['page_num']}** (score: {c.get(sk, c['score'])})")
                        st.caption((c["text"][:250] + "...") if len(c["text"]) > 250 else c["text"])

        if final_payload:
            intent      = final_payload.get("intent","chat")
            explanation = final_payload.get("explanation","")
            chunks      = final_payload.get("chunks", [])
            result      = final_payload.get("result")
            suggestions = final_payload.get("suggestions", [])

            if final_payload.get("sql"):
                with st.expander("\U0001f5c4\ufe0f SQL Executed", expanded=False):
                    st.code(final_payload["sql"], language="sql")

            if intent == "chat":
                st.markdown(explanation)
            else:
                st.subheader("\U0001f4ac Answer")
                st.markdown(explanation)

            if chunks and intent in ("pdf","both"):
                st.divider()
                st.subheader("\U0001f4ce Sources")
                for i, c in enumerate(chunks, 1):
                    sk = "rerank_score" if "rerank_score" in c else "score"
                    with st.expander(f"{i}. {c['filename']} — Page {c['page_num']}  (relevance: {round(c.get(sk,c['score']),3)})"):
                        st.markdown(c["text"])

            if result and result.get("data") and intent in ("db","both"):
                st.divider()
                st.subheader("\U0001f4cb Database Results")
                df = pd.DataFrame(result["data"])
                m  = st.columns(4)
                m[0].metric("Rows", result["row_count"])
                if "billed" in df.columns: m[1].metric("Total Billed", f"${df['billed'].sum():,.2f}")
                if "administered" in df.columns: m[2].metric("Total Administered", f"${df['administered'].sum():,.2f}")
                if {"billed","administered"}.issubset(df.columns):
                    gap = df["billed"].sum() - df["administered"].sum()
                    m[3].metric("Billing Gap", f"${gap:,.2f}", delta=f"${gap:,.2f}", delta_color="inverse")
                col_cfg = {}
                for col in df.columns:
                    if col.lower() in ("billed","administered"):
                        col_cfg[col] = st.column_config.NumberColumn(col.replace("_"," ").title(), format="$%.2f")
                    elif "date" in col.lower():
                        col_cfg[col] = st.column_config.DateColumn(col.replace("_"," ").title())
                    else:
                        col_cfg[col] = st.column_config.TextColumn(col.replace("_"," ").title())
                st.dataframe(df, use_container_width=True, hide_index=True, column_config=col_cfg)
                num_cols = df.select_dtypes(include="number").columns.tolist()
                if num_cols and len(df) > 1:
                    chart_col = st.selectbox("\U0001f4c8 Visualise:", num_cols, key=f"chart_{len(st.session_state.messages)}")
                    label_col = next((c for c in df.columns if "name" in c.lower()), None)
                    st.bar_chart(df.set_index(label_col)[chart_col] if label_col else df[chart_col])

            if suggestions:
                st.divider()
                st.markdown(f'<p style="color:{OPTUM_ORANGE};font-weight:600">\U0001f4a1 You might want to ask:</p>', unsafe_allow_html=True)
                btn_cols = st.columns(len(suggestions))
                for bi, (col, sug) in enumerate(zip(btn_cols, suggestions)):
                    if col.button(sug, key=f"sug_{len(st.session_state.messages)}_{bi}"):
                        st.session_state.pending_input = sug
                        st.rerun()

            save_turn(st.session_state.session_id, "assistant", explanation, mem_col)
            st.session_state.messages.append({"role":"assistant","content":explanation,"suggestions":suggestions})