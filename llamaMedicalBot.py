import requests
import json
import sqlite3
import streamlit as st
import pandas as pd

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="A2T Billing Agent", page_icon="💰", layout="wide")

# ─── Ollama config ────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

DB_PATH = "billing.db"

# ─── Schema discovery constants ───────────────────────────────────────────────
MAX_SAMPLE_ROWS  = 1   # sample rows sent for billing table
MAX_OTHER_TABLES = 3   # cap on non-billing tables included in schema context
MAX_EXPLAIN_ROWS = 5   # result rows included in the explanation prompt

# ─── Sample DB init ───────────────────────────────────────────────────────────
# ─── Schema discovery ─────────────────────────────────────────────────────────
def discover_schema() -> dict:
    """
    Always returns 'billing' table first.
    Other tables capped at MAX_OTHER_TABLES.
    Only billing gets a sample row.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    all_tables = [r["name"] for r in cur.fetchall()]

    priority   = [t for t in all_tables if t == "billing"]
    others     = [t for t in all_tables if t != "billing"][:MAX_OTHER_TABLES]
    table_list = priority + others

    schema = {}
    for table in table_list:
        cur.execute(f"PRAGMA table_info({table})")
        cols = [
            {
                "name":    r["name"],
                "type":    r["type"],
                "notnull": bool(r["notnull"]),
                "pk":      bool(r["pk"]),
            }
            for r in cur.fetchall()
        ]
        cur.execute(f"SELECT COUNT(*) AS cnt FROM {table}")
        row_count = cur.fetchone()["cnt"]

        sample_rows = []
        if table == "billing":
            cur.execute(f"SELECT * FROM {table} LIMIT {MAX_SAMPLE_ROWS}")
            sample_rows = [dict(r) for r in cur.fetchall()]

        schema[table] = {"columns": cols, "sample_rows": sample_rows, "row_count": row_count}

    conn.close()
    return schema


def schema_to_compact_text(schema: dict) -> str:
    """Single-line-per-table compact schema string."""
    lines = []
    for table, info in schema.items():
        col_parts = []
        for c in info["columns"]:
            flags = ("PK " if c["pk"] else "") + ("NN" if c["notnull"] else "")
            # Wrap column names in backticks so the LLM always quotes them correctly
            quoted_name = f"`{c['name']}`"
            col_parts.append(
                f"{quoted_name} {c['type']}{(' ' + flags.strip()) if flags.strip() else ''}"
            )
        lines.append(f"{table}({', '.join(col_parts)}) — {info['row_count']} rows")
        if info["sample_rows"]:
            lines.append(f"  e.g. {info['sample_rows'][0]}")
    return "\n".join(lines)

# ─── SQL execution ────────────────────────────────────────────────────────────
def run_sql_query(query: str) -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return {"success": True, "message": "No rows returned.", "data": [], "row_count": 0}
        data = [dict(r) for r in rows]
        return {"success": True, "data": data, "row_count": len(data)}
    except Exception as e:
        return {"success": False, "error": str(e), "data": []}

# ─── LLM caller — Ollama (local Llama 3.1 8B) ────────────────────────────────
def call_llm(prompt: str, max_retries: int = 3) -> str:
    """
    Calls the local Ollama server running llama3.1:8b.
    Uses stream=False so we get a single JSON response.
    Retries on connection errors (e.g. model still loading).
    """
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,          # get full response at once
        "options": {
            "temperature": 0.1,   # low temp → deterministic SQL output
            "num_predict": 512,   # cap output tokens
        },
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["response"]
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                import time; time.sleep(2 ** attempt)
            else:
                raise RuntimeError(
                    "Cannot connect to Ollama. "
                    "Make sure it is running: `ollama serve` "
                    f"and the model is pulled: `ollama pull {OLLAMA_MODEL}`"
                )
        except Exception as e:
            if attempt < max_retries - 1:
                import time; time.sleep(2 ** attempt)
            else:
                raise

# ─── Prompts ──────────────────────────────────────────────────────────────────

def build_intent_prompt(question: str) -> str:
    """Step 0: classify whether the question needs a DB query or is just chat."""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You classify user messages for a billing assistant. Output ONLY raw JSON, no markdown.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Message: "{question}"

Decide:
- "needs_data": true if the message asks about invoices, billing, amounts, customers, discrepancies, status, dates, totals, or any database information.
- "needs_data": false if it is a greeting, small talk, thanks, or anything unrelated to billing data.
- "reply": if needs_data is false, write a short friendly reply. Otherwise leave blank.
- "reasoning": one sentence explaining your decision.

Output exactly: {{"needs_data": true/false, "reasoning": "...", "reply": "..."}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{"""


def build_sql_prompt(schema_text: str, question: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a SQLite expert. Output ONLY a raw JSON object — no markdown, no explanation, no code fences.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Schema:
{schema_text}

Question: {question}

Rules:
- Prefer the 'billing' table unless another table is clearly needed.
- Write a single valid SQLite SELECT query. Do NOT use SELECT * — only select columns needed to answer the question.
- ALWAYS wrap every column name and table name in backticks (e.g. `Column Name`, `table_name`) — even single-word names.
- Output exactly: {{"reasoning":"one sentence explaining which columns and filters you chose","sql":"SELECT ..."}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{"""


def build_fix_prompt(schema_text: str, bad_sql: str, error: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a SQLite expert. Output ONLY raw JSON — no markdown, no code fences.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Schema:
{schema_text}

This query failed:
{bad_sql}
Error: {error}

Return corrected JSON only: {{"sql":"SELECT ..."}}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{"""


def build_explain_prompt(question: str, sql: str, rows: list, row_count: int) -> str:
    preview   = rows[:MAX_EXPLAIN_ROWS]
    truncated = f" (first {len(preview)} of {row_count})" if row_count > MAX_EXPLAIN_ROWS else ""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a business analyst. Write a concise summary in plain English using bullet points.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {question}
SQL used: {sql}
Results{truncated}: {json.dumps(preview)}

Write 3-5 bullet points. Highlight totals, gaps, anomalies. Do NOT repeat raw data.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def parse_json(raw: str) -> dict:
    """
    Robustly parse JSON from Llama output.
    The prompt ends with '{' so Llama continues from there — we prepend it back.
    Also strips markdown fences and truncates to first complete JSON object.
    """
    text = raw.strip()
    if not text.startswith("{"):
        text = "{" + text
    text = text.lstrip("```json").lstrip("```").strip()

    depth, end = 0, 0
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end:
        text = text[:end]

    return json.loads(text)


# ─── Agent generator ──────────────────────────────────────────────────────────
def run_agent(user_question: str):
    """
    Yields step dicts for the UI; last dict has final=True.

    Flow:
      Step 1 — Intent check  (always runs, very cheap — no schema sent)
        → if chat:  reply naturally, STOP. DB is never touched.
        → if data:  continue below
      Step 2 — Schema discovery
      Step 3 — SQL planning with explicit reasoning
      Step 4 — SQL execution  (+ 4b self-correct on failure)
      Step 5 — Business explanation
    """

    # ── Step 1: Intent classification ────────────────────────────────────────
    yield {
        "step":   "🤔 Step 1: Understanding Your Request",
        "detail": "Checking whether this question needs a database query or is just conversation…",
    }

    intent_raw = call_llm(build_intent_prompt(user_question))
    try:
        intent = parse_json(intent_raw)
    except Exception:
        intent = {"needs_data": True, "reasoning": "Could not parse intent — defaulting to data query.", "reply": ""}

    needs_data = intent.get("needs_data", True)
    reasoning  = intent.get("reasoning", "")
    chat_reply = intent.get("reply", "").strip()

    if not needs_data:
        # Pure conversation — skip DB entirely
        yield {
            "step":   "💬 Decision: No Database Query Needed",
            "detail": (
                f"**Reasoning:** {reasoning}\n\n"
                "✅ Responding conversationally — no schema scan, no SQL executed."
            ),
        }
        yield {
            "final":       True,
            "explanation": chat_reply or "Hello! How can I help you with your billing data today?",
            "result":      None,
            "skipped_db":  True,
        }
        return

    # Data question — announce why we are querying
    yield {
        "step":   "✅ Decision: Database Query Required",
        "detail": (
            f"**Reasoning:** {reasoning}\n\n"
            "➡️ Proceeding to discover schema and write a targeted SQL query."
        ),
    }

    # ── Step 2: Schema discovery ──────────────────────────────────────────────
    yield {
        "step":   "🔍 Step 2: Discovering Database Schema",
        "detail": "Scanning all tables — **billing** always loaded first.",
    }

    schema      = discover_schema()
    schema_text = schema_to_compact_text(schema)
    table_names = list(schema.keys())

    yield {
        "step":   "✅ Schema Discovered",
        "detail": (
            f"Found **{len(table_names)}** table(s): `" + "`, `".join(table_names) + "`\n\n"
            f"Compact schema passed to LLM:\n```\n{schema_text}\n```"
        ),
        "schema": schema,
    }

    # ── Step 3: SQL planning ──────────────────────────────────────────────────
    yield {
        "step":   "🧠 Step 3: Planning SQL Query",
        "detail": f"Asking **{OLLAMA_MODEL}** to reason about the schema and write a targeted query…",
    }

    raw_sql = call_llm(build_sql_prompt(schema_text, user_question))

    try:
        plan       = parse_json(raw_sql)
        sql_query  = plan["sql"]
        sql_reason = plan.get("reasoning", "")
    except Exception:
        sql_query  = next(
            (l.strip() for l in raw_sql.splitlines() if "SELECT" in l.upper()), None
        )
        sql_reason = raw_sql

    yield {
        "step":   "💡 Query Plan",
        "detail": f"**Why this query:** {sql_reason}",
        "sql":    sql_query,
    }

    if not sql_query:
        yield {
            "final":       True,
            "explanation": "I understood your question needs data, but couldn't form a valid SQL query. Please rephrase.",
            "result":      None,
        }
        return

    # ── Step 4: Execute ───────────────────────────────────────────────────────
    yield {
        "step":   "⚙️ Step 4: Executing SQL Query",
        "detail": f"Running against `{DB_PATH}`…",
        "sql":    sql_query,
    }

    result = run_sql_query(sql_query)

    # ── Step 4b: Self-correct on failure ─────────────────────────────────────
    if not result["success"]:
        yield {
            "step":   "🔧 Step 4b: Self-Correcting Query",
            "detail": f"**Error:** `{result.get('error')}` — asking LLM to diagnose and fix…",
        }
        try:
            fix_raw   = call_llm(build_fix_prompt(schema_text, sql_query, result.get("error", "")))
            sql_query = parse_json(fix_raw)["sql"]
            yield {
                "step":   "🔄 Step 4c: Retrying Corrected Query",
                "detail": "Running the corrected query…",
                "sql":    sql_query,
            }
            result = run_sql_query(sql_query)
        except Exception as fix_err:
            yield {"final": True, "explanation": f"Self-correction failed: {fix_err}", "result": None}
            return

    if not result["success"]:
        yield {"final": True, "explanation": f"Query failed: {result.get('error')}", "result": None}
        return

    yield {
        "step":   f"✅ Step 5: Retrieved {result['row_count']} Row(s)",
        "detail": "Generating business summary from results…",
    }

    # ── Step 5: Explain ───────────────────────────────────────────────────────
    explanation = call_llm(
        build_explain_prompt(user_question, sql_query,
                             result.get("data", []), result["row_count"])
    )

    yield {"step": "📝 Step 6: Summary Generated", "detail": "Rendering final results…"}
    yield {"final": True, "explanation": explanation, "result": result, "sql": sql_query}


# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("💰 A2T Billing Agent")
st.caption(
    f"Powered by **{OLLAMA_MODEL}** running locally via Ollama — "
    "no API keys, no token limits."
)

with st.sidebar:
    st.header("ℹ️ How It Works")
    st.markdown("""
1. **Schema Discovery** — `billing` scanned first, others capped  
2. **Compact Prompt** — col names/types + 1 sample row only  
3. **SQL Generation** — local Llama 3.1 writes a targeted query  
4. **Self-Correction** — auto-retry on SQL errors  
5. **Explanation** — Llama summarises results in plain English  
""")
    st.divider()

    st.subheader("🦙 Ollama Setup")
    st.code(f"ollama pull {OLLAMA_MODEL}", language="bash")
    st.code("ollama serve", language="bash")
    st.caption(f"Endpoint: `{OLLAMA_URL}`")

    # Live connection check
    st.divider()
    st.subheader("🔌 Connection Status")
    try:
        ping = requests.get("http://localhost:11434", timeout=2)
        st.success("Ollama is running ✅")
        # Check if model is available
        models_resp = requests.get("http://localhost:11434/api/tags", timeout=2).json()
        available   = [m["name"] for m in models_resp.get("models", [])]
        if any(OLLAMA_MODEL in m for m in available):
            st.success(f"`{OLLAMA_MODEL}` is ready ✅")
        else:
            st.warning(f"`{OLLAMA_MODEL}` not found.\nRun: `ollama pull {OLLAMA_MODEL}`")
            if available:
                st.caption("Available models: " + ", ".join(available))
    except Exception:
        st.error("Ollama not reachable ❌\nRun: `ollama serve`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about your billing data…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        final_payload = None

        for step in run_agent(user_input):
            if step.get("final"):
                final_payload = step
                break

            with st.expander(step["step"], expanded=False):
                if step.get("detail"):
                    st.markdown(step["detail"])

                if "schema" in step:
                    for tname, tinfo in step["schema"].items():
                        badge = " ⭐ *primary*" if tname == "billing" else ""
                        st.markdown(f"**`{tname}`**{badge} — {tinfo['row_count']} rows")
                        st.dataframe(pd.DataFrame(tinfo["columns"]),
                                     use_container_width=True, hide_index=True)
                        if tinfo["sample_rows"]:
                            st.caption("Sample row:")
                            st.dataframe(pd.DataFrame(tinfo["sample_rows"]),
                                         use_container_width=True, hide_index=True)

                if step.get("sql"):
                    st.code(step["sql"], language="sql")

        if final_payload:
            if final_payload.get("sql"):
                with st.expander("🗄️ SQL Query Executed", expanded=True):
                    st.code(final_payload["sql"], language="sql")

            if final_payload.get("skipped_db"):
                st.markdown(final_payload.get("explanation", ""))
            else:
                st.subheader("📊 Business Summary")
                st.markdown(final_payload.get("explanation", "No explanation."))

            result = final_payload.get("result")
            if result and result.get("data"):
                st.divider()
                st.subheader("📋 Query Results")

                df = pd.DataFrame(result["data"])

                m = st.columns(4)
                m[0].metric("Rows Returned", result["row_count"])
                if "billed" in df.columns:
                    m[1].metric("Total Billed", f"${df['billed'].sum():,.2f}")
                if "administered" in df.columns:
                    m[2].metric("Total Administered", f"${df['administered'].sum():,.2f}")
                if {"billed", "administered"}.issubset(df.columns):
                    gap = df["billed"].sum() - df["administered"].sum()
                    m[3].metric("Billing Gap", f"${gap:,.2f}",
                                delta=f"${gap:,.2f}", delta_color="inverse")

                col_cfg = {}
                for col in df.columns:
                    if col in ("billed", "administered"):
                        col_cfg[col] = st.column_config.NumberColumn(
                            col.replace("_", " ").title(), format="$%.2f")
                    elif "date" in col.lower():
                        col_cfg[col] = st.column_config.DateColumn(
                            col.replace("_", " ").title())
                    else:
                        col_cfg[col] = st.column_config.TextColumn(
                            col.replace("_", " ").title())

                st.dataframe(df, use_container_width=True,
                             hide_index=True, column_config=col_cfg)

                num_cols = df.select_dtypes(include="number").columns.tolist()
                if num_cols and len(df) > 1:
                    chart_col = st.selectbox(
                        "📈 Visualise:", num_cols,
                        key=f"chart_{len(st.session_state.messages)}")
                    label_col = next(
                        (c for c in df.columns if "name" in c.lower()), None)
                    chart_data = (
                        df.set_index(label_col)[chart_col] if label_col else df[chart_col]
                    )
                    st.bar_chart(chart_data)

            elif result and result.get("message"):
                st.info(result["message"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_payload.get("explanation", "")
            })