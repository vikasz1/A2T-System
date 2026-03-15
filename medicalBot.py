from google import genai
import json
import time
import sqlite3
import streamlit as st
import pandas as pd

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="A2T Billing Agent", page_icon="💰", layout="wide")

# ─── API Key ──────────────────────────────────────────────────────────────────
api_key = ""
if not api_key:
    try:
        with open("api-key.txt") as f:
            api_key = f.readline().strip()
    except FileNotFoundError:
        raise RuntimeError("API key not found.")

client = genai.Client(api_key=api_key)
DB_PATH = "billing.db"

# ─── Token budget constants ───────────────────────────────────────────────────
# Each controls how much data flows into the LLM prompts.
MAX_SAMPLE_ROWS   = 1   # sample rows sent per table (billing only)
MAX_OTHER_TABLES  = 3   # cap on non-billing tables included in schema context
MAX_EXPLAIN_ROWS  = 5   # result rows included in the explanation prompt

# ─── Sample DB init ───────────────────────────────────────────────────────────
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS billing")
    cur.execute("""
        CREATE TABLE billing (
            invoice_id    INTEGER PRIMARY KEY,
            customer_name TEXT    NOT NULL,
            billed        REAL    NOT NULL,
            administered  REAL    NOT NULL,
            service_date  TEXT    NOT NULL,
            status        TEXT    NOT NULL,
            description   TEXT
        )
    """)
    cur.executemany("INSERT INTO billing VALUES (?,?,?,?,?,?,?)", [
        (101, "Acme Corp",         10000,  8000, "2026-01-15", "active",  "Quarterly billing"),
        (102, "Tech Solutions",     5000,  5000, "2026-01-20", "active",  "Monthly service"),
        (103, "Global Industries",  7500,  4000, "2026-01-25", "pending", "Service adjustment"),
        (104, "StartUp Inc",        2500,  3500, "2026-02-01", "active",  "Under billing"),
        (105, "Mega Corporation",  15000, 12000, "2026-02-05", "active",  "Full service"),
        (106, "Cloud Services Ltd", 8000,  8000, "2026-02-10", "active",  "Cloud hosting"),
    ])
    conn.commit()
    conn.close()

init_database()

# ─── Schema discovery ─────────────────────────────────────────────────────────
def discover_schema() -> dict:
    """
    Always returns the 'billing' table first.
    Other tables are capped at MAX_OTHER_TABLES.
    Only billing gets sample rows (to keep token usage low).
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    all_tables = [r["name"] for r in cur.fetchall()]

    # billing always first
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

        # Sample rows only for billing table
        sample_rows = []
        if table == "billing":
            cur.execute(f"SELECT * FROM {table} LIMIT {MAX_SAMPLE_ROWS}")
            sample_rows = [dict(r) for r in cur.fetchall()]

        schema[table] = {"columns": cols, "sample_rows": sample_rows, "row_count": row_count}

    conn.close()
    return schema


def schema_to_compact_text(schema: dict) -> str:
    """
    Ultra-compact schema string to minimise tokens.
    Format: tablename(col TYPE [PK][NN], ...) — N rows
    """
    lines = []
    for table, info in schema.items():
        col_parts = []
        for c in info["columns"]:
            flags = ("PK " if c["pk"] else "") + ("NN" if c["notnull"] else "")
            col_parts.append(
                f"{c['name']} {c['type']}{(' ' + flags.strip()) if flags.strip() else ''}"
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

# ─── LLM caller ───────────────────────────────────────────────────────────────
def call_llm(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            return resp.text
        except Exception as e:
            err  = str(e)
            wait = 2 ** attempt
            if any(k in err for k in ["quota", "ResourceExhausted", "429"]):
                if "Please retry in" in err:
                    try: wait = float(err.split("Please retry in ")[-1].split("s")[0])
                    except: pass
                if attempt < max_retries - 1:
                    time.sleep(wait)
                    continue
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                raise

# ─── Token-lean prompts ───────────────────────────────────────────────────────
def build_sql_prompt(schema_text: str, question: str) -> str:
    return (
        f"SQLite schema:\n{schema_text}\n\n"
        f"Question: \"{question}\"\n"
        "Prefer the 'billing' table unless another table is clearly needed.\n"
        "Reply ONLY with valid JSON (no markdown):\n"
        '{"reasoning":"<one sentence>","sql":"<query>"}'
    )

def build_fix_prompt(schema_text: str, bad_sql: str, error: str) -> str:
    return (
        f"SQLite schema:\n{schema_text}\n\n"
        f"Failed query:\n{bad_sql}\nError: {error}\n\n"
        'Return corrected JSON only: {"sql":"<fixed query>"}'
    )

def build_explain_prompt(question: str, sql: str, rows: list, row_count: int) -> str:
    preview   = rows[:MAX_EXPLAIN_ROWS]
    truncated = f" (first {len(preview)} of {row_count})" if row_count > MAX_EXPLAIN_ROWS else ""
    return (
        f"Q: {question}\n"
        f"SQL: {sql}\n"
        f"Results{truncated}: {json.dumps(preview)}\n\n"
        "Write a 3-5 bullet business summary. "
        "Highlight gaps, anomalies, totals. No raw data repetition."
    )

# ─── Agent generator ──────────────────────────────────────────────────────────
def run_agent(user_question: str):
    """Yields step dicts; last one has final=True."""

    # Step 1 — Schema discovery
    yield {"step": "🔍 Step 1: Discovering Database Schema",
           "detail": "Scanning all tables — **billing** table always loaded first."}

    schema      = discover_schema()
    schema_text = schema_to_compact_text(schema)
    table_names = list(schema.keys())

    yield {"step": "✅ Schema Discovered",
           "detail": (
               f"Found **{len(table_names)}** table(s): `{'`, `'.join(table_names)}`\n\n"
               f"Token-efficient compact form sent to LLM:\n```\n{schema_text}\n```"
           ),
           "schema": schema}

    # Step 2 — Plan query
    yield {"step": "🧠 Step 2: Planning SQL Query",
           "detail": "Sending compact schema + question to LLM (minimal tokens)…"}

    sql_prompt = build_sql_prompt(schema_text, user_question)
    raw = call_llm(sql_prompt).strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    try:
        plan      = json.loads(raw)
        sql_query = plan["sql"]
        reasoning = plan.get("reasoning", "")
    except Exception:
        sql_query = next(
            (l.strip() for l in raw.splitlines() if l.strip().upper().startswith("SELECT")),
            None,
        )
        reasoning = raw

    yield {"step": "💡 Step 3: Query Plan",
           "detail": f"**LLM Reasoning:** {reasoning}",
           "sql": sql_query}

    if not sql_query:
        yield {"final": True, "step": "❌ No SQL Generated",
               "explanation": "I couldn't form a SQL query. Please rephrase.", "result": None}
        return

    # Step 3 — Execute
    yield {"step": "⚙️ Step 4: Executing SQL Query",
           "detail": f"Running against `{DB_PATH}`…", "sql": sql_query}

    result = run_sql_query(sql_query)

    # Step 4b — Self-correct
    if not result["success"]:
        yield {"step": "🔧 Step 4b: Self-Correcting Query",
               "detail": f"Error: `{result.get('error')}` — asking LLM for a fix…"}

        fix_raw = call_llm(
            build_fix_prompt(schema_text, sql_query, result.get("error", ""))
        ).strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            sql_query = json.loads(fix_raw)["sql"]
            yield {"step": "🔄 Step 4c: Retrying Corrected Query",
                   "detail": "Running corrected query…", "sql": sql_query}
            result = run_sql_query(sql_query)
        except Exception:
            pass

    if not result["success"]:
        yield {"final": True, "step": "❌ Query Failed",
               "explanation": f"Query failed: {result.get('error')}", "result": None}
        return

    yield {"step": f"✅ Step 5: Retrieved {result['row_count']} Row(s)",
           "detail": "Generating lean business summary (only top rows sent to LLM)…"}

    # Step 5 — Explain
    explanation = call_llm(
        build_explain_prompt(user_question, sql_query,
                             result.get("data", []), result["row_count"])
    )

    yield {"step": "📝 Step 6: Summary Generated", "detail": "Rendering results…"}
    yield {"final": True, "explanation": explanation, "result": result, "sql": sql_query}


# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("💰 A2T Billing Agent")
st.caption(
    "Natural-language queries over your billing database — "
    "schema-aware, token-efficient, step-by-step."
)

with st.sidebar:
    st.header("ℹ️ How It Works")
    st.markdown("""
1. **Schema Discovery** — `billing` scanned first, others capped  
2. **Compact Prompt** — col names/types only + 1 sample row  
3. **SQL Generation** — LLM writes a targeted query  
4. **Self-Correction** — auto-retry on SQL errors  
5. **Lean Explanation** — only first few rows sent back to LLM  
""")
    st.divider()
    st.subheader("⚙️ Token Budget")
    st.metric("Sample rows in prompt",    MAX_SAMPLE_ROWS)
    st.metric("Max extra tables",         MAX_OTHER_TABLES)
    st.metric("Rows sent for explanation", MAX_EXPLAIN_ROWS)
    st.divider()
    if st.button("🔄 Reset Sample Data"):
        init_database()
        st.success("Database reset.")

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

        # Stream steps
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
                        st.dataframe(
                            pd.DataFrame(tinfo["columns"]),
                            use_container_width=True, hide_index=True
                        )
                        if tinfo["sample_rows"]:
                            st.caption("Sample row:")
                            st.dataframe(
                                pd.DataFrame(tinfo["sample_rows"]),
                                use_container_width=True, hide_index=True
                            )

                if step.get("sql"):
                    st.code(step["sql"], language="sql")

        # Final result
        if final_payload:
            if final_payload.get("sql"):
                with st.expander("🗄️ SQL Query Executed", expanded=True):
                    st.code(final_payload["sql"], language="sql")

            st.subheader("📊 Business Summary")
            st.markdown(final_payload.get("explanation", "No explanation."))

            result = final_payload.get("result")
            if result and result.get("data"):
                st.divider()
                st.subheader("📋 Query Results")

                df = pd.DataFrame(result["data"])

                # Metrics row
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

                # Column formatting
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

                # Bar chart
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