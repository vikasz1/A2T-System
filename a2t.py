from google import genai
import os
import json
import time
import sqlite3
import streamlit as st
import pandas as pd

# Configure Gemini - Load API key from file
api_key = ""
if not api_key:
    try:
        with open("api-key.txt", "r") as f:
            api_key = f.readline().strip()
    except FileNotFoundError:
        raise RuntimeError("API key not found in GEMINI_API_KEY env var or api-key.txt")

client = genai.Client(api_key=api_key)

# Initialize SQLite database with billing data
def init_database():
    """Initialize SQLite database with sample billing data."""
    conn = sqlite3.connect("billing.db")
    cursor = conn.cursor()
    
    # Drop existing table if it exists
    cursor.execute("DROP TABLE IF EXISTS invoices")
    
    # Create invoices table
    cursor.execute("""
        CREATE TABLE invoices (
            invoice_id INTEGER PRIMARY KEY,
            customer_name TEXT NOT NULL,
            billed REAL NOT NULL,
            administered REAL NOT NULL,
            service_date TEXT NOT NULL,
            status TEXT NOT NULL,
            description TEXT
        )
    """)
    
    # Insert sample data
    data = [
        (101, "Acme Corp", 10000.00, 8000.00, "2026-01-15", "active", "Quarterly billing"),
        (102, "Tech Solutions", 5000.00, 5000.00, "2026-01-20", "active", "Monthly service"),
        (103, "Global Industries", 7500.00, 4000.00, "2026-01-25", "pending", "Service adjustment"),
        (104, "StartUp Inc", 2500.00, 3500.00, "2026-02-01", "active", "Under billing"),
        (105, "Mega Corporation", 15000.00, 12000.00, "2026-02-05", "active", "Full service"),
        (106, "Cloud Services Ltd", 8000.00, 8000.00, "2026-02-10", "active", "Cloud hosting"),
    ]
    
    cursor.executemany(
        "INSERT INTO invoices VALUES (?, ?, ?, ?, ?, ?, ?)",
        data
    )
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_database()

def get_table_schema():
    """Get schema information for the invoices table."""
    conn = sqlite3.connect("billing.db")
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(invoices)")
    schema = cursor.fetchall()
    conn.close()
    return schema

# Build schema description for the LLM
SCHEMA_INFO = """
Database Table: invoices
Columns:
- invoice_id (INTEGER, PRIMARY KEY): Unique invoice identifier
- customer_name (TEXT): Name of the customer
- billed (REAL): Amount billed to customer
- administered (REAL): Amount actually administered/provided
- service_date (TEXT): Date of service
- status (TEXT): Invoice status (active, pending, etc.)
- description (TEXT): Description of the service

Common queries:
- Find invoices where billed > administered (discrepancies)
- Find invoices with pending status
- Get all invoices for a specific customer
- Find high-value invoices (billed > 10000)
"""



# Tools
# -----------------------------

def run_sql_query(query: str):
    """Execute SQL query on the billing database and return results."""
    try:
        conn = sqlite3.connect("billing.db")
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {"success": True, "message": "Query executed but returned no results", "data": []}
        
        # Convert rows to list of dictionaries
        results = [dict(row) for row in rows]
        return {"success": True, "data": results, "row_count": len(results)}
    except Exception as e:
        return {"success": False, "error": str(e)}


def read_file(filename):
    """Read and return file contents."""
    try:
        with open(filename, "r") as f:
            return f.read()
    except:
        return "file not found"


# Tool registry
tools = {
    "run_sql_query": run_sql_query,
    "read_file": read_file
}

# Helper function for API calls with retry logic
def call_model_with_retry(prompt, max_retries=3):
    """Call Gemini API with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )
        except Exception as e:
            # Try to extract retry delay from error message
            error_msg = str(e)
            wait_time = 2 ** attempt  # Default exponential backoff
            
            # Check for quota/rate limit errors
            if any(keyword in error_msg for keyword in ["quota", "ResourceExhausted", "429"]):
                # Try to find retry delay in error message
                if "Please retry in" in error_msg:
                    try:
                        wait_time = float(error_msg.split("Please retry in ")[-1].split('s')[0])
                    except (ValueError, IndexError):
                        pass
                
                if attempt < max_retries - 1:
                    print(f"\n[Quota limit hit] Waiting {wait_time:.1f} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print("\n[ERROR] Quota exceeded and max retries reached. Please upgrade to a paid plan or try again later.")
                    raise
            else:
                # For other errors, retry with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"\n[Error] {type(e).__name__}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise

# -----------------------------
# Agent prompt
# -----------------------------

SYSTEM_PROMPT = """
You are an AI agent that can query a billing database and read files.

Available tools:
1. run_sql_query(query) - Execute SQL queries on the billing database

Database Schema:
""" + SCHEMA_INFO + """

Instructions:
- When the user asks about billing data, form a SQL query and call run_sql_query
- Always write valid SQL queries based on the schema provided
- Return results in JSON format if you need to call a tool:

{
"tool": "run_sql_query",
"parameters": {"query": "SELECT * FROM invoices WHERE ..."}
}

- If no tool is required, respond naturally with the answer
"""

# -----------------------------
# Agent Loop
# -----------------------------

def agent(user_input):

    prompt = SYSTEM_PROMPT + "\nUser: " + user_input

    response = call_model_with_retry(prompt)

    text = response.text

    # Check if tool call
    try:
        data = json.loads(text)

        tool_name = data["tool"]
        params = data["parameters"]

        tool = tools[tool_name]

        result = tool(**params)

        # Send result back to LLM for explanation
        final_prompt = f"""
User Question: {user_input}

Tool Result:
{json.dumps(result, indent=2)}

Provide a clear, professional explanation of the results to the user.
"""

        final_response = call_model_with_retry(final_prompt)

        return {
            "success": True,
            "result": result,
            "explanation": final_response.text
        }

    except:
        return {
            "success": True,
            "explanation": text
        }


# -----------------------------
# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Page config
st.set_page_config(
    page_title="A2T Billing Agent",
    page_icon="💰",
    layout="wide"
)

st.title("💰 A2T Billing Agent")
st.write("Query your billing database with natural language. Ask about invoices, discrepancies, and more.")

# Sidebar with schema info
with st.sidebar:
    st.header("Database Schema")
    st.info("""
    **invoices table** contains:
    - invoice_id: Unique identifier
    - customer_name: Customer name
    - billed: Amount billed
    - administered: Amount provided
    - service_date: Date of service
    - status: Invoice status
    - description: Service description
    """)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask something about your billing data:")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.write(user_input)
    
    # Call agent and display response
    with st.chat_message("assistant"):
        with st.spinner("Processing your query..."):
            response = agent(user_input)
        
        # Display explanation
        st.markdown(response["explanation"])
        
        # Display data if available
        if response.get("success") and "result" in response:
            result = response["result"]
            
            if result.get("success") and result.get("data"):
                st.divider()
                st.subheader("Query Results")
                
                # Convert to DataFrame for nice display
                df = pd.DataFrame(result["data"])
                
                # Display metrics if applicable
                if len(result["data"]) > 0:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Records Found", result.get("row_count", 0))
                    with col2:
                        # Calculate total billed if column exists
                        if "billed" in df.columns:
                            total_billed = df["billed"].sum()
                            st.metric("Total Billed", f"${total_billed:,.2f}")
                
                # Display the data table
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "billed": st.column_config.NumberColumn("Billed", format="$%.2f"),
                        "administered": st.column_config.NumberColumn("Administered", format="$%.2f"),
                    }
                )
            elif result.get("message"):
                st.info(result["message"])
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response["explanation"]})