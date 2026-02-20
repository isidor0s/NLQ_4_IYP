"""
IYP Assistant Web Application

A Streamlit-based web interface for querying the Internet Yellow Pages (IYP)
Neo4j knowledge graph using natural language.

Features:
    - Interactive chat interface with conversation history
    - Real-time query execution with reasoning step display
    - Automatic retry with LIMIT on query timeout
    - Connection diagnostics for troubleshooting
    - Export query results as JSON

Usage:
    streamlit run app.py
"""

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import json
import functools
import threading
import time
import socket
import urllib.request
import os
from typing import Optional
from dataclasses import dataclass


# =============================================================================
# PAGE CONFIG & CONSTANTS
# =============================================================================

st.set_page_config(
    page_title="NLQ4IYP Assistant",
    page_icon="🌐",
    layout="wide"
)

APP_TITLE = "NLQ4IYP: Natural Language Queries for IYP Graph"
APP_TAGLINE = "Ask questions about global Internet infrastructure and get answers from the Internet Yellow Pages Graph."
WARNING_TEXT = "Expand reasoning steps to see how the agent reached its answer and verify the queries."

SAMPLE_QUESTIONS = [
    "Find informations for AS 6799.",
    "Which IXPs are located in Greece?",
    "List the ASNs registered as Vodafone (AS3329) customers.",
    "List facilities in Greece",
]

IYP_CONSOLE_URL = "https://iyp.iijlab.net/iyp/browser/?dbms=iyp-bolt.iijlab.net:443"

# Neo4j connection defaults
DEFAULT_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://iyp-bolt.ihr.live:7687")
DEFAULT_NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
DEFAULT_NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Tool result truncation
MAX_TOOL_RESULT_ITEMS = 50


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are a helpful Neo4j Cypher query assistant with access to Neo4j database tools. Your job is to translate user questions into Cypher queries and explain the results.

### CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE:

- You MUST use the read_neo4j_cypher tool to execute queries for EVERY question
- ALWAYS execute at least one Cypher query to get real data from the database
- Generate Cypher queries based ONLY on the provided schema below
- If a query returns insufficient data, refine it and execute another query
- Base your final answer ONLY on the actual query results you receive

### STRATEGIC QUERY RULES (Apply These to the Schema Below)

**Rule 1: Empty Results Require Investigation**
- If your query returns `[]`, DO NOT conclude the data doesn't exist
- Systematically try these alternatives IN ORDER:
  a) Try alternative node labels (check schema for similar nodes)
  b) Swap relationship direction: `-[:REL]->` becomes `<-[:REL]-`
  c) Look for multi-hop paths through intermediate nodes
  d) Broaden your fuzzy matching (use shorter CONTAINS strings)

**Rule 2: Multi-Hop Relationship Discovery**
- Direct relationships often don't exist - look for intermediate nodes
- Pattern: Instead of `(A)-[direct]-(C)`, try `(A)--(intermediate)--(C)`
- Common intermediaries: Ranking, Country, Organization, OpaqueID
- Check the schema's "3. Schema:" section for valid connection paths

**Rule 3: LIMIT Discipline**
- NEVER add `LIMIT` unless user explicitly asks for "top N", "first N"
- For "top N" ranking queries: use `<= N` not `< N` (inclusive comparison)
- If a query TIMES OUT, retry with `LIMIT 100` added
- IMPORTANT: For queries that might return thousands of rows (e.g., all prefixes for a country), use LIMIT only after checking the total count first

**Rule 4: Relationship Properties Matter**
- Many relationships have discriminating properties that filter results to specific data sources or contexts. ALWAYS inspect relationship properties in the schema before writing your query.
- ALWAYS check if your target relationship has these properties in the schema
- These properties filter results to specific data sources or contexts
- Inspect schema carefully to understand what property values are valid

**Rule 5: Return Format Matching**
- Default: `RETURN n` (full node), not `RETURN n.property`
- ONLY return specific properties if user explicitly asks for that field
- Check similar examples in the schema for expected return format

**Rule 6: Smart String Matching**
- For names, descriptions, or text fields: prefer `toLower(property) CONTAINS 'value'` for flexibility
- For identifiers, codes, or known exact values (IPs, ASNs, country codes): exact match is fine
- When uncertain about exact spelling/format, use fuzzy matching
- Apply fuzzy matching to BOTH node searches AND relationship property searches when appropriate

**Rule 7: Bidirectional Relationship Exploration**
- Don't assume relationship direction from question wording
- If first direction fails, immediately try the opposite
- Use undirected `-[r]-` when uncertain, then inspect results

**Rule 8: Schema Cross-Reference Before Finalizing**
- Before executing complex queries, verify each hop exists in "3. Schema:" section
- Ensure relationship types actually connect those node types
- Don't invent relationships - use only what's documented

**Rule 9: Efficient Query Patterns**
- Start traversal from the more constrained node (smaller set)
- Use `WITH` clauses to filter early in the query pipeline
- For aggregations, use `COUNT()` instead of `COLLECT()` when you only need counts
- Pattern: `MATCH (small:Node {{property: 'specific'}})-[:REL]-(large:Node)` is faster than the reverse


{schema}
"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@dataclass
class TruncationConfig:
    """Configuration for output truncation."""
    max_lines: int = 20
    max_chars: int = 2000


def extract_text_content(content) -> str:
    """Extract text from various LLM response formats."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and part.get('type') == 'text':
                text_parts.append(part.get('text', ''))
        return "".join(text_parts)
    return str(content) if content else ""


def is_neo4j_timeout_error(error_msg: str) -> bool:
    """Check if an error message indicates Neo4j timeout."""
    error_lower = error_msg.lower().replace(" ", "")
    return "transactiontimedout" in error_lower or "timedout" in error_lower


def classify_query_step(query: str) -> str:
    """Classify a Cypher query into a human-readable step name."""
    query_lower = query.lower()
    if "tolower" in query_lower or "contains" in query_lower:
        return "LOOKING UP VALUE"
    if "labels(" in query_lower:
        return "CHECKING LABELS"
    if "count(" in query_lower:
        return "AGGREGATING"
    if "limit" in query_lower:
        return "FETCHING LIMITED"
    return "EXECUTING QUERY"


def safe_json_dumps(data, indent: int = 2) -> str:
    """Safely convert data to JSON string."""
    try:
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                return json.dumps(parsed, indent=indent, default=str)
            except json.JSONDecodeError:
                return data
        return json.dumps(data, indent=indent, default=str)
    except TypeError:
        return str(data)


def truncate_output(text: str, config: TruncationConfig = None) -> tuple[str, bool]:
    """Truncate text if too long. Returns (text, was_truncated)."""
    config = config or TruncationConfig()
    lines = text.split('\n')
    
    if len(lines) > config.max_lines or len(text) > config.max_chars:
        truncated = '\n'.join(lines[:config.max_lines])
        if len(truncated) > config.max_chars:
            truncated = truncated[:config.max_chars]
        return truncated, True
    return text, False


def count_items_in_json(data) -> Optional[int]:
    """Get item count if data is a list."""
    try:
        if isinstance(data, str):
            parsed = json.loads(data)
            if isinstance(parsed, list):
                return len(parsed)
        elif isinstance(data, list):
            return len(data)
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def get_error_hint(error_msg: str, last_query: Optional[str] = None) -> str:
    """Get a helpful hint for common errors."""
    if is_neo4j_timeout_error(error_msg):
        hint = "⏱️ Query timed out - too much data or complex query."
        if last_query:
            hint += f"\n\n**Captured query:**\n```\n{last_query}\n```"
            hint += "\n\n**You can run this query directly on:**"
            hint += "\n- https://iyp.iijlab.net (web interface)"
            hint += "\n- bolt://iyp-bolt.ihr.live:7687 (Neo4j Browser)"
        hint += "\n\n💡 Tip: Add 'LIMIT 100' to reduce data size."
        return hint
    if "429" in error_msg.lower() or "quota" in error_msg.lower():
        return "API rate limit hit. Please wait a moment and try again."
    if "syntax" in error_msg.lower():
        return "Cypher syntax error in the generated query."
    return ""


# =============================================================================
# CONNECTION DIAGNOSTICS
# =============================================================================

def test_neo4j_connection(host="iyp-bolt.ihr.live", port=7687, timeout=10) -> tuple[bool, Optional[str]]:
    """Test if Neo4j server is reachable via socket."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0, None
    except Exception as e:
        return False, str(e)


def get_server_ip(host="iyp-bolt.ihr.live") -> tuple[Optional[str], Optional[str]]:
    """Resolve server hostname to IP."""
    try:
        return socket.gethostbyname(host), None
    except socket.gaierror as e:
        return None, str(e)


def get_outbound_ip() -> tuple[Optional[str], Optional[str]]:
    """Get the outbound IP that servers see."""
    try:
        external_ip = urllib.request.urlopen('https://api.ipify.org', timeout=5).read().decode('utf8')
        return external_ip, None
    except Exception as e:
        return None, str(e)


# =============================================================================
# CUSTOM STYLES
# =============================================================================

st.markdown("""
<style>
button[kind="primary"] {
    background-color: #d9534f !important;
    border-color: #d9534f !important;
    color: #ffffff !important;
}
button[kind="primary"]:hover {
    background-color: #c9302c !important;
    border-color: #c12e2a !important;
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "chat_history": [],
        "agent": None,
        "system_prompt": None,
        "memory_enabled": False,
        "memory_depth": 0,
        "user_api_key": "",           # Applied/active API key
        "pending_api_key": "",         # Key being typed (not yet applied)
        "agent_key": None,
        "selected_question": None,
        "all_queries": [],
        # Neo4j connection settings (pending until Connect is clicked)
        "neo4j_uri": DEFAULT_NEO4J_URI,
        "neo4j_username": DEFAULT_NEO4J_USERNAME,
        "neo4j_password": DEFAULT_NEO4J_PASSWORD,
        "neo4j_database": DEFAULT_NEO4J_DATABASE,
        "neo4j_config_hash": None,  # Track config changes
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

init_session_state()


# =============================================================================
# ASYNC UTILITIES
# =============================================================================

def ensure_background_loop():
    """Ensure a background asyncio event loop is running."""
    loop = st.session_state.get('_asyncio_loop')
    loop_thread = st.session_state.get('_asyncio_loop_thread')

    if loop is None or loop.is_closed() or loop_thread is None or not loop_thread.is_alive():
        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()
        st.session_state._asyncio_loop = loop
        st.session_state._asyncio_loop_thread = loop_thread
    return loop


def run_async_task(coro):
    """Run an async coroutine in the background event loop."""
    loop = ensure_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


# =============================================================================
# AGENT INITIALIZATION
# =============================================================================

def load_system_prompt_schema() -> str:
    """Load schema from system-prompt file in the same directory as this app."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(script_dir, "system-prompt")
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _truncate_json_content(content: str) -> str:
    """Truncate JSON array if it exceeds MAX_TOOL_RESULT_ITEMS."""
    if not isinstance(content, str):
        return content
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content
    
    if isinstance(parsed, list) and len(parsed) > MAX_TOOL_RESULT_ITEMS:
        total = len(parsed)
        truncated = parsed[:MAX_TOOL_RESULT_ITEMS]
        truncated.append({
            "__note": f"Showing {MAX_TOOL_RESULT_ITEMS} of {total} total results. "
                      f"Press Export Results for full data."
        })
        return json.dumps(truncated)
    return content


def _wrap_tool_with_truncation(tool):
    """Wrap Neo4j tool so large results are truncated for LLM but full data goes to artifact."""
    original_coro = tool.coroutine
    if original_coro is None:
        return tool

    @functools.wraps(original_coro)
    async def _truncated(*args, **kwargs):
        result = await original_coro(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            content, _ = result
            return (_truncate_json_content(content), content)
        return result

    tool.coroutine = _truncated
    return tool


def initialize_agent(api_key: str, neo4j_uri: str, neo4j_username: str, neo4j_password: str, neo4j_database: str):
    """Initialize the agent with MCP tools and system prompt.
    
    Note: Not cached to ensure API keys are session-isolated for security.
    """
    schema = load_system_prompt_schema()
    if not schema:
        st.warning("Warning: system-prompt file not found. Agent may have limited schema knowledge.")
    
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(schema=schema)
    read_timeout = os.getenv("NEO4J_READ_TIMEOUT", "120")
    
    async def get_tools():
        client = MultiServerMCPClient({
            "neo4j": {
                "command": "uvx",
                "args": [
                    "--with", "fastmcp<3.0.0"
                    "mcp-neo4j-cypher@0.5.2",
                    "--transport", "stdio",
                    "--db-url", neo4j_uri,
                    "--username", neo4j_username,
                    "--password", neo4j_password,
                    "--database", neo4j_database,
                    "--read-timeout", read_timeout,
                ],
                "transport": "stdio"
            }
        })
        tools = await client.get_tools()
        return [t for t in tools if t.name == 'read_neo4j_cypher']
    
    try:
        tools = run_async_task(get_tools())
    except Exception as e:
        st.error(f"Failed to initialize MCP tools: {e}")
        return None, None
    
    tools = [_wrap_tool_with_truncation(t) for t in tools]
    
    # Initialize Gemini model
    gemini = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        temperature=0
    )
    
    agent = create_react_agent(gemini, tools)
    return agent, system_prompt


# =============================================================================
# AGENT QUERY EXECUTION
# =============================================================================

async def query_agent(agent, messages):
    """Query the agent with streaming and capture all steps."""
    steps = []
    final_response = None
    captured_queries = []
    last_query = None

    try:
        async for event in agent.astream(
            {"messages": messages},
            config={"recursion_limit": 50},
            stream_mode="values"
        ):
            if "messages" not in event:
                continue
            
            event_messages = event["messages"]
            if not event_messages:
                continue
            
            last_msg = event_messages[-1]
            msg_type = last_msg.__class__.__name__

            if msg_type == "AIMessage":
                tool_calls = getattr(last_msg, 'tool_calls', [])
                content = getattr(last_msg, 'content', '')

                if tool_calls:
                    for tool_call in tool_calls:
                        tool_name = tool_call.get('name', 'unknown')
                        args = tool_call.get('args', {})
                        
                        if tool_name == 'read_neo4j_cypher':
                            query = args.get('cypher') or args.get('query', '')
                            if query:
                                if query not in captured_queries:
                                    captured_queries.append(query)
                                last_query = query
                                steps.append({
                                    'type': 'tool_call',
                                    'tool': tool_name,
                                    'step_label': classify_query_step(query),
                                    'query': query,
                                    'args': args
                                })
                        else:
                            steps.append({
                                'type': 'tool_call',
                                'tool': tool_name,
                                'step_label': 'TOOL CALL',
                                'args': args
                            })
                elif content:
                    steps.append({
                        'type': 'final_answer',
                        'content': extract_text_content(content)
                    })

            elif msg_type == "ToolMessage":
                full_content = getattr(last_msg, 'artifact', None) or getattr(last_msg, 'content', '')
                tool_name = getattr(last_msg, 'name', 'unknown')

                try:
                    result = json.loads(full_content) if isinstance(full_content, str) else full_content
                except (json.JSONDecodeError, TypeError):
                    result = full_content

                content_str = full_content if isinstance(full_content, str) else str(full_content)
                is_error = "error" in content_str.lower()
                error_hint = get_error_hint(content_str, last_query) if is_error else None

                steps.append({
                    'type': 'tool_result',
                    'tool': tool_name,
                    'result': result,
                    'is_error': is_error,
                    'error_hint': error_hint,
                })

            final_response = event

    except Exception as e:
        error_hint = get_error_hint(str(e), last_query)
        steps.append({
            'type': 'error',
            'message': str(e),
            'hint': error_hint
        })
        return None, steps, captured_queries, last_query

    if final_response:
        response_data = final_response.get("agent", final_response) if "agent" in final_response else final_response
        cleaned = extract_response(response_data)
        return cleaned, steps, captured_queries, last_query

    return None, steps, captured_queries, last_query


def extract_response(response):
    """Extract structured data from agent response."""
    messages = response.get("messages", [])
    ai_msgs = [m for m in messages if getattr(m, "role", None) == "assistant" or m.__class__.__name__ == "AIMessage"]
    ai_answer = ""
    if ai_msgs:
        ai_answer = extract_text_content(getattr(ai_msgs[-1], "content", ""))
    return {"llm_answer": ai_answer, "message_count": len(messages)}


# =============================================================================
# MESSAGE BUILDING
# =============================================================================

def build_conversation_messages(system_prompt: str, user_query: str) -> list:
    """Build conversation messages with optional history."""
    messages = [{"role": "system", "content": system_prompt}]

    history_depth = st.session_state.memory_depth if st.session_state.memory_enabled else 0
    history_depth = min(history_depth, 5)

    if history_depth and st.session_state.chat_history:
        recent_history = st.session_state.chat_history[-history_depth:]
        for chat in recent_history:
            messages.append({"role": "user", "content": chat['question']})
            messages.append({"role": "assistant", "content": chat.get('answer', '')})

    messages.append({"role": "user", "content": user_query})
    return messages


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_title():
    """Display app title."""
    st.title(APP_TITLE)
    st.markdown(APP_TAGLINE)


def render_connection_diagnostics():
    """Render connection diagnostics expander."""
    with st.expander("Connection Diagnostics", expanded=False):
        if st.button("Test Neo4j Connection"):
            with st.spinner("Testing connection..."):
                server_ip, dns_error = get_server_ip()
                if server_ip:
                    st.info(f"Server IP: {server_ip}")
                else:
                    st.error(f"DNS resolution failed: {dns_error}")
                
                can_connect, conn_error = test_neo4j_connection()
                if can_connect:
                    st.success("Socket connection successful - port 7687 is reachable")
                else:
                    st.error(f"Cannot connect to server: {conn_error}")
                
                outbound_ip, _ = get_outbound_ip()
                if outbound_ip:
                    st.info(f"Your outbound IP: {outbound_ip}")
                
                st.markdown("---")
                if can_connect:
                    st.success("Network connectivity looks good.")
                else:
                    st.error("Cannot reach Neo4j server. Run this app locally or contact IYP team.")


def render_user_guidance():
    """Show user guidance and sample questions."""
    st.info(WARNING_TEXT)
    st.markdown("**Try one of these example questions:**")
    
    cols = st.columns(2)
    for idx, question in enumerate(SAMPLE_QUESTIONS):
        col = cols[idx % 2]
        with col:
            if st.button(question, key=f"sample_q_{idx}", use_container_width=True):
                st.session_state.selected_question = question
    
    st.caption(f"IYP Console reference: {IYP_CONSOLE_URL}")


def render_sidebar() -> str:
    """Render sidebar with API key input, Neo4j settings, memory controls, and stats."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.info("ℹ️ **Public Demo** - Provide your own Gemini API key.")
        st.markdown("")
        # Gemini API key with Connect button
        st.markdown("### Gemini API")
        st.markdown("[Get your API key from Google AI Studio →](https://aistudio.google.com/apikey)")
        
        # Show current connection status
        if st.session_state.agent:
            st.success("Connected")
        
        # Initialize pending_api_key from user_api_key if not set
        if not st.session_state.pending_api_key and st.session_state.user_api_key:
            st.session_state.pending_api_key = st.session_state.user_api_key
        
        # Use pending key for input, only apply on button click
        pending_key = st.text_input(
            "API Key",
            value=st.session_state.pending_api_key,
            type="password",
            help="Enter your Google AI Studio API key",
        )
        st.session_state.pending_api_key = pending_key.strip()
        
        # Connect button - applies the API key
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Connect", use_container_width=True, type="primary"):
                if st.session_state.pending_api_key:
                    st.session_state.user_api_key = st.session_state.pending_api_key
                    st.session_state.agent = None 
                    st.session_state.agent_key = None
                    st.session_state.neo4j_config_hash = None
                    st.rerun()
                else:
                    st.warning("Please enter an API key")
        with col2:
            if st.button("Clear", use_container_width=True):
                st.session_state.user_api_key = ""
                st.session_state.pending_api_key = ""
                st.session_state.agent = None
                st.session_state.agent_key = None
                st.session_state.system_prompt = None
                st.session_state.neo4j_config_hash = None
                st.rerun()

        # Neo4j connection settings
        st.markdown("### Neo4j Database")
        st.caption(f"{st.session_state.neo4j_uri}")
        
        with st.expander("Connection Settings", expanded=False):
            pending_uri = st.text_input(
                "Neo4j URI",
                value=st.session_state.neo4j_uri,
                help="bolt://host:port (default: IYP public database)"
            )
            pending_username = st.text_input(
                "Username",
                value=st.session_state.neo4j_username,
            )
            pending_password = st.text_input(
                "Password",
                value=st.session_state.neo4j_password,
                type="password",
                help="Leave empty for IYP public database"
            )
            pending_database = st.text_input(
                "Database",
                value=st.session_state.neo4j_database,
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Apply", use_container_width=True):
                    st.session_state.neo4j_uri = pending_uri
                    st.session_state.neo4j_username = pending_username
                    st.session_state.neo4j_password = pending_password
                    st.session_state.neo4j_database = pending_database
                    st.session_state.agent = None
                    st.session_state.neo4j_config_hash = None
                    st.rerun()
            with col2:
                if st.button("Reset", use_container_width=True):
                    st.session_state.neo4j_uri = DEFAULT_NEO4J_URI
                    st.session_state.neo4j_username = DEFAULT_NEO4J_USERNAME
                    st.session_state.neo4j_password = DEFAULT_NEO4J_PASSWORD
                    st.session_state.neo4j_database = DEFAULT_NEO4J_DATABASE
                    st.session_state.agent = None
                    st.session_state.neo4j_config_hash = None
                    st.rerun()
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.all_queries = []
            st.rerun()

        st.markdown("### Memory")
        st.session_state.memory_enabled = st.checkbox(
            "Remember previous answers",
            value=st.session_state.memory_enabled
        )
        if st.session_state.memory_enabled:
            current_depth = st.session_state.memory_depth if 1 <= st.session_state.memory_depth <= 5 else 1
            st.session_state.memory_depth = st.slider(
                "How many previous answers?",
                min_value=1, max_value=5, value=current_depth, step=1
            )
        else:
            st.session_state.memory_depth = 0

        st.markdown("---")
        st.markdown("### Session Stats")
        st.text(f"Conversations: {len(st.session_state.chat_history)}")
        st.text(f"Queries executed: {len(st.session_state.all_queries)}")

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        
This app uses:
- Google Gemini 2.5 Flash via LangChain
- MCP Neo4j server tools
- IYP public Neo4j database""")

    return st.session_state.user_api_key


def render_reasoning_step(step: dict, idx: int):
    """Render a single reasoning step."""
    step_type = step.get('type', '')
    
    if step_type == 'tool_call':
        label = step.get('step_label', 'TOOL CALL')
        st.markdown(f"**Step {idx}: [{label}]**")
        if 'query' in step:
            st.code(step['query'], language='cypher')
        else:
            st.json(step.get('args', {}))
            
    elif step_type == 'tool_result':
        st.markdown(f"**Step {idx}: Result**")
        if step.get('is_error'):
            st.error("Error in query execution")
            if step.get('error_hint'):
                st.warning(f"Hint: {step['error_hint']}")
        else:
            result = step.get('result')
            item_count = count_items_in_json(result)
            result_str = safe_json_dumps(result)
            truncated, was_truncated = truncate_output(result_str)
            
            if was_truncated:
                if item_count:
                    st.caption(f"Showing preview ({item_count} total items)")
                try:
                    st.json(json.loads(truncated))
                except (json.JSONDecodeError, TypeError):
                    st.text(truncated)
                st.caption("(Output truncated)")
            else:
                if item_count:
                    st.caption(f"{item_count} items returned")
                st.json(result)
                
    elif step_type == 'final_answer':
        st.markdown(f"**Step {idx}: Final Answer**")
        st.write(step.get('content', ''))
        
    elif step_type == 'error':
        st.markdown(f"**Step {idx}: Error**")
        st.error(step.get('message', 'Unknown error'))
        if step.get('hint'):
            st.warning(f"Hint: {step['hint']}")
    
    st.markdown("---")


def render_chat_history():
    """Render chat history with reasoning steps."""
    for chat_idx, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat['question'])
        
        with st.chat_message("assistant"):
            st.write(chat['answer'])
            
            if chat.get('steps'):
                with st.expander(f"🧠 View Reasoning Steps ({len(chat['steps'])} steps)"):
                    for idx, step in enumerate(chat['steps'], 1):
                        render_reasoning_step(step, idx)
            
            if chat.get('queries'):
                st.caption("Queries used:")
                for q in chat['queries']:
                    st.code(q, language='cypher')
            
            if chat.get('answer_time'):
                st.caption(f"Answered in {chat['answer_time']:.1f}s")
            
            # Collect results for export
            all_results = []
            for step in chat.get('steps', []):
                if step.get('type') == 'tool_result' and not step.get('is_error'):
                    result = step.get('result')
                    if result:
                        all_results.append(result)
            
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("Delete", key=f"delete_chat_{chat_idx}", type="primary"):
                    st.session_state.chat_history.pop(chat_idx)
                    st.rerun()
            with col2:
                export_data = {
                    "question": chat['question'],
                    "answer": chat['answer'],
                    "queries": chat.get('queries', []),
                    "results": all_results,
                }
                json_str = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
                st.download_button(
                    label="Export Results",
                    data=json_str,
                    file_name=f"iyp_results_{chat_idx + 1}.json",
                    mime="application/json",
                    key=f"export_{chat_idx}"
                )


# =============================================================================
# RETRY HELPERS
# =============================================================================

def check_for_timeout_in_steps(steps: list) -> tuple[bool, Optional[str]]:
    """Check if any step has a timeout error."""
    failing_query = None
    for step in steps:
        if step.get('type') == 'tool_call' and step.get('query'):
            failing_query = step.get('query')
        if step.get('type') == 'tool_result' and step.get('is_error'):
            result_str = str(step.get('result', ''))
            if is_neo4j_timeout_error(result_str):
                return True, failing_query
    return False, None


def check_for_empty_results_in_steps(steps: list) -> tuple[bool, Optional[str]]:
    """Check if all tool results in steps are empty."""
    last_query = None
    has_any_results = False
    for step in steps:
        if step.get('type') == 'tool_call' and step.get('query'):
            last_query = step.get('query')
        if step.get('type') == 'tool_result' and not step.get('is_error'):
            result_str = str(step.get('result', ''))
            stripped = result_str.strip()
            if stripped and stripped not in ('[]', '"[]"', ''):
                has_any_results = True
    return not has_any_results, last_query


# =============================================================================
# MAIN QUERY HANDLER
# =============================================================================

def handle_user_query(user_query: str):
    """Run full agent interaction for a user question with retry logic."""
    with st.chat_message("user"):
        st.write(user_query)
    
    with st.chat_message("assistant"):
        conversation_messages = build_conversation_messages(
            st.session_state.system_prompt,
            user_query
        )
        
        all_steps = []
        all_queries = []
        response_data = None
        max_retries = 3
        used_limit = False
        total_start = time.time()
        
        for attempt in range(max_retries):
            if attempt == 0:
                status_text = "🤔 Agent is thinking..."
            elif not used_limit:
                status_text = f"Retrying with LIMIT (attempt {attempt + 1})..."
            else:
                status_text = f"🔄 Agent is rethinking (attempt {attempt + 1})..."
            
            with st.spinner(status_text):
                try:
                    response_data, steps, captured_queries, last_query = run_async_task(
                        query_agent(st.session_state.agent, conversation_messages)
                    )
                    
                    all_steps.extend(steps)
                    for q in captured_queries:
                        if q not in all_queries:
                            all_queries.append(q)
                        if q not in st.session_state.all_queries:
                            st.session_state.all_queries.append(q)
                    
                    # Check for timeout - retry with LIMIT or rethink
                    has_timeout, failing_query = check_for_timeout_in_steps(steps)
                    if has_timeout and attempt < max_retries - 1:
                        if used_limit:
                            st.warning("⚠️ LIMIT retry also timed out. Asking agent to rethink...")
                            query_context = f"\n\nThe failing query was:\n```cypher\n{failing_query}\n```" if failing_query else ""
                            rethink_msg = f"""The previous query timed out even with LIMIT.{query_context}

Please generate a completely NEW query with a different, more efficient approach for: "{user_query}"

Check the schema for correct node labels, relationship types and directions."""
                            conversation_messages.append({"role": "assistant", "content": "The query timed out even with LIMIT."})
                            conversation_messages.append({"role": "user", "content": rethink_msg})
                            continue
                        
                        st.warning("⏱️ Query timed out. Retrying with LIMIT...")
                        retry_msg = f"The previous query timed out. Please add LIMIT 100 and retry."
                        if failing_query:
                            retry_msg = f"The previous query timed out. Please add LIMIT 100:\n\n{failing_query}"
                        conversation_messages.append({"role": "assistant", "content": "The query timed out."})
                        conversation_messages.append({"role": "user", "content": retry_msg})
                        used_limit = True
                        continue
                    
                    # Check for empty results after LIMIT - ask to rethink
                    if used_limit and attempt < max_retries - 1:
                        is_empty, empty_query = check_for_empty_results_in_steps(steps)
                        if is_empty:
                            st.warning("⚠️ LIMIT retry returned 0 results. Asking agent to rethink...")
                            query_context = f"\n\nThe failing query was:\n```cypher\n{empty_query}\n```" if empty_query else ""
                            rethink_msg = f"""The previous query returned 0 results.{query_context}

Please generate a completely NEW query with a different approach for: "{user_query}"

Check the schema for correct node labels, relationship types and directions."""
                            conversation_messages.append({"role": "assistant", "content": response_data.get('llm_answer', 'No results.')})
                            conversation_messages.append({"role": "user", "content": rethink_msg})
                            continue
                    
                    break
                        
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    hint = get_error_hint(str(exc))
                    if hint:
                        st.warning(f"Hint: {hint}")
                    response_data, all_steps, all_queries = None, [], []
                    break
        
        total_elapsed = time.time() - total_start
        
        if response_data:
            if all_steps:
                with st.expander(f"🧠 View Reasoning Steps ({len(all_steps)} steps)", expanded=False):
                    for idx, step in enumerate(all_steps, 1):
                        render_reasoning_step(step, idx)
            
            st.write(response_data['llm_answer'])
            
            if all_queries:
                st.caption("Queries used:")
                for q in all_queries:
                    st.code(q, language='cypher')
            
            st.caption(f"Answered in {total_elapsed:.1f}s")
            
            st.session_state.chat_history.append({
                'question': user_query,
                'answer': response_data['llm_answer'],
                'steps': all_steps,
                'queries': all_queries,
                'answer_time': total_elapsed,
            })
        else:
            st.error("Failed to get response from agent")


# =============================================================================
# MAIN APP
# =============================================================================

def get_neo4j_config_hash() -> str:
    """Get a hash of current Neo4j config to detect changes."""
    return f"{st.session_state.neo4j_uri}|{st.session_state.neo4j_username}|{st.session_state.neo4j_database}"


def main():
    """Main application entry point."""
    render_title()
    current_api_key = render_sidebar()
    
    if not current_api_key:
        st.info("Enter your Gemini API key in the sidebar to start chatting.")
        render_connection_diagnostics()
        st.stop()
    
    # Check if API key or Neo4j config changed
    current_config_hash = get_neo4j_config_hash()
    config_changed = (
        (st.session_state.agent_key and st.session_state.agent_key != current_api_key) or
        (st.session_state.neo4j_config_hash and st.session_state.neo4j_config_hash != current_config_hash)
    )
    if config_changed:
        st.session_state.agent = None
    
    if st.session_state.agent is None:
        with st.spinner("Initializing agent..."):
            st.session_state.agent, st.session_state.system_prompt = initialize_agent(
                current_api_key,
                st.session_state.neo4j_uri,
                st.session_state.neo4j_username,
                st.session_state.neo4j_password,
                st.session_state.neo4j_database
            )
            st.session_state.agent_key = current_api_key
            st.session_state.neo4j_config_hash = current_config_hash
            
            if st.session_state.agent:
                st.success("Agent initialized successfully!")
            else:
                st.error("Failed to initialize agent. Check your API key and connection.")
                render_connection_diagnostics()
                st.stop()
    
    render_user_guidance()
    render_connection_diagnostics()
    st.markdown("---")
    render_chat_history()
    
    user_query = st.chat_input("Ask a question about the IYP database...")
    
    if st.session_state.selected_question:
        user_query = st.session_state.selected_question
        st.session_state.selected_question = None
    
    if user_query:
        handle_user_query(user_query)
        st.rerun()


if __name__ == "__main__":
    main()

