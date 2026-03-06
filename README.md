# NLQ4IYP: Natural Language Queries for the Internet Yellow Pages Graph

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nlq4iyp.streamlit.app/)

 An agent-based natural language interface for querying the [Internet Yellow Pages (IYP)](https://iyp.iijlab.net/) knowledge graph, developed as part of a **bachelor thesis**

Users can ask questions about global Internet infrastructure in plain English and receive answers with data from the IYP Neo4j database.

## What is IYP?

The [Internet Yellow Pages (IYP)](https://iyp.iijlab.net/) is a Neo4j graph database composed of over 60 Internet measurement datasets, designed to facilitate the exploration and analysis of Internet topology data. For more details, see the paper [*"The Wisdom of the Measurement Crowd"*](https://www.iijlab.net/en/members/romain/pdf/romain_imc2024.pdf) and [IYP Tutorial](https://tutorial.iyp.ihr.live/content/start/what-is-iyp.html).

## Features

- 💬 **Natural Language Queries** — Ask questions in plain English; the agent translates them to Cypher
- 🧠 **Transparent Reasoning** — Inspect the full ReAct reasoning chain: generated Cypher queries, intermediate results, and the final answer
- 🔍 **Live Database Access** — Queries execute in real time against the public IYP Neo4j instance
- 📊 **Result Export** — Download full query results as JSON
- ⚡ **Auto-Retry on Timeout** — Intelligent `LIMIT` fallback when queries are too broad
- 💾 **Conversation Memory** — Optionally retain previous Q&A pairs as context for follow-up questions
- 🛠️ **Configurable Neo4j Endpoint** — Connect to any compatible Neo4j instance via the sidebar

## Architecture

```
User Question
     │
     ▼
┌──────────────────────┐
│   Gemini 2.5 Flash   │  ← Translates to Cypher
│   (ReAct Agent)      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   MCP Neo4j Server   │  ← Executes queries via MCP
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   IYP Neo4j Database │  ← Returns results
└──────────┬───────────┘
           │
           ▼
  Natural Language Answer
```

The agent follows a **ReAct (Reasoning + Acting)** loop:

1. **Reason** about the user's question
2. **Generate** a Cypher query based on the IYP schema
3. **Execute** the query against the database via MCP tools
4. **Evaluate** results, refine the query if results are empty or insufficient
5. **Respond** with a human-readable answer

The system prompt includes the full IYP graph schema (node types, relationship types, and their properties) along with 166 example Cypher queries, enabling the LLM to generate accurate queries without fine-tuning.

## Sample Questions

- *"Find the IXPs where AS8075 (Microsoft) is a member"*
- *"Which ASes are located in Japan?"*
- *"What prefixes does AS2497 originate?"*
- *"Find the CAIDA rank for AS15169 (Google)"*
- *"List facilities in Frankfurt"*
- *"What is the AS hegemony value between AS2907 and AS2497?"*

## Getting Started

### Streamlit Cloud (No Setup)

👉 **[Launch NLQ4IYP Assistant](https://nlq4iyp.streamlit.app/)**

Enter your [Google AI Studio API key](https://aistudio.google.com/apikey) in the sidebar to start querying.

### Run Locally
Clone the repository:
```bash
git clone https://github.com/isidor0s/NLQ_4_IYP.git
cd NLQ_4_IYP
```
Install dependencies:
```bash
pip install -r requirements.txt
```
run the app:
```bash
streamlit run app.py
```

> **Prerequisites:** Python 3.10+, [`uvx`](https://docs.astral.sh/uv/) (used to launch the MCP Neo4j server process)

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_GENAI_API_KEY` | Gemini API key (required) | — |
| `NEO4J_URI` | IYP database endpoint | `bolt://iyp-bolt.ihr.live:7687` |
| `NEO4J_USERNAME` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | *(empty for public access)* |
| `NEO4J_DATABASE` | Neo4j database name | `neo4j` |
| `NEO4J_READ_TIMEOUT` | Query timeout in seconds | `120` |

For Streamlit Cloud deployment, add `GOOGLE_GENAI_API_KEY` to your app secrets. Users can also enter their key directly in the sidebar.

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | [Streamlit](https://streamlit.io/) |
| **LLM** | [Google Gemini 2.5 Flash](https://ai.google.dev/) |
| **Agent Framework** | [LangChain Agents (create_agent)](https://python.langchain.com/docs/modules/agents/) |
| **LLM Integration** | [LangChain Google GenAI](https://python.langchain.com/docs/integrations/chat/google_generative_ai/) |
| **Database Access** | [MCP Neo4j Cypher](https://github.com/neo4j-contrib/mcp-neo4j) via [Model Context Protocol](https://modelcontextprotocol.io/) |
| **Data Source** | [Internet Yellow Pages (IYP)](https://iyp.iijlab.net/) |

## Project Structure

```
├── app.py             # Main Streamlit application (~1100 lines)
├── system-prompt      # IYP graph schema + example Cypher queries
├── requirements.txt   # Python dependencies
└── README.md
```

## Limitations

- Query results are truncated to 50 items in the UI (full results available via JSON export)
- Complex or broad queries may timeout on the public IYP database
- The LLM may occasionally generate incorrect Cypher, always verify via the reasoning panel

## Evaluation

System performance is validated through a comprehensive benchmark framework.

📊 **[Evaluation Repository](https://github.com/isidor0s/nlq4iyp_evaluation)** — For more details

## Related Links

- [IYP Console](https://iyp.iijlab.net/)
- [IHR Platform](https://www.ihr.live/en) 
- [IYP Documentation](https://github.com/InternetHealthReport/internet-yellow-pages)

---

*Bachelor thesis project: Querying the Internet Yellow Pages knowledge graph with natural language*
