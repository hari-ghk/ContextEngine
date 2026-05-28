# ContextEngine 🧠

> A production-grade Generative AI and Agentic AI system built from scratch — featuring RAG, autonomous agents, MCP tool integration, LLM observability, evals, and guardrails.

---

## Overview

ContextEngine is a end-to-end AI backend system that demonstrates how to build, orchestrate, and productionize Large Language Models. It goes beyond simple chat — combining retrieval-augmented generation, autonomous agents, real tool execution via MCP, and a full observability stack.

Built with a senior backend engineer's mindset: clean architecture, separation of concerns, measurable quality, and production-ready patterns.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                            │
│    /chat (Phase 1)  /query (Phase 2)  /ask (Phase 4)       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Guardrails Layer                          │
│     Prompt injection detection · Harmful content filter    │
│     Off-topic query detection · Input validation           │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Agent Layer (ReAct Loop)                  │
│                                                             │
│   ┌─────────────────┐  ┌──────────────┐  ┌─────────────┐  │
│   │  RAG Tool       │  │  Web Search  │  │  MCP Tool   │  │
│   │  ChromaDB       │  │  Tavily API  │  │  Python     │  │
│   │  Semantic Search│  │  Live Web    │  │  Execution  │  │
│   └─────────────────┘  └──────────────┘  └─────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   LLM Layer                                 │
│         Claude Haiku 4.5 (dev) · Claude Sonnet 4.6 (prod)  │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Observability Layer                       │
│   Token tracking · Cost per request · Latency (ms)         │
│   Faithfulness score · Answer relevance score · Sources    │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Features

### Phase 1 — LLM-Powered Chat API
- FastAPI REST endpoint powered by a Groq-hosted LLM
- Dynamic system prompts — caller controls model behaviour
- Stateful conversation via message history
- Pydantic request validation and error handling

### Phase 2 — RAG Pipeline
- Document ingestion with recursive chunking (500 tokens, 50 overlap)
- Local embeddings via `sentence-transformers/all-MiniLM-L6-v2` — free, no API cost
- ChromaDB vector store with persistent storage
- Semantic search — meaning-based retrieval, not keyword matching
- Grounded answers with source citations

### Phase 3 — Agentic AI + MCP
- ReAct agent loop via LangGraph — Reason → Act → Observe → Repeat
- Three tools: RAG search, web search (Tavily), Python code execution
- MCP (Model Context Protocol) server for sandboxed code execution
- Temperature=0 for deterministic, consistent tool selection

### Phase 4 — Production-Grade Claude API
- Claude Haiku 4.5 via Anthropic API — superior tool use and instruction following
- Full observability: token count, cost (USD), latency (ms), source files
- LLM-as-judge evals: faithfulness score + answer relevance score
- Guardrails: prompt injection detection, harmful content filtering, off-topic blocking

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| LLM (Dev) | Groq — Llama 3.1 8B / 70B |
| LLM (Prod) | Anthropic — Claude Haiku 4.5 |
| Agent Framework | LangGraph + LangChain |
| Vector Database | ChromaDB (persistent, local) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Web Search | Tavily API |
| MCP Server | Anthropic MCP SDK (FastMCP) |
| Validation | Pydantic v2 |
| Evals | Custom LLM-as-judge (Claude) |

---

## Project Structure

```
ContextEngine/
├── main.py                 # Phase 1: LLM-powered chat API
├── rag.py                  # Phase 2: RAG query API
├── agent.py                # Phase 3: Groq-based ReAct agent
├── claude_agent.py         # Phase 4: Production Claude API
├── ingest.py               # One-time document ingestion pipeline
├── db.py                   # Shared ChromaDB client + embedding model
├── python_repl_mcp.py      # MCP server: sandboxed Python execution
├── evals.py                # LLM-as-judge: faithfulness + relevance
├── guardrails.py           # Input validation + safety layer
├── .env                    # API keys (never committed)
└── .gitignore
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- [Groq API key](https://console.groq.com) (free)
- [Anthropic API key](https://console.anthropic.com) (pay-per-use)
- [Tavily API key](https://app.tavily.com) (free tier)

### Installation

```bash
# Clone the repository
git clone https://github.com/hari-ghk/ContextEngine.git
cd ContextEngine

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install fastapi uvicorn openai python-dotenv chromadb \
    sentence-transformers langchain langchain-community \
    langchain-text-splitters langchain-groq langchain-anthropic \
    langgraph langchain-mcp-adapters tavily-python mcp anthropic
```

### Environment Setup

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
TAVILY_API_KEY=your_tavily_key_here
```

### Ingest Documents

```bash
# Download Python docs (plain text format) from python.org
# Place in python-3.14-docs-text/ folder, then:
python ingest.py
```

### Run the APIs

```bash
# Terminal 1 — MCP server (code execution)
python python_repl_mcp.py

# Terminal 2 — Production API
uvicorn claude_agent:app --reload --port 8002

# Terminal 3 — Groq agent (optional)
uvicorn agent:app --reload --port 8001

# Terminal 4 — Basic chat + RAG (optional)
uvicorn main:app --reload
```

---

## API Reference

### `POST /ask` — Production Claude Agent

The flagship endpoint. Combines RAG, web search, MCP code execution, guardrails, evals, and full observability.

**Request:**
```json
{
  "user_query": "What are Python generators and how do they work?"
}
```

**Response:**
```json
{
  "answer": "Python generators are functions that use the yield keyword...",
  "sources": ["python-3.14-docs-text/tutorial/classes.txt"],
  "model": "claude-haiku-4-5-20251001",
  "tokens_used": 2717,
  "latency_ms": 21578,
  "cost_usd": 0.003901,
  "faithfulness_score": 0.75,
  "relevance_score": 0.95
}
```

### `POST /chat` — Basic LLM Chat

```json
{
  "system_message": "You are a helpful assistant",
  "user_message": "What is a large language model?",
  "message_history": []
}
```

### `POST /query` — RAG Query

```json
{
  "user_query": "How do Python decorators work?"
}
```

### `POST /agent` — Groq ReAct Agent

```json
{
  "user_query": "What are the new features in Python 3.13?"
}
```

---

## How It Works

### RAG Pipeline

```
Documents → Chunk (500 tokens, 50 overlap) → Embed (all-MiniLM-L6-v2)
→ Store in ChromaDB → Query time: embed question → semantic search
→ retrieve top-5 chunks → inject into LLM prompt → grounded answer
```

### Agent ReAct Loop

```
User question → LLM reasons: which tool do I need?
→ Calls tool (RAG / web search / code execution)
→ Observes result → reasons again → final answer
```

### Evals (LLM-as-Judge)

```
Faithfulness: Claude judges if answer claims are supported by retrieved context
Answer Relevance: Claude judges if answer actually addresses the question
Both return a score 0.0–1.0 with reasoning
```

### Guardrails

```
Every request → Claude validates input →
blocks: prompt injection / harmful content / off-topic queries
passes: legitimate Python questions
```

---

## Design Decisions

**Why ChromaDB over Pinecone?**
Local, persistent, zero cost for development. Trivial to swap for Qdrant or Pinecone in production.

**Why separate ingestion from querying?**
Ingestion is a one-time batch operation. In production this would be a scheduled job (cron/Lambda) triggered by document changes, not part of the query path.

**Why custom evals over RAGAS?**
Full transparency and control. LLM-as-judge with Claude gives us the same quality as RAGAS without opaque dependencies or OpenAI coupling.

**Why temperature=0 for agents?**
Deterministic tool selection. Agents need consistent reasoning paths — randomness causes unpredictable tool usage.

**Why MCP for code execution?**
MCP (Model Context Protocol) is Anthropic's open standard for AI tool integration. Building an MCP server demonstrates understanding of the emerging agentic infrastructure layer.

---

## Production Considerations

In a production deployment this system would add:

- **Incremental ingestion** — detect document changes via MD5 hashing, re-ingest only changed files
- **Redis** for conversation history persistence across sessions  
- **Async ingestion pipeline** — Celery or AWS Lambda triggered on S3 upload
- **Rate limiting** — per-user token budgets
- **Structured logging** — ship observability metrics to Datadog/CloudWatch
- **Eval regression suite** — automated test set run on every deployment

---

## Author

**Harikiran** — Senior Backend Engineer transitioning into AI-first engineering  
GitHub: [@hari-ghk](https://github.com/hari-ghk)

---

*Built with curiosity, patience, and a lot of evening sessions.* 🚀