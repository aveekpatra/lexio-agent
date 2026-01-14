# Lexio Agent - MCP Agentic Backend

FastAPI backend with MCP-based agentic workflow for Czech legal search.

## Features

- 🔍 Hybrid search (vector + BM25) on Railway PostgreSQL
- ⚖️ Court judgment search via Qdrant
- 🌐 Web search via Perplexity Sonar
- 🤖 LangGraph-based agent orchestration
- 📡 Server-Sent Events streaming

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
uvicorn app.main:app --reload
```

## Environment Variables

```
DATABASE_URL=postgres://...@railway.net/railway
OPENAI_API_KEY=sk-...
QDRANT_URL=https://...
QDRANT_API_KEY=...
PERPLEXITY_API_KEY=pplx-...
```

## API

```
GET /stream?query=...&tools=law,case,web
```

Streams agent events as SSE.
