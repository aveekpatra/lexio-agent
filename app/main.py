"""
Lexio Agent - FastAPI Application.

Slash commands:
  /law <query>     - AI law research
  /case <query>    - AI judgment research
  /search <query>  - Quick search (no AI)
  /web <query>     - Web search (Perplexity)
"""
import json
import re
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import settings
from app.agent import agent
from app import tools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown."""
    logger.info("Starting Lexio Agent...")
    yield
    logger.info("Shutting down Lexio Agent...")
    await tools.close_pool()


app = FastAPI(
    title="Lexio Agent",
    description="Legal AI assistant with slash commands",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# SLASH COMMAND PARSER
# =============================================================================

def parse_command(text: str) -> tuple[str, str]:
    """
    Parse slash command from input.
    """
    text = text.strip()
    
    # Match /command pattern
    match = re.match(r'^/(\w+)\s+(.+)$', text, re.DOTALL)
    if match:
        command = match.group(1).lower()
        query = match.group(2).strip()
        
        # Validate command
        valid_commands = ["law", "case", "search", "web", "analyze", "doc"]
        if command in valid_commands:
            return command, query
    
    # No valid slash command - default to /law
    return "law", text


# =============================================================================
# MAIN ENDPOINT
# =============================================================================

@app.get("/query")
async def query_stream(
    q: str = Query(..., min_length=1, description="Query with optional slash command")
):
    """
    Main query endpoint with slash command support.
    """
    command, query = parse_command(q)
    logger.info(f"Received query: command={command}, query='{query}'")
    
    # Quick search - no AI, just return results
    if command == "search":
        async def generate():
            results = await agent.quick_search(query, source="law")
            yield f"data: {json.dumps({'event': 'results', 'results': results}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'event': 'done'})}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    # Web search - no AI iterative, just Perplexity
    if command == "web":
        async def generate():
            result = await tools.web_search(query)
            if result:
                yield f"data: {json.dumps({'event': 'answer', 'content': result['answer']}, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps({'event': 'error', 'content': 'Web search failed'})}\n\n"
            yield f"data: {json.dumps({'event': 'done'})}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    # AI research - law or case
    async def generate():
        try:
            async for event in agent.ask(query, mode=command):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"}
    )


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "model": settings.AGENT_MODEL}


@app.get("/")
async def root():
    """API info."""
    return {
        "service": "Lexio Agent",
        "version": "1.0.0",
        "model": settings.AGENT_MODEL,
        "commands": {
            "/law <query>": "AI law research (default)",
            "/case <query>": "AI judgment research",
            "/search <query>": "Quick search, no AI",
            "/web <query>": "Web search via Perplexity"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
