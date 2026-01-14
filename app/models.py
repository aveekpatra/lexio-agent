"""
Pydantic models for API requests and responses.
"""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field


# =============================================================================
# TOOL MODELS
# =============================================================================

class SearchResult(BaseModel):
    """A single search result from any tool."""
    id: str
    text: str
    score: float
    source: Literal["law", "case", "web"]
    metadata: dict = Field(default_factory=dict)


class ToolCall(BaseModel):
    """Represents a tool invocation."""
    tool: str
    input: dict
    

class ToolResult(BaseModel):
    """Result from a tool execution."""
    tool: str
    results: List[SearchResult]
    duration_ms: int


# =============================================================================
# AGENT EVENT MODELS (for SSE streaming)
# =============================================================================

class AgentEventThinking(BaseModel):
    """Agent is reasoning about the query."""
    event: Literal["thinking"] = "thinking"
    thought: str


class AgentEventToolCall(BaseModel):
    """Agent is calling a tool."""
    event: Literal["tool_call"] = "tool_call"
    tool: str
    input: dict


class AgentEventToolResult(BaseModel):
    """Tool returned results."""
    event: Literal["tool_result"] = "tool_result"
    tool: str
    results: List[SearchResult]
    duration_ms: int


class AgentEventAnswer(BaseModel):
    """Final answer chunk (streamed)."""
    event: Literal["answer"] = "answer"
    content: str
    done: bool = False


class AgentEventError(BaseModel):
    """An error occurred."""
    event: Literal["error"] = "error"
    message: str


class AgentEventDone(BaseModel):
    """Agent finished processing."""
    event: Literal["done"] = "done"
    total_duration_ms: int


# Union type for all events
AgentEvent = (
    AgentEventThinking 
    | AgentEventToolCall 
    | AgentEventToolResult 
    | AgentEventAnswer 
    | AgentEventError 
    | AgentEventDone
)
