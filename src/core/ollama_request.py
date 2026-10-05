"""
Pydantic models for Ollama API request payloads.
"""

from typing import Any

from pydantic import BaseModel


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

class Message(BaseModel):
    """
    A single chat message payload.

    Attributes:
        role: Message role (e.g. ``"user"``, ``"system"``).
        content: Message text content.

    """

    role: str
    content: str


class OllamaRequest(BaseModel):
    """
    Request body for the Ollama /api/chat endpoint.

    Attributes:
        model: Ollama model name to query.
        messages: Conversation messages to send.
        tools: Optional tool definitions for function calling.
        stream: Whether to stream the response.
        think: Whether to enable model thinking mode.

    """

    model: str
    messages: list[Message]
    tools: list[dict[str, Any]]
    stream: bool = False
    think: bool = False
