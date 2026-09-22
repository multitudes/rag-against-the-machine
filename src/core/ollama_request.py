from typing import Any

from pydantic import BaseModel


class Message(BaseModel):
    """
    A single chat message payload."""

    role: str
    content: str


class OllamaRequest(BaseModel):
    """
    Request body for the Ollama /api/chat endpoint."""

    model: str
    messages: list[Message]
    tools: list[dict[str, Any]]
    stream: bool = False
    think: bool = False
