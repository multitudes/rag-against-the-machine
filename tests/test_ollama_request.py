"""Tests for Ollama request payload models."""

from core.ollama_request import Message, OllamaRequest


def test_message_fields() -> None:
    """Message stores role and content."""
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_ollama_request_defaults() -> None:
    """OllamaRequest defaults stream and think to False."""
    req = OllamaRequest(
        model="qwen3:0.6b",
        messages=[Message(role="user", content="q")],
        tools=[],
    )
    assert req.stream is False
    assert req.think is False
    dumped = req.model_dump()
    assert dumped["model"] == "qwen3:0.6b"
    assert dumped["messages"][0]["content"] == "q"
