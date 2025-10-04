from pydantic import BaseModel
from typing import List, Dict, Any


# The message payload for the API request to Ollama
class Message(BaseModel):
    role: str
    content: str


class OllamaRequest(BaseModel):
    model: str
    messages: List[Message]
    tools: List[Dict[str, Any]]
    stream: bool = False
    think: bool = False
