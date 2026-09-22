import uuid

from pydantic import BaseModel, Field


# Core Models
class MinimalSource(BaseModel):
    """Represents a minimal source of information."""

    file_path: str
    first_character_index: int
    last_character_index: int


class ChunkSource(BaseModel):
    """Represents a chunk of text and its minimal source."""

    text: str
    source: MinimalSource


# Question Models
class UnansweredQuestion(BaseModel):
    """Represents an unanswered question."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represents an answered question with sources."""

    sources: list[MinimalSource]
    answer: str


# Dataset Models
class RagDataset(BaseModel):
    """Represents a dataset of RAG questions."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


# Search Result Models
class MinimalSearchResults(BaseModel):
    """Represents the search results for a question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Represents search results with an answer."""

    answer: str


# Student Models
class StudentSearchResults(BaseModel):
    """Represents student search results."""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Represents student search results with answers."""

    search_results: list[MinimalAnswer]
    k: int
