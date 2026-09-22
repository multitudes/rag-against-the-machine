"""
Pydantic data models for the RAG pipeline.

These schemas define the JSON shapes exchanged between CLI commands
and validated by the moulinette.
"""

import uuid

from pydantic import BaseModel, Field


# Core Models
class MinimalSource(BaseModel):
    """
    Represents a minimal source of information.

    Attributes:
        file_path: Path to the source file.
        first_character_index: Inclusive start offset in the file.
        last_character_index: Exclusive end offset in the file.

    """

    file_path: str
    first_character_index: int
    last_character_index: int


class ChunkSource(BaseModel):
    """
    Represents a chunk of text and its minimal source.

    Attributes:
        text: The chunk text content.
        source: Location of the chunk in the original file.

    """

    text: str
    source: MinimalSource


# Question Models
class UnansweredQuestion(BaseModel):
    """
    Represents an unanswered question.

    Attributes:
        question_id: Unique identifier for the question.
        question: The question text.

    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
    Represents an answered question with sources.

    Attributes:
        sources: Ground-truth source locations for the answer.
        answer: The ground-truth answer text.

    """

    sources: list[MinimalSource]
    answer: str


# Dataset Models
class RagDataset(BaseModel):
    """
    Represents a dataset of RAG questions.

    Attributes:
        rag_questions: List of answered or unanswered questions.

    """

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


# Search Result Models
class MinimalSearchResults(BaseModel):
    """
    Represents the search results for a question.

    Attributes:
        question_id: Unique identifier for the question.
        question: The question text.
        retrieved_sources: Top-k sources returned by retrieval.

    """

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """
    Represents search results with an answer.

    Attributes:
        answer: Generated answer text from the LLM.

    """

    answer: str


# Student Models
class StudentSearchResults(BaseModel):
    """
    Represents student search results.

    Attributes:
        search_results: List of per-question search results.
        k: Number of sources retrieved per question.

    """

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """
    Represents student search results with answers.

    Attributes:
        search_results: List of per-question results with answers.
        k: Number of sources retrieved per question.

    """

    search_results: list[MinimalAnswer]
    k: int
