"""Tests for Pydantic models in core.schemas."""

from core.schemas import (
    AnsweredQuestion,
    ChunkSource,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)


def test_minimal_source_roundtrip() -> None:
    """MinimalSource serialises and validates the same fields."""
    src = MinimalSource(
        file_path="a.py",
        first_character_index=0,
        last_character_index=4,
    )
    restored = MinimalSource.model_validate(src.model_dump())
    assert restored.file_path == "a.py"
    assert restored.first_character_index == 0
    assert restored.last_character_index == 4


def test_chunk_source_embeds_minimal_source() -> None:
    """ChunkSource keeps text and source together."""
    chunk = ChunkSource(
        text="code",
        source=MinimalSource(
            file_path="a.py",
            first_character_index=0,
            last_character_index=4,
        ),
    )
    assert chunk.text == "code"
    assert chunk.source.file_path == "a.py"


def test_unanswered_question_generates_id() -> None:
    """question_id is auto-generated when omitted."""
    q = UnansweredQuestion(question="hello?")
    assert q.question == "hello?"
    assert q.question_id


def test_answered_question_requires_sources() -> None:
    """AnsweredQuestion stores sources and the ground-truth answer."""
    q = AnsweredQuestion(
        question="hello?",
        sources=[
            MinimalSource(
                file_path="a.py",
                first_character_index=0,
                last_character_index=1,
            ),
        ],
        answer="hi",
    )
    assert q.answer == "hi"
    assert len(q.sources) == 1


def test_rag_dataset_accepts_mixed_questions() -> None:
    """RagDataset accepts answered and unanswered questions."""
    dataset = RagDataset(
        rag_questions=[
            UnansweredQuestion(question="u?"),
            AnsweredQuestion(
                question="a?",
                sources=[],
                answer="yes",
            ),
        ],
    )
    assert len(dataset.rag_questions) == 2


def test_search_and_answer_models() -> None:
    """Student result wrappers store k and nested results."""
    src = MinimalSource(
        file_path="a.py",
        first_character_index=0,
        last_character_index=1,
    )
    result = MinimalSearchResults(
        question_id="1",
        question="q",
        retrieved_sources=[src],
    )
    answer = MinimalAnswer(
        question_id="1",
        question="q",
        retrieved_sources=[src],
        answer="a",
    )
    student = StudentSearchResults(search_results=[result], k=5)
    with_answers = StudentSearchResultsAndAnswer(
        search_results=[answer],
        k=5,
    )
    assert student.k == 5
    assert with_answers.search_results[0].answer == "a"
