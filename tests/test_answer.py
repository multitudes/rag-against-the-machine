"""Tests for answering.answer (network calls are mocked)."""

from unittest.mock import MagicMock, patch

from answering.answer import (
    answer_from_search_result,
    calling_llm,
    create_prompt,
    get_answer,
)
from core.schemas import MinimalSearchResults, UnansweredQuestion


def test_create_prompt_includes_context_and_question() -> None:
    """create_prompt embeds the context and the question."""
    prompt = create_prompt("CTX", "What is RAG?")
    assert "CTX" in prompt
    assert "What is RAG?" in prompt
    assert "Context:" in prompt
    assert "Question:" in prompt


def test_calling_llm_success_is_mocked() -> None:
    """calling_llm returns the mocked Ollama message content."""
    fake = MagicMock()
    fake.raise_for_status.return_value = None
    fake.json.return_value = {"message": {"content": "ok"}}
    with patch("answering.answer.requests.post", return_value=fake):
        assert calling_llm("prompt") == "ok"


def test_calling_llm_failure_returns_empty() -> None:
    """calling_llm returns '' when the request raises."""
    with patch(
        "answering.answer.requests.post",
        side_effect=OSError("offline"),
    ):
        assert calling_llm("prompt") == ""


def test_answer_from_search_result_mocked(
    sample_search_result: MinimalSearchResults,
) -> None:
    """answer_from_search_result uses mocked context and LLM."""
    with (
        patch(
            "answering.answer.retrieve_context_from_sources",
            return_value=["ctx"],
        ),
        patch("answering.answer.calling_llm", return_value="ans"),
    ):
        result = answer_from_search_result(sample_search_result)
    assert result.answer == "ans"
    assert result.question_id == sample_search_result.question_id
    assert result.retrieved_sources == (
        sample_search_result.retrieved_sources
    )


def test_get_answer_mocked(
    sample_question: UnansweredQuestion,
    sample_search_result: MinimalSearchResults,
) -> None:
    """get_answer searches then answers without hitting the network."""
    fake_searcher = MagicMock()
    fake_searcher.search_one.return_value = sample_search_result
    with (
        patch("answering.answer.Searcher", return_value=fake_searcher),
        patch(
            "answering.answer.answer_from_search_result",
            return_value=MagicMock(answer="stub"),
        ) as mock_answer,
    ):
        get_answer(sample_question, k=3, index_dir="data/processed")
    fake_searcher.search_one.assert_called_once()
    mock_answer.assert_called_once_with(sample_search_result)
