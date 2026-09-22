"""
Answer generation module for the RAG system."""

import logging

import requests

from core.config import OLLAMA_API_URL
from core.ollama_request import Message, OllamaRequest
from core.schemas import MinimalAnswer, MinimalSearchResults, UnansweredQuestion
from retrieval.search import Searcher, retrieve_context_from_sources

logger = logging.getLogger(__name__)


def calling_llm(prompt: str) -> str:
    """
    Send a prompt to the LLM and return the response text.

    Args:
        prompt: The full prompt string to send.

    Returns:
        The model's response content, or empty string on failure.

    """
    answer_content = ""
    messages = [Message(role="user", content=prompt)]
    try:
        data = OllamaRequest(
            model="qwen3:0.6b",
            messages=messages,
            tools=[],
            stream=False,
        )
        response = requests.post(
            OLLAMA_API_URL,
            data=data.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        response_data = response.json()
        answer_content = str(response_data["message"]["content"])
        logger.debug("Answer:\n%s", answer_content)
        return answer_content
    except Exception:
        logger.exception("Could not generate answer")
        return answer_content


def create_prompt(context_str: str, question: str) -> str:
    """
    Build the RAG prompt from context and question.

    Args:
        context_str: Concatenated context chunks.
        question: The user question.

    Returns:
        Formatted prompt string.

    """
    prompt = (
        "Use the following context to answer the question.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}"
    )
    return prompt


def answer_from_search_result(
    search_result: MinimalSearchResults,
) -> MinimalAnswer:
    """
    Generate an answer from a pre-computed MinimalSearchResults.

    Reads the source files for context and calls the LLM.
    Does NOT perform a new BM25 search.

    Args:
        search_result: A MinimalSearchResults with question and sources.

    Returns:
        MinimalAnswer with question_id, question, retrieved_sources, answer.

    """
    context_chunks: list[str] = retrieve_context_from_sources(search_result)
    context_str = "\n\n---\n\n".join(context_chunks)
    prompt = create_prompt(context_str, search_result.question)
    answer_content = calling_llm(prompt)
    return MinimalAnswer(
        question_id=search_result.question_id,
        question=search_result.question,
        retrieved_sources=search_result.retrieved_sources,
        answer=answer_content,
    )


def get_answer(
    unansweredQuestion: UnansweredQuestion,
    k: int,
    index_dir: str = "data/processed",
) -> MinimalAnswer:
    """
    Full single-question pipeline: search → retrieve context → generate answer.

    Args:
        unansweredQuestion: The question to answer.
        k: Number of sources to retrieve.
        index_dir: Path to the BM25 index directory.

    Returns:
        MinimalAnswer with sources and generated answer.

    """
    logger.debug("Answering: '%s'", unansweredQuestion.question)
    searcher = Searcher(index_dir=index_dir)
    search_result = searcher.search_one(
        unansweredQuestion=unansweredQuestion,
        k=k,
    )
    return answer_from_search_result(search_result)
