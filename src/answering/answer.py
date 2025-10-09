
import logging
import requests
from retrieval.search import Searcher
from core.ollama_request import OllamaRequest, Message
from core.schemas import MinimalAnswer, UnansweredQuestion

API_URL = "http://localhost:11434/api/chat"

logger = logging.getLogger(__name__)


def calling_llm(prompt: str) -> str:
    """
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
            API_URL,
            data=data.model_dump_json(),
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        response_data = response.json()
        answer_content = response_data['message']['content']

        logger.info("\nAnswer:\n")
        logger.info(answer_content)
        return answer_content
    except Exception as e:
        logger.error(f"Could not generate answer : {e}")
        return answer_content


def create_prompt(context_str: str, question: str) -> str:
    """
    """
    prompt = f"""
    Use the following context to answer the question.
    If the answer is not in the context, say you don't know.

    Context:
    {context_str}

    Question: {question}
    """
    return prompt


def get_answer(unansweredQuestion: UnansweredQuestion, k: int, ):
    logger.info("Answering a question using RAG...")
    logger.info(f"Question: {unansweredQuestion.question}")
    # get context
    searcher = Searcher(index_dir="bm25s_indices/")
    search_results = searcher.search_one(unansweredQuestion=unansweredQuestion,
                                         k=k)
    logger.info(
        f"{search_results.question_id} \n"
        f"{search_results.retrieved_sources}"
    )
    # extract chunks
    context_chunks = searcher.retrieve_context(search_results)
    logger.info(f"Retrieved {len(context_chunks)} chunks")
    context_str = "\n\n---\n\n".join(context_chunks)

    prompt = create_prompt(context_str, unansweredQuestion.question)
    answer_content = calling_llm(prompt)

    # Create MinimalAnswer by combining search results
    # and the new answer
    minimal_answer = MinimalAnswer(
        question_id=unansweredQuestion.question_id,
        retrieved_sources=search_results.retrieved_sources,
        answer=answer_content
    )
    return minimal_answer
