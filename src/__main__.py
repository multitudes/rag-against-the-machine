# src/__main__.py
import os
import fire
import json
import time
import logging
import requests
from tqdm import tqdm
from datetime import datetime
from retrieval.search import Searcher
from src.utils import write_search_to_file
from ingestion.chunking import chunk_content
from ingestion.indexing import create_bm25_index
from ingestion.file_processing import get_all_files
from core.ollama_request import OllamaRequest, Message
from ingestion.file_processing import extract_files_from_questions
from core.schemas import UnansweredQuestion, StudentSearchResults
from core.schemas import StudentSearchResultsAndAnswer, MinimalAnswer

API_URL = "http://localhost:11434/api/chat"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)


class RagCLI:
    """
    CLI for the RAG project.
    The CLI will be called like
    `uv run python -m src index`
    """

    def __init__(self,
                 repo_path="data/raw/vllm-0.10.1",
                 mode="full",
                 #  mode='selective',
                 questions_file=("data/questions.tsv"),
                 search_string="OpenAI compatible server",
                 k=5,
                 search_dataset_path=(
                     "data/datasets/UnansweredQuestions/\
                        Dataset_2025-09-21_valid_unanswered.json"),
                 chunk_size=2000
                 ):
        self.repo_path = repo_path
        self.questions_file = questions_file
        self.mode = mode
        self.search_string = search_string
        self.k = k
        self.search_dataset_path = search_dataset_path
        self.chunk_size = chunk_size
        if chunk_size > 2000:
            raise ValueError("Chunk Size cannot exceed 2000.")

    def index(self):
        """
        Ingest documents from a repository for indexing.
        Uses the instance variables set during initialization.
        """
        logger.info("Ingesting documents...")
        logger.info(f"Repository path: {self.repo_path}")
        logger.info(f"Ingestion mode: {self.mode}")
        logger.info(f"Questions file: {self.questions_file}")

        start_time = time.time()
        try:
            if self.mode == "selective":
                # Load questions to find which files to process
                files_to_process = extract_files_from_questions(
                    self.questions_file)
            else:
                # Get all files in repository

                files_to_process = get_all_files(self.repo_path)

            chunks = []
            for file_path in tqdm(files_to_process, desc="Chunking files"):
                # logger.info(f"Processing file: {file_path}")
                chunks.extend(chunk_content(file_path, self.chunk_size))

            # # 3. Create searchable index (using bm25s)
            create_bm25_index(chunks, "bm25s_indices/")
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
        finally:
            # this finally block always run..
            end_time = time.time()
            duration = end_time-start_time
            print(f"Created index in {duration:.2f} seconds")

    def search(self, search_string=None, k=None):
        """
        Search the indexed documents.
        """
        logger.info("Searching documents...")
        if search_string is None:
            search_string = self.search_string
        if k is None:
            k = self.k
        logger.info(f"Search query: {search_string}")
        logger.info(f"Number of top results to return: {k}")
        search_string = "What command is used to start the\
            vLLM OpenAI-compatible server?"

        try:
            searcher = Searcher(index_dir="bm25s_indices/")
            min_search_res = searcher.search_one(query=search_string, k=k)
            result = StudentSearchResults(
                search_results=[min_search_res],
                k=k
            )
            # Convert the Pydantic model to a pretty-logger.infoed JSON string
            # and logger.info it
            if result:
                logger.info(result.model_dump_json(indent=4))
                write_search_to_file(result, "data/output/search_results")
            else:
                logger.info("No results found.")
        except Exception as e:
            logger.error(f"{e}")

    def search_dataset(self, dataset_path):
        """
        Search using a dataset of questions.
        """
        searcher = Searcher(index_dir="bm25s_indices/")
        if dataset_path is None:
            dataset_path = self.search_dataset_path
        logger.info(f"Searching using dataset: {dataset_path}")
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            questions_data = data.get("rag_questions", [])
            unanswered = [UnansweredQuestion(**item)
                          for item in questions_data]
            result = searcher.search_dataset(unanswered)
            if result:
                write_search_to_file(result, "data/output/search_results")
                logger.info(result.model_dump_json(indent=4))
            else:
                logger.info("No results found.")
            return result
        except Exception as e:
            logger.error(f"Failed collecting questions from dataset: {e}")

    def measure_recall_at_k_on_dataset(self, search_results_path,
                                       ground_truth_path):
        """
        Evaluate search results by measuring recall@k on a dataset.
        Args:
            search_results_path: Path to the search results JSON file
            ground_truth_path: Path to the ground
            truth/answered questions JSON file
        """
        print("📊 Measuring recall@k on dataset...")
        original_level = logger.level
        logger.setLevel(logging.INFO)
        logger.info(f"Search results: {search_results_path}")
        logger.info(f"Ground truth: {ground_truth_path}")
        try:
            with open(search_results_path, 'r', encoding='utf-8') as f:
                search_data = json.load(f)
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                ground_truth_data = json.load(f)

            ground_truth_map = {
                item['question_id']: [source['file_path']
                                      for source in item.get('sources', [])]
                for item in ground_truth_data.get('rag_questions', [])
            }
            total_questions = 0
            total_hits = 0

            for result in search_data.get('search_results', []):
                question_id = result.get('question_id')
                print(f"{question_id}")
                if question_id not in ground_truth_map:
                    logger.warning(
                        f"Question ID {question_id} from search results not found in ground truth. Skipping.")
                    continue

                total_questions += 1

                ground_truth_paths = ground_truth_map[question_id]
                print(f"{ground_truth_paths}")
                retrieved_paths = {source['file_path']
                                   for source in
                                   result.get('retrieved_sources', [])}
                print(f"{retrieved_paths}")
                is_hit = any(gt_path in retrieved_paths for gt_path
                             in ground_truth_paths)
                if is_hit:
                    total_hits += 1
            if total_questions == 0:
                logger.error(
                    "No matching questions found between search results and ground truth.")
                return 0.0
            recall = total_hits / total_questions

            print(f"recall is {recall}")
            logger.info("Recall@k measurement completed!")
        except Exception as e:
            logger.error(f"Error evaluating recall: {e}")
        finally:
            # Always restore the original logging level
            logger.setLevel(original_level)

    def generate(self, output_path=None):
        """
        Generate answers using the RAG system.
        """
        logger.info("Generating answers using RAG...")
        if output_path:
            logger.info(f"Output will be saved to: {output_path}")
        else:
            # Build the filename based on current date
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"Dataset_{date_str}_valid.json"
            # Create the full path
            output_dir = "data/datasets/AnsweredQuestions"
            output_path = os.path.join(output_dir, filename)
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            logger.info("No output path provided, results will be ")
            logger.info(f"saved to: {output_path}")
        # Implement generation logic here
        logger.info("Answer generation completed!")

    def answer(self, question, k=5):
        """
        Answer a single question using the RAG system.

        Args:
            question: The question to answer
            k: The number of top results to return
        """
        logger.info("Answering a question using RAG...")
        logger.info(f"Question: {question}")
        # get context
        searcher = Searcher(index_dir="bm25s_indices/")
        search_results = searcher.search_one(query=question, k=k)
        logger.info(
            f"{search_results.question_id} \n"
            f"{search_results.retrieved_sources}"
        )

        # extract chunks
        context_chunks = []

        for source in search_results.retrieved_sources:
            try:
                with open(source.file_path, 'r', encoding='utf-8') as f:
                    f.seek(source.first_character_index)
                    content = f.read(
                        source.last_character_index
                        - source.first_character_index)
                    context_chunks.append(content)
            except Exception as e:
                logger.error(f"Error reading file {source.file_path}: {e}")
        if not context_chunks:
            logger.error("Could not retrieve any content. Abort")
            return

        logger.info(f"Retrieved {len(context_chunks)} chunks")

        logger.info("Generating answer...")
        context_str = "\n\n---\n\n".join(context_chunks)

        prompt = f"""
        Use the following context to answer the question.
        If the answer is not in the context, say you don't know.

        Context:
        {context_str}

        Question: {question}
        """
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

            # Create MinimalAnswer by combining search results
            # and the new answer
            minimal_answer = MinimalAnswer(
                question_id=search_results.question_id,
                retrieved_sources=search_results.retrieved_sources,
                answer=answer_content
            )

            final_result = StudentSearchResultsAndAnswer(
                search_results=[minimal_answer],
                k=k
            )
            output_dir = "data/output"
            os.makedirs(output_dir, exist_ok=True)
            current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            outputfilename = f"answer_result_{current_date}.json"
            output_path = os.path.join(output_dir, outputfilename)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_result.model_dump_json(indent=4))
            logger.info(f"Answer and sources saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to get answer from LLM: {e}")
        question = UnansweredQuestion(question=question)

        logger.info("Question answered!")


def main():
    """Main entry point for the CLI."""
    logger.info("🤘 Rage Against the Machine - RAG System")
    fire.Fire(RagCLI)


if __name__ == "__main__":
    main()
