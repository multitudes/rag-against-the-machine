# src/__main__.py
import os
import fire
import time
import logging
import requests
from tqdm import tqdm
from datetime import datetime
from retrieval.search import Searcher
from answering.answer import get_answer
from core.config import OLLAMA_HEALTH_URL
from src.utils import write_search_to_file, calculate_overlap_percentage
from src.utils import save_search_results_and_answer_to_json
from ingestion.chunking import chunk_content
from ingestion.indexing import create_bm25_index
from ingestion.file_processing import get_all_files
from ingestion.file_processing import extract_files_from_questions
from core.schemas import UnansweredQuestion, StudentSearchResults
from core.schemas import StudentSearchResultsAndAnswer, RagDataset
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
        Fire CLI - Ingest documents from a repository for indexing.
        Uses the instance variables set during initialization.
        """
        logger.debug("Ingesting documents...")
        logger.debug(f"Repository path: {self.repo_path}")
        logger.debug(f"Ingestion mode: {self.mode}")
        logger.debug(f"Questions file: {self.questions_file}")

        # Validate that the repository path exists
        if not os.path.exists(self.repo_path):
            logger.error(f"Repository path does not exist: {self.repo_path}")
            return

        start_time = time.time()
        original_level = logger.level
        logger.setLevel(logging.ERROR)
        try:
            if self.mode == "selective":
                # Load questions to find which files to process
                if not os.path.exists(self.questions_file):
                    logger.error(
                        f"File does not exist: {self.questions_file}")
                    return
                files_to_process = extract_files_from_questions(
                    self.questions_file)
            else:
                # Get all files in repository
                files_to_process = get_all_files(self.repo_path)

            if not files_to_process:
                logger.warning("No files found to process")
                return

            chunks = []
            for file_path in tqdm(files_to_process, desc="Chunking files"):
                # logger.debug(f"Processing file: {file_path}")
                chunks.extend(chunk_content(file_path, self.chunk_size))

            if not chunks:
                logger.warning("No chunks created from files")
                return

            create_bm25_index(chunks, "bm25s_indices/")

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")

        finally:
            # this finally block always run..
            logger.setLevel(original_level)
            end_time = time.time()
            duration = end_time-start_time
            print(f"Created index in {duration:.2f} seconds")

    def search(self, search_string=None, k=None):
        """
        Search the indexed documents.
        """
        write_path = "data/output/search_results"
        logger.debug("Searching documents...")
        if search_string is None:
            search_string = self.search_string
        if k is None:
            k = self.k
        logger.debug(f"Search query: {search_string}")
        logger.debug(f"Number of top results to return: {k}")

        # Check if index directory exists
        if not os.path.exists("bm25s_indices/"):
            logger.error("Index directory not found. Please run 'ingest' "
                         "command first.")
            return

        try:
            searcher = Searcher(index_dir="bm25s_indices/")
            # Create an UnansweredQuestion object from the search string
            unanswered_question = UnansweredQuestion(question=search_string)
            min_search_res = searcher.search_one(
                unansweredQuestion=unanswered_question, k=k)
            result = StudentSearchResults(
                search_results=[min_search_res],
                k=k
            )
            # Convert the Pydantic model to a pretty-logger.debuged JSON string
            # and logger.debug it
            if result:
                # logger.debug(result.model_dump_json(indent=4))
                write_search_to_file(result, write_path)
            else:
                logger.debug("No results found.")
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            logger.info(f"Saved to file {write_path}")
        except FileNotFoundError as e:
            logger.error(f"Index files not found: {e}")
            logger.error("Please run 'ingest' command first.")
        except Exception as e:
            logger.error(f"Search failed: {e}")

    def search_dataset(self, dataset_path=None):
        """
        Search using a dataset of questions.
        """
        output_path = "data/output/search_results"
        if dataset_path is None:
            dataset_path = self.search_dataset_path

        # Check if index directory exists
        if not os.path.exists("bm25s_indices/"):
            logger.error("Index directory not found. Please run 'ingest' "
                         "command first.")
            return

        # Check if dataset file exists
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found: {dataset_path}")
            return

        logger.info(f"Searching using dataset: {dataset_path}")
        try:
            searcher = Searcher(index_dir="bm25s_indices/")
            with open(dataset_path, 'r', encoding='utf-8') as f:
                unanswered = RagDataset.model_validate_json(f.read())
            result = searcher.search_dataset(unanswered.rag_questions)
            if result:
                write_search_to_file(result, output_path)
            else:
                logger.debug("No results found.")
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            logger.info(f"Saved to file : {output_path}")
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
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
        # Validate input files exist
        if not os.path.exists(search_results_path):
            logger.error(f"Search results file not found: "
                         f"{search_results_path}")
            return 0.0

        if not os.path.exists(ground_truth_path):
            logger.error(f"Ground truth file not found: {ground_truth_path}")
            return 0.0

        original_level = logger.level
        logger.setLevel(logging.INFO)

        try:
            logger.debug("📊 Measuring recall@k on dataset (with overlap)...")
            logger.debug(f"Search results: {search_results_path}")
            logger.debug(f"Ground truth: {ground_truth_path}")

            with open(search_results_path, 'r', encoding='utf-8') as f:
                search_data = StudentSearchResults.model_validate_json(
                    f.read())
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                ground_truth_data = RagDataset.model_validate_json(f.read())

            # Build a map of question_id to sources from ground truth
            ground_truth_map = {}
            for item in ground_truth_data.rag_questions:
                # Check if item has sources (AnsweredQuestion)
                if hasattr(item, 'sources'):
                    ground_truth_map[item.question_id] = item.sources
                else:
                    ground_truth_map[item.question_id] = []

            # for each question I need to get the recall, append and at
            # the end do the average
            quest_recall_scores = []
            # go through my results and get the question_id to match it to
            # the ground_truth ones
            for result in search_data.search_results:
                question_id = result.question_id
                if question_id not in ground_truth_map:
                    logger.warning(f"Question ID {question_id} from "
                                   f"search results not found in ground "
                                   f"truth. Skipping.")
                    continue
                # these are the sources in the reference ground truth but
                # in this example it is actually mostly an array of one
                ground_truth_sources = ground_truth_map[question_id]
                # my sources are depending of the k variable when searching
                retrieved_sources = result.retrieved_sources

                # cannot continue without my ref
                if not ground_truth_sources:
                    continue

                number_found = 0
                for gt_source in ground_truth_sources:
                    is_found = False
                    for ret_source in retrieved_sources:
                        if gt_source.file_path == ret_source.file_path:
                            overlap = calculate_overlap_percentage(
                                gt_source.first_character_index,
                                gt_source.last_character_index,
                                ret_source.first_character_index,
                                ret_source.last_character_index
                            )
                            if overlap >= 5.0:
                                is_found = True
                                # because I dont need two same sources
                                break
                    # catching the break above
                    if is_found:
                        number_found += 1

                recall_for_question = (
                    number_found / len(ground_truth_sources)
                )
                quest_recall_scores.append(recall_for_question)

            if not quest_recall_scores:
                logger.error("No matching questions found to evaluate.")
                return 0.0

            # The final recall is the average of recall over all questions
            final_recall = (
                sum(quest_recall_scores) / len(quest_recall_scores)
            )

            logger.info(f"Final Recall Score (with >=5% overlap): "
                        f"{final_recall:.2%}")
            logger.info("Recall@k measurement completed!")

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Error evaluating recall: {e}", exc_info=True)
            return 0.0
        finally:
            logger.setLevel(original_level)

    def answer_one(self, question, k=5):
        """
        Answer a single question using the RAG system.
        Args:
            question (str): The question to answer.
            k (int): The number of top results to return.
        """
        # Check if Ollama is running before processing the dataset
        try:
            response = requests.get(
                OLLAMA_HEALTH_URL, timeout=2)
            response.raise_for_status()
        except requests.exceptions.RequestException:
            logger.error(
                "Ollama is not running or not accessible")
            logger.error("Please start Ollama before running this command.")
            return
        # Check if index directory exists
        if not os.path.exists("bm25s_indices/"):
            logger.error("Index directory not found. Please run 'ingest' "
                         "command first.")
            return

        try:
            unansweredQuestion = UnansweredQuestion(question=question)
            minimal_answer = get_answer(unansweredQuestion, k)

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
            logger.setLevel(logging.INFO)
            logger.info(f"Answer and sources saved to {output_path}")
            logger.info("Question answered!")
        except FileNotFoundError as e:
            logger.error(f"Required files not found: {e}")
            logger.error("Please run 'ingest' command first.")
        except Exception as e:
            logger.error(f"Failed to get answer from LLM: {e}")

    def answer_dataset(self, output_path=None):
        """
        Generate answers for a dataset of questions using the RAG system.
        Args:
            output_path (str, optional): Path to save
            the answers. Defaults to None.
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        # Check if Ollama is running before processing the dataset
        try:
            response = requests.get(
                OLLAMA_HEALTH_URL, timeout=2)
            response.raise_for_status()
        except requests.exceptions.RequestException:
            logger.error(
                "Ollama is not running or not accessible")
            logger.error("Please start Ollama before running this command.")
            return
        logger.debug("Generating answers using RAG...")
        dataset_path = (
            "data/datasets/UnansweredQuestions/"
            "Dataset_2025-09-21_valid_unanswered.json"
        )

        # Check if index directory exists
        if not os.path.exists("bm25s_indices/"):
            logger.error("Index directory not found. Please run 'ingest' "
                         "command first.")
            return

        # Check if dataset file exists
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found at: {dataset_path}")
            return

        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = RagDataset.model_validate_json(f.read())
        except Exception as e:
            logger.error(f"Failed to parse dataset file: {e}")
            return

        start_time = time.time()
        minimal_answers = []
        original_level = logger.level
        logger.setLevel(logging.ERROR)
        try:
            for question in tqdm(dataset.rag_questions):
                minimal_answer = get_answer(question, k=self.k)
                minimal_answers.append(minimal_answer)
            # Structure the final results using the appropriate Pydantic
            # model
            final_result = StudentSearchResultsAndAnswer(
                search_results=minimal_answers,
                k=self.k
            )
            save_search_results_and_answer_to_json(final_result)
        except Exception as e:
            logger.error("Failed to generate answer for questions")
            logger.error(f"'{question.question}': {e}")
        finally:
            logger.setLevel(original_level)
            duration = time.time() - start_time
            logger.debug(f"Answered {len(minimal_answers)} questions ")
            logger.debug(f"in {duration:.2f}s")


def main():
    """
    Main entry point for the CLI.
    Initializes and runs the RagCLI using Fire.
    """
    try:
        logger.debug("🤘 Rage Against the Machine - RAG System")
        fire.Fire(RagCLI)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")


if __name__ == "__main__":
    main()
