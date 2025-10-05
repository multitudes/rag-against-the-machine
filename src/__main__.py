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
from answering.answer import get_answer
from src.utils import write_search_to_file, calculate_overlap_percentage
from ingestion.chunking import chunk_content
from ingestion.indexing import create_bm25_index
from ingestion.file_processing import get_all_files
from ingestion.file_processing import extract_files_from_questions
from core.schemas import UnansweredQuestion, StudentSearchResults
from core.schemas import StudentSearchResultsAndAnswer

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
        # search_string = "What command is used to start the\
        #     vLLM OpenAI-compatible server?"

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
        original_level = logger.level
        logger.setLevel(logging.INFO)
        
        try:
            logger.info("📊 Measuring recall@k on dataset (with overlap)...")
            logger.info(f"Search results: {search_results_path}")
            logger.info(f"Ground truth: {ground_truth_path}")
            
            with open(search_results_path, 'r', encoding='utf-8') as f:
                search_data = json.load(f)
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                ground_truth_data = json.load(f)

            ground_truth_map = {
                item['question_id']: item.get('sources', [])
                for item in ground_truth_data.get('rag_questions', [])
            }
            
            # for each question I need to get the recall, append and at 
            # the end do the average
            quest_recall_scores = []
            # go through my results and get the question_id to match it to 
            # the ground_truth ones
            for result in search_data.get('search_results', []):
                question_id = result.get('question_id')
                if question_id not in ground_truth_map:
                    logger.warning(f"Question ID {question_id} from \
                                   earch results not found in ground\
                                   truth. Skipping.")
                    continue
                # these are the sources in the reference ground truth but 
                # in this example it is actually mostly an array of one
                ground_truth_sources = ground_truth_map[question_id]
                # my sources are depending of the k variable when searching
                retrieved_sources = result.get('retrieved_sources', [])
                
                # cannot continue without my ref
                if not ground_truth_sources:
                    continue

                number_found = 0
                for gt_source in ground_truth_sources:
                    is_found = False
                    for ret_source in retrieved_sources:
                        if gt_source['file_path'] == ret_source['file_path']:
                            overlap = calculate_overlap_percentage(
                                gt_source['first_character_index'],
                                gt_source['last_character_index'],
                                ret_source['first_character_index'],
                                ret_source['last_character_index']
                            )
                            if overlap >= 5.0:
                                is_found = True
                                # because I dont need to find two same sources
                                break
                    # catching the break above
                    if is_found:
                        number_found += 1
                
                recall_for_question = number_found / len(ground_truth_sources)
                quest_recall_scores.append(recall_for_question)

            if not quest_recall_scores:
                logger.error("No matching questions found to evaluate.")
                return 0.0

            # The final recall is the average of the recall over all questions
            final_recall = sum(quest_recall_scores) / len(quest_recall_scores)
            
            logger.info(f"Final Recall Score (with >=5% overlap): {final_recall:.2%}")
            logger.info("Recall@k measurement completed!")
            return final_recall
            
        except Exception as e:
            logger.error(f"Error evaluating recall: {e}", exc_info=True)
            return 0.0
        finally:
            logger.setLevel(original_level)

    def answer(self, question, k=5):
        """
        Answer a single question using the RAG system.

        Args:
            question: The question to answer
            k: The number of top results to return
        """
        minimal_answer = get_answer(question, k)

        final_result = StudentSearchResultsAndAnswer(
            search_results=[minimal_answer],
            k=k
        )
        try:
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

    def answer_dataset(self, output_path=None):
        """
        Generate answers using the RAG system.
        """
        logger.info("Generating answers using RAG...")
        # Build the filename based on current date
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"Dataset_{date_str}_valid.json"
        # Create the full path
        output_dir = "data/output/search_results/"
        output_path = os.path.join(output_dir, filename)
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"results will be saved to: {output_path}")
        
        logger.info("Answer generation completed!")


def main():
    """Main entry point for the CLI."""
    logger.info("🤘 Rage Against the Machine - RAG System")
    fire.Fire(RagCLI)


if __name__ == "__main__":
    main()
