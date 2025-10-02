# src/__main__.py
import os
import fire
import json
from datetime import datetime
# Updated imports for the new project structure
from ingestion.file_processing import extract_files_from_questions, get_all_files
from ingestion.indexing import create_bm25_index
from ingestion.chunking import chunk_content
from retrieval.search import Searcher
from core.schemas import UnansweredQuestion, StudentSearchResults
import logging

logger = logging.getLogger(__name__)

class RagCLI:
    """
    CLI for the RAG project.
    The CLI will be called like
    `uv run python -m src index`
    """
    def __init__(self,
                 repo_path="assets/vllm-0.10.1",
                 #mode="full",
                 mode='selective',
                 questions_file=("data/questions.tsv"),
                 search_string="OpenAI compatible server",
                 k=10,
                 search_dataset_path=("data/datasets/UnansweredQuestions/Dataset_2025-09-21_valid_unanswered.json")
                 ):
        self.repo_path = repo_path
        self.questions_file = questions_file
        self.mode = mode
        self.search_string = search_string
        self.k = k
        self.search_dataset_path = search_dataset_path

    def index(self):
        """
        Ingest documents from a repository for indexing.
        Uses the instance variables set during initialization.
        """
        logger.info("Ingesting documents...")
        logger.info(f"Repository path: {self.repo_path}")
        logger.info(f"Ingestion mode: {self.mode}")
        logger.info(f"Questions file: {self.questions_file}")
        try:
            if self.mode == "selective":
                # Load questions to find which files to process
                files_to_process = extract_files_from_questions(self.questions_file)
            else:
                # Get all files in repository
                files_to_process = get_all_files(self.repo_path)

            chunks = []
            for file_path in files_to_process:
                logger.info(f"Processing file: {file_path}")
                chunks += chunk_content(file_path)
            
            # # 3. Create searchable index (using bm25s)
            create_bm25_index(chunks, "bm25s_indices/")

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            return

    def search(self, search_string=None, k=None):
        """
        Search the indexed documents.
        """
        print("Searching documents...")
        if search_string is None:
            search_string = self.search_string
        if k is None:
            k = self.k
        print(f"Search query: {search_string}")
        print(f"Number of top results to return: {k}")
        search_string = "What command is used to start the vLLM OpenAI-compatible server?"

        try:
            searcher = Searcher(index_dir="bm25s_indices/")
            min_search_res = searcher.search_one(query=search_string, k=k)
            result = StudentSearchResults(
               search_results=[min_search_res],
                k=k
            )
            # Convert the Pydantic model to a pretty-printed JSON string 
            # and print it
            if result:
                print(result.model_dump_json(indent=4))
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
            with open(dataset_path, 'r',encoding='utf-8') as f:
                data = json.load(f)
            questions_data = data.get("rag_questions", [])
            unanswered = [UnansweredQuestion(**item) for item in questions_data]
            result = searcher.search_dataset(unanswered)
            if result:
                output_dir = "data/results"
                os.makedirs(output_dir, exist_ok=True)
                current_date = datetime.now().strftime("%Y-%m-%d")
                output_filename = f"search_results_{current_date}.json"
                output_path = os.path.join(output_dir, output_filename)

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(result.model_dump_json(indent=4))
                logger.info(f"Search results saved to {output_path}")
            else:
                logger.info("No results found.")
            return result
        except Exception as e:
            logger.error(f"Failed collecting questions from dataset: {e}")

    def evaluate(self, search_results_path, ground_truth_path):
        """
        Evaluate search results by measuring recall@k on a dataset.

                Args:
            search_results_path: Path to the search results JSON file
            ground_truth_path: Path to the ground
            truth/answered questions JSON file
        """
        print("📊 Measuring recall@k on dataset...")
        print(f"Search results: {search_results_path}")
        print(f"Ground truth: {ground_truth_path}")
        # Implement recall@k evaluation logic here
        print("✅ Recall@k measurement completed!")

    def generate(self, output_path=None):
        """
        Generate answers using the RAG system.
        """
        print("Generating answers using RAG...")
        if output_path:
            print(f"Output will be saved to: {output_path}")
        else:
            # Build the filename based on current date
            current_date = datetime.now()
            date_str = current_date.strftime("%Y-%m-%d")
            filename = f"Dataset_{date_str}_valid.json"
            # Create the full path
            output_dir = "data/datasets/AnsweredQuestions"
            output_path = os.path.join(output_dir, filename)
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            print("No output path provided, results will be ")
            print(f"saved to: {output_path}")
        # Implement generation logic here
        print("✅ Answer generation completed!")

    def answer(self, question, k=5):
        """
        Answer a single question using the RAG system.

        Args:
            question: The question to answer
            k: The number of top results to return
        """
        print("Answering a question using RAG...")
        print(f"Question: {question}")
        # Implement single question answering logic here
        print("✅ Question answered!")


def main():
    """Main entry point for the CLI."""
    print("🤘 Rage Against the Machine - RAG System")
    fire.Fire(RagCLI)


if __name__ == "__main__":
    main()
