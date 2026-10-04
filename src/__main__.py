"""Main entry point for the RAG CLI."""

import logging
import time
from pathlib import Path

import fire
import requests
from tqdm import tqdm

from answering.answer import answer_from_search_result, get_answer
from api.server import run_server
from core.config import (
    API_DEFAULT_HOST,
    API_DEFAULT_PORT,
    MAX_CHUNK_SIZE,
    OLLAMA_HEALTH_URL,
)
from core.schemas import (
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)
from ingestion.file_processing import get_all_files
from ingestion.incremental import (
    collect_chunks,
    write_file_manifest,
)
from ingestion.indexing import create_bm25_index
from retrieval.cache import (
    clear_index_caches,
    get_cached_searcher,
    lookup_query,
    store_query,
)
from retrieval.search import Searcher
from retrieval.semantic import (
    create_semantic_index,
    embeddings_path,
    merge_semantic_index,
)
from utils import calculate_overlap_percentage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_INDEX_DIR = "data/processed"
DEFAULT_REPO_PATH = "data/raw/vllm-0.10.1"


def _print_sources(sources: list[MinimalSource]) -> None:
    """
    Print source locations in the single-search CLI format.

    Args:
        sources: Retrieved MinimalSource rows.

    Returns:
        None.

    """
    for source in sources:
        print(
            f"{source.file_path} "
            f"[{source.first_character_index}:"
            f"{source.last_character_index}]",
        )


class RagCLI:
    """
    CLI for the RAG (Retrieval-Augmented Generation) system.

    Every command is invoked as:
        uv run python -m src <command> [options]
    """

    # ------------------------------------------------------------------
    # index
    # ------------------------------------------------------------------

    def index(
        self,
        max_chunk_size: int = MAX_CHUNK_SIZE,
        repo_path: str = DEFAULT_REPO_PATH,
        index_dir: str = DEFAULT_INDEX_DIR,
        semantic: bool = False,
        incremental: bool = False,
    ) -> None:
        """
        Ingest data/raw/ and build the BM25 index under data/processed/.

        Args:
            max_chunk_size: Maximum characters per chunk (default 2000).
            repo_path: Root directory of the corpus to index.
            index_dir: Output directory for the index files.
            semantic: If True, also build a MiniLM vector index.
            incremental: If True, re-chunk only files that changed.

        """
        if max_chunk_size > MAX_CHUNK_SIZE:
            logger.error(
                "max_chunk_size cannot exceed %d "
                "(moulinette rejects longer sources).",
                MAX_CHUNK_SIZE,
            )
            return
        if max_chunk_size <= 0:
            logger.error("max_chunk_size must be a positive integer.")
            return
        if not Path(repo_path).exists():
            logger.error("Repository path does not exist: %s", repo_path)
            return

        logger.info(
            "Indexing corpus at '%s' "
            "(max_chunk_size=%d incremental=%s) …",
            repo_path,
            max_chunk_size,
            incremental,
        )
        start_time = time.time()
        try:
            files_to_process = get_all_files(repo_path)
            if not files_to_process:
                logger.warning("No files found to process.")
                return

            result = collect_chunks(
                files_to_process,
                max_chunk_size,
                index_dir,
                incremental,
            )
            if result.nothing_changed:
                duration = time.time() - start_time
                print(
                    f"Index already up to date under "
                    f"{index_dir} ({duration:.1f}s)",
                )
                return
            if not result.chunks:
                logger.warning("No chunks created from files.")
                return

            create_bm25_index(result.chunks, index_dir)
            emb_path = embeddings_path(index_dir)
            if result.old_chunks and emb_path.exists():
                merge_semantic_index(
                    result.old_chunks,
                    result.chunks,
                    result.unchanged_files,
                    index_dir,
                )
            elif semantic:
                create_semantic_index(
                    [chunk.text for chunk in result.chunks],
                    index_dir,
                )
            write_file_manifest(
                files_to_process,
                max_chunk_size,
                index_dir,
            )
            clear_index_caches(index_dir)

        except Exception:
            logger.exception("Indexing failed")
            return

        duration = time.time() - start_time
        print(
            f"Ingestion complete! Indices saved under "
            f"{index_dir} ({duration:.1f}s)",
        )

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 5,
        index_dir: str = DEFAULT_INDEX_DIR,
        semantic: bool = False,
        hybrid: bool = False,
        cache: bool = False,
    ) -> None:
        """
        Return the top-k sources for a single query.

        Args:
            query: The search query string.
            k: Number of results to return (default 5).
            index_dir: Path to the BM25 index directory.
            semantic: If True, rank with MiniLM instead of BM25.
            hybrid: If True, fuse BM25 and MiniLM (wins over semantic).
            cache: If True, reuse cached index and query results.

        """
        if not query or not query.strip():
            logger.error("Query cannot be empty.")
            return
        if k <= 0:
            logger.error("k must be a positive integer.")
            return
        if not Path(index_dir).exists():
            logger.error(
                "Index directory '%s' not found. Please run 'index' first.",
                index_dir,
            )
            return

        try:
            if cache:
                cached = lookup_query(
                    query,
                    k,
                    index_dir,
                    semantic,
                    hybrid,
                )
                if cached is not None:
                    _print_sources(cached)
                    return
            if cache:
                searcher = get_cached_searcher(index_dir)
            else:
                searcher = Searcher(index_dir=index_dir)
            unanswered = UnansweredQuestion(question=query)
            result = searcher.search_one(
                unanswered_question=unanswered,
                k=k,
                semantic=semantic,
                hybrid=hybrid,
            )
            if cache:
                store_query(
                    query,
                    k,
                    index_dir,
                    semantic,
                    hybrid,
                    result.retrieved_sources,
                )
            _print_sources(result.retrieved_sources)
        except FileNotFoundError:
            logger.exception("Index files not found")
        except Exception:
            logger.exception("Search failed")

    # ------------------------------------------------------------------
    # search_dataset
    # ------------------------------------------------------------------

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        index_dir: str = DEFAULT_INDEX_DIR,
    ) -> None:
        """
        Run search over a dataset, write a StudentSearchResults JSON.

        Args:
            dataset_path: Path to the UnansweredQuestions JSON dataset.
            k: Number of results per question (default 10).
            save_directory: Directory to save the output JSON file.
            index_dir: Path to the BM25 index directory.

        """
        if not Path(index_dir).exists():
            logger.error(
                "Index directory '%s' not found. Please run 'index' first.",
                index_dir,
            )
            return
        if not Path(dataset_path).exists():
            logger.error("Dataset file not found: %s", dataset_path)
            return
        if k <= 0:
            logger.error("k must be a positive integer.")
            return

        logger.info("Searching dataset '%s' with k=%d …", dataset_path, k)
        try:
            with Path(dataset_path).open(encoding="utf-8") as f:
                dataset = RagDataset.model_validate_json(f.read())

            questions = [
                q for q in dataset.rag_questions
                if isinstance(q, UnansweredQuestion)
            ]

            searcher = Searcher(index_dir=index_dir)
            results_list = []
            for question in tqdm(questions, desc="Searching questions"):
                res = searcher.search_one(unanswered_question=question, k=k)
                results_list.append(res)
            result = StudentSearchResults(search_results=results_list, k=k)

            Path(save_directory).mkdir(parents=True, exist_ok=True)
            output_path = Path(save_directory) / Path(dataset_path).name
            with Path(output_path).open("w", encoding="utf-8") as f:
                f.write(result.model_dump_json(indent=4))
            print(f"Saved student_search_results to {output_path}")

        except FileNotFoundError:
            logger.exception("File not found")
        except Exception:
            logger.exception("search_dataset failed")

    # ------------------------------------------------------------------
    # answer (single query)
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        k: int = 5,
        index_dir: str = DEFAULT_INDEX_DIR,
    ) -> None:
        """
        Answer a single query using the retrieved context.

        Args:
            query: The question to answer.
            k: Number of sources to retrieve (default 5).
            index_dir: Path to the BM25 index directory.

        """
        if not query or not query.strip():
            logger.error("Query cannot be empty.")
            return
        if k <= 0:
            logger.error("k must be a positive integer.")
            return
        if not Path(index_dir).exists():
            logger.error(
                "Index directory '%s' not found. Please run 'index' first.",
                index_dir,
            )
            return
        if not self._check_ollama():
            return

        try:
            unanswered = UnansweredQuestion(question=query)
            minimal_answer = get_answer(
                unanswered_question=unanswered, k=k, index_dir=index_dir,
            )
            final_result = StudentSearchResultsAndAnswer(
                search_results=[minimal_answer],
                k=k,
            )
            print(final_result.model_dump_json(indent=2))
        except FileNotFoundError:
            logger.exception("Required files not found")
        except Exception:
            logger.exception("answer failed")

    # ------------------------------------------------------------------
    # answer_dataset
    # ------------------------------------------------------------------

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """
        Generate answers for a dataset from pre-computed search results.

        Reads a StudentSearchResults JSON produced by search_dataset,
        retrieves context from source files, calls the LLM for each
        question, and writes a StudentSearchResultsAndAnswer JSON.

        Args:
            student_search_results_path: Path to StudentSearchResults JSON.
            save_directory: Directory to save the output JSON file.

        """
        if not Path(student_search_results_path).exists():
            logger.error(
                "Search results file not found: %s",
                student_search_results_path,
            )
            return
        if not self._check_ollama():
            return

        try:
            with Path(student_search_results_path).open(encoding="utf-8") as f:
                student_results = StudentSearchResults.model_validate_json(
                    f.read(),
                )
        except Exception:
            logger.exception("Failed to parse search results")
            return

        total = len(student_results.search_results)
        logger.info("Loaded %d questions …", total)

        minimal_answers = []
        try:
            for search_result in tqdm(
                student_results.search_results,
                desc="Generating answers",
            ):
                minimal_answer = answer_from_search_result(search_result)
                minimal_answers.append(minimal_answer)
        except Exception:
            logger.exception("Answer generation failed")

        final_result = StudentSearchResultsAndAnswer(
            search_results=minimal_answers,
            k=student_results.k,
        )

        Path(save_directory).mkdir(parents=True, exist_ok=True)
        output_path = (
            Path(save_directory) / Path(student_search_results_path).name
        )
        with Path(output_path).open("w", encoding="utf-8") as f:
            f.write(final_result.model_dump_json(indent=4))
        print(
            f"Processed {len(minimal_answers)} of {total} questions\n"
            f"Saved student_search_results_and_answer to {output_path}",
        )

    # ------------------------------------------------------------------
    # serve (bonus 5)
    # ------------------------------------------------------------------

    def serve(
        self,
        host: str = API_DEFAULT_HOST,
        port: int = API_DEFAULT_PORT,
        index_dir: str = DEFAULT_INDEX_DIR,
    ) -> None:
        """
        Serve search and answer over a local HTTP API.

        Args:
            host: Bind address (default 127.0.0.1).
            port: TCP port (default 8000).
            index_dir: Path to the BM25 index directory.

        """
        if not host or not str(host).strip():
            logger.error("host cannot be empty.")
            return
        if port <= 0 or port > 65535:
            logger.error("port must be in 1..65535.")
            return
        if not Path(index_dir).exists():
            logger.error(
                "Index directory '%s' not found. Please run 'index' first.",
                index_dir,
            )
            return
        try:
            get_cached_searcher(index_dir)
        except FileNotFoundError:
            logger.exception("Index files not found")
            return

        print(
            f"RAG API at http://{host}:{port}\n"
            f"  GET  /health\n"
            f"  GET  /search?query=...&k=5\n"
            f"  POST /search\n"
            f"  POST /answer",
        )
        try:
            run_server(host, port, index_dir)
        except OSError:
            logger.exception(
                "Could not bind API on %s:%s",
                host,
                port,
            )

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """
        Report recall@k against a ground-truth dataset (for local testing).

        Note: the official recall@k used during the defence is computed by
        the provided moulinette, not by this command.

        Args:
            student_search_results_path: Path to StudentSearchResults JSON.
            dataset_path: Path to the AnsweredQuestions ground-truth JSON.

        """
        if not Path(student_search_results_path).exists():
            logger.error(
                "Search results file not found: %s",
                student_search_results_path,
            )
            return
        if not Path(dataset_path).exists():
            logger.error("Ground truth file not found: %s", dataset_path)
            return

        try:
            with Path(student_search_results_path).open(encoding="utf-8") as f:
                search_data = StudentSearchResults.model_validate_json(f.read())
            with Path(dataset_path).open(encoding="utf-8") as f:
                ground_truth_data = RagDataset.model_validate_json(f.read())
        except Exception:
            logger.exception("Failed to load evaluation files")
            return

        ground_truth_map = {}
        for item in ground_truth_data.rag_questions:
            if hasattr(item, "sources"):
                ground_truth_map[item.question_id] = item.sources

        k = search_data.k
        quest_recall_scores = []

        for result in search_data.search_results:
            qid = result.question_id
            if qid not in ground_truth_map:
                logger.warning(
                    "Question %s not in ground truth. Skipping.", qid,
                )
                continue
            gt_sources = ground_truth_map[qid]
            if not gt_sources:
                continue

            number_found = 0
            for gt_src in gt_sources:
                for ret_src in result.retrieved_sources:
                    if gt_src.file_path == ret_src.file_path:
                        overlap = calculate_overlap_percentage(
                            gt_src.first_character_index,
                            gt_src.last_character_index,
                            ret_src.first_character_index,
                            ret_src.last_character_index,
                        )
                        if overlap >= 5.0:
                            number_found += 1
                            break
            quest_recall_scores.append(number_found / len(gt_sources))

        if not quest_recall_scores:
            logger.error("No matching questions found to evaluate.")
            return

        final_recall = sum(quest_recall_scores) / len(quest_recall_scores)
        print(
            f"Recall@{k}: {final_recall:.3f} "
            f"({final_recall:.1%}) over "
            f"{len(quest_recall_scores)} questions",
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _check_ollama(self) -> bool:
        """
        Return True if the Ollama server is reachable, else log an error.

        Returns:
            True when Ollama responds successfully, False otherwise.

        """
        try:
            response = requests.get(OLLAMA_HEALTH_URL, timeout=2)
            response.raise_for_status()
        except requests.exceptions.RequestException:
            logger.exception(
                "Ollama is not running or not accessible. "
                "Please start Ollama before running this command.",
            )
            return False
        return True


def main() -> None:
    """Main entry point — initialises and runs the RagCLI via Fire."""
    try:
        fire.Fire(RagCLI)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")


if __name__ == "__main__":
    main()
