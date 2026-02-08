"""
Orchestrator — the single coordinator for the entire processing pipeline.

Responsibilities:
  * Iterate over a list of URLs, skipping anything already completed.
  * For each URL: download → extract (if archive) → classify → process
    → index.
  * Expose a search() facade and backup/restore helpers.
  * Enforce memory discipline: explicit gc.collect() after every batch.
"""
import gc
import shutil
from pathlib import Path
from typing import List

from src.base import FileMetadata, FileType, ProcessingState
from src.config import get_config
from src.database import Database
from src.downloaders.url_downloader import URLDownloader
from src.extractors.zip_extractor import ZipExtractor
from src.classifiers.magic_classifier import MagicByteClassifier
from src.processors.text_processor import TextProcessor
from src.processors.image_processor import ImageProcessor
from src.processors.pdf_processor import PDFProcessor
from src.indexers.vector_indexer import VectorIndexer
from src.retrievers.hybrid_retriever import HybridRetriever
from src.backup.filesystem_backup import FileSystemBackup


class Orchestrator:
    """Wires together every component and drives the pipeline."""

    def __init__(self):
        self.config = get_config()
        self.db = Database()

        # Pipeline stages
        self.downloader = URLDownloader()
        self.extractor = ZipExtractor()
        self.classifier = MagicByteClassifier()

        # Processors — ImageProcessor is shared with PDFProcessor for OCR
        self.image_processor = ImageProcessor()
        self.text_processor = TextProcessor()
        self.pdf_processor = PDFProcessor(image_processor=self.image_processor)

        # Ordered list; first matching processor wins
        self._processors = [
            self.text_processor,
            self.image_processor,
            self.pdf_processor,
        ]

        self.indexer = VectorIndexer(database=self.db)
        self.retriever = HybridRetriever(database=self.db)
        self.backup_service = FileSystemBackup()

    # ---------------------------------------------------------- public API

    def process_urls(self, urls: List[str], clean_after: bool = True) -> int:
        """
        Run the full pipeline for every URL in *urls*.

        Already-completed URLs (tracked via processing_state) are skipped
        automatically — this is the incremental-indexing behaviour.

        Args:
            urls: Source URLs to fetch and index
            clean_after: Delete staging files when finished

        Returns:
            Total number of individual files successfully processed
        """
        total = 0

        for url in urls:
            state = self.db.get_processing_state(url)
            if state == ProcessingState.COMPLETED:
                print(f"Skipping {url} — already processed")
                continue

            try:
                self.db.update_processing_state(url, ProcessingState.PROCESSING)

                print(f"\nDownloading {url}...")
                path = self.downloader.download(url, self.config.staging_path / "downloads")

                total += self._process_downloaded_file(path, url)

                self.db.update_processing_state(url, ProcessingState.COMPLETED)

            except Exception as e:
                print(f"Error processing {url}: {e}")
                self.db.update_processing_state(url, ProcessingState.FAILED)

        if clean_after:
            self._cleanup_staging()

        gc.collect()
        return total

    def search(self, query: str, limit: int = 10) -> list:
        """Delegate to the configured retriever."""
        return self.retriever.search(query, limit)

    def backup_data(self) -> bool:
        """Back up both databases to the configured backup path."""
        if not self.config.backup_enabled:
            print("Backup disabled in configuration")
            return False

        return self.backup_service.backup(
            [self.config.database_path, self.config.vector_db_path],
            self.config.backup_path
        )

    def restore_data(self, backup_path: Path) -> bool:
        """Restore databases from a previous backup directory."""
        return self.backup_service.restore(backup_path, self.config.database_path.parent)

    # ---------------------------------------------------------- internals

    def _process_downloaded_file(self, file_path: Path, original_url: str) -> int:
        """
        If *file_path* is an archive, extract and process every file inside.
        Otherwise process it directly.  The archive itself is deleted after
        successful extraction.

        Returns:
            Number of files processed from this download
        """
        count = 0

        if self.extractor.is_archive(file_path):
            print(f"Extracting {file_path.name}...")
            extract_dir = self.config.staging_path / "extracted" / file_path.stem

            for extracted in self.extractor.extract(file_path, extract_dir):
                if self._process_single_file(extracted, original_url):
                    count += 1

                # Periodic GC every batch_size files
                if count > 0 and count % self.config.batch_size == 0:
                    gc.collect()

            # Archive no longer needed
            file_path.unlink()
        else:
            if self._process_single_file(file_path, original_url):
                count += 1

        return count

    def _process_single_file(self, file_path: Path, original_url: str) -> bool:
        """
        Classify → deduplicate → process → index → copy to library.

        Args:
            file_path: Path on disk (in staging)
            original_url: URL the file originated from

        Returns:
            True if the file was fully processed and indexed
        """
        try:
            # --- deduplication via checksum ---
            checksum = Database.calculate_checksum(file_path)
            if self.db.file_exists(checksum):
                print(f"Duplicate: {file_path.name} (checksum {checksum[:8]}…), skipping")
                return False

            # --- classify ---
            file_type = self.classifier.classify(file_path)
            if file_type == FileType.UNKNOWN:
                print(f"Unknown type: {file_path.name}, skipping")
                return False

            print(f"Processing {file_path.name} as {file_type.value}...")

            # --- determine library subdirectory based on type ---
            if file_type == FileType.TEXT:
                lib_subdir = self.config.library_path / 'text'
            elif file_type == FileType.IMAGE:
                lib_subdir = self.config.library_path / 'images'
            elif file_type == FileType.PDF:
                lib_subdir = self.config.library_path / 'pdfs'
            else:
                lib_subdir = self.config.library_path

            # --- copy to library (use checksum as filename to avoid collisions) ---
            lib_filename = f"{checksum}{file_path.suffix}"
            library_path = lib_subdir / lib_filename
            shutil.copy2(file_path, library_path)

            # --- persist metadata ---
            metadata = FileMetadata(
                file_path=str(file_path),
                original_url=original_url,
                file_type=file_type,
                file_size=file_path.stat().st_size,
                checksum=checksum,
                library_path=str(library_path)
            )
            file_id = self.db.add_file(metadata)

            # --- find and run the right processor ---
            processor = None
            for p in self._processors:
                if p.can_process(file_type):
                    processor = p
                    break

            if processor is None:
                print(f"No processor for {file_type.value}")
                return False

            content = processor.process(file_path)
            content.file_id = file_id

            # --- persist content ---
            self.db.add_content(content)

            # --- index (skip if already present) ---
            if not self.indexer.is_indexed(file_id):
                self.indexer.index(file_id, content)

            print(f"OK: {file_path.name}")
            return True

        except Exception as e:
            print(f"Error in {file_path.name}: {e}")
            return False

    def _cleanup_staging(self):
        """Remove all temporary files from the staging area."""
        for subdir in ("downloads", "extracted"):
            path = self.config.staging_path / subdir
            if path.exists():
                shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)
        print("Staging area cleaned")