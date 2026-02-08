"""
Vector indexer using sentence-transformers.

The embedding model is lazy-loaded on first use and can be explicitly
unloaded via _unload_model() to reclaim memory between processing runs.

Both single-file and batch indexing paths are provided.  Batch mode
encodes all texts in one call to the model, which is significantly
faster on CPU than encoding one at a time.
"""
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

from src.base import AbstractIndexer, FileContent
from src.config import get_config
from src.database import Database

# Maximum number of characters from extracted_text that are fed into
# the embedding model.  Longer documents are truncated here.
_MAX_EMBED_CHARS = 5000


class VectorIndexer(AbstractIndexer):
    """Generates and stores sentence embeddings for indexed content."""

    def __init__(self, database: Optional[Database] = None):
        self.config = get_config()
        self.db = database or Database()
        self._model: Optional[SentenceTransformer] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------- model mgmt

    def _load_model(self):
        """Lazy-load the sentence-transformer model."""
        if self._model is not None:
            return
        print("Loading embedding model...")
        self._model = SentenceTransformer(self.config.embedding_model)
        self._model.to(self.device)

    def _unload_model(self):
        """Release model memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---------------------------------------------------------- interface

    def index(self, file_id: int, content: FileContent) -> bool:
        """
        Generate an embedding for *content* and persist it.

        The text fed to the model is chosen as follows:
          - extracted_text (truncated to _MAX_EMBED_CHARS) if available
          - otherwise description

        Args:
            file_id: Identifier of the file row in metadata.db
            content: The FileContent produced by a processor

        Returns:
            True if an embedding was stored; False if there was nothing to embed
        """
        text = self._pick_text(content)
        if not text:
            return False

        try:
            self._load_model()
            embedding = self._model.encode(text, convert_to_tensor=False, show_progress_bar=False)
            self.db.add_embedding(file_id, embedding.tolist())
            return True
        except Exception as e:
            print(f"Failed to index file {file_id}: {e}")
            return False

    def is_indexed(self, file_id: int) -> bool:
        """Return True when an embedding already exists for *file_id*."""
        return self.db.get_embedding(file_id) is not None

    # ---------------------------------------------------------- batch mode

    def batch_index(self, items: List) -> int:
        """
        Index a list of (file_id, FileContent) pairs in one model call.

        This is the preferred way to index many files — the model encodes
        all texts together, which amortises overhead on CPU.

        Args:
            items: List of (file_id, FileContent) tuples

        Returns:
            Number of embeddings successfully stored
        """
        if not items:
            return 0

        self._load_model()

        texts: List[str] = []
        file_ids: List[int] = []

        for file_id, content in items:
            text = self._pick_text(content)
            if text:
                texts.append(text)
                file_ids.append(file_id)

        if not texts:
            return 0

        try:
            embeddings = self._model.encode(
                texts,
                convert_to_tensor=False,
                show_progress_bar=True,
                batch_size=self.config.batch_size
            )

            stored = 0
            for fid, emb in zip(file_ids, embeddings):
                try:
                    self.db.add_embedding(fid, emb.tolist())
                    stored += 1
                except Exception as e:
                    print(f"Failed to store embedding for file {fid}: {e}")

            return stored

        except Exception as e:
            print(f"Batch indexing failed: {e}")
            return 0

    # ---------------------------------------------------------- helpers

    @staticmethod
    def _pick_text(content: FileContent) -> Optional[str]:
        """Choose and truncate the best text to embed."""
        if content.extracted_text:
            return content.extracted_text[:_MAX_EMBED_CHARS]
        if content.description:
            return content.description
        return None

    def __del__(self):
        self._unload_model()
