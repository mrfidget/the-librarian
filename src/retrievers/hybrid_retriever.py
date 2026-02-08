"""
Search / retrieval layer.

Three concrete retrievers are provided:

  ExactMatchRetriever  — FTS5 full-text search only.
  SemanticRetriever    — Vector similarity using sqlite-vec's native cosine distance.
  HybridRetriever      — Dispatches to one or the other based on the
                         shape of the user's query.

Query-type heuristic (HybridRetriever):
  * If the query contains double quotes or the words "contains" /
    "phrase", treat it as an exact-match request.  A quoted substring
    is extracted and used as the FTS5 MATCH term.
  * Otherwise fall back to semantic (vector) search.
"""
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

from src.base import AbstractRetriever, SearchResult, FileType
from src.config import get_config
from src.database import Database


# ---------------------------------------------------------------------------
# ExactMatchRetriever
# ---------------------------------------------------------------------------

class ExactMatchRetriever(AbstractRetriever):
    """Full-text search using SQLite FTS5."""

    def __init__(self, database: Optional[Database] = None):
        self.db = database or Database()

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Run *query* against the FTS5 index and return results.

        Args:
            query: FTS5 MATCH expression
            limit: Cap on returned rows

        Returns:
            List of SearchResult (score is fixed at 1.0 for exact matches)
        """
        rows = self.db.search_text(query, limit)
        return [
            SearchResult(
                file_id=fid,
                file_path=path,
                description=desc,
                score=1.0,
                file_type=ftype,
                library_path=lib_path
            )
            for fid, path, desc, ftype, lib_path in rows
        ]


# ---------------------------------------------------------------------------
# SemanticRetriever
# ---------------------------------------------------------------------------

class SemanticRetriever(AbstractRetriever):
    """Vector similarity search over stored embeddings."""

    def __init__(self, database: Optional[Database] = None):
        self.config = get_config()
        self.db = database or Database()
        self._model: Optional[SentenceTransformer] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        if self._model is None:
            print("Loading embedding model for search...")
            self._model = SentenceTransformer(self.config.embedding_model)
            self._model.to(self.device)

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Encode *query* and use sqlite-vec's native similarity search.

        Args:
            query: Natural-language search query
            limit: Maximum results

        Returns:
            List of SearchResult sorted by similarity (lower distance = better match)
        """
        self._load_model()

        query_emb = self._model.encode(query, convert_to_tensor=False, show_progress_bar=False)

        # Use database's native vector search (much faster than Python-side scoring)
        scored = self.db.vector_search(query_emb.tolist(), limit)

        # Hydrate results with metadata
        results: List[SearchResult] = []
        for fid, distance in scored:
            with self.db._get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT f.file_path, c.description, f.file_type, f.library_path
                    FROM files f
                    LEFT JOIN content c ON f.id = c.file_id
                    WHERE f.id = ?
                """, (fid,))
                row = cur.fetchone()

            if row:
                # Convert cosine distance to similarity score (1 - distance/2)
                # Cosine distance ranges 0-2, so this maps to 1.0 (perfect) to 0.0 (opposite)
                similarity = 1.0 - (distance / 2.0)
                
                results.append(SearchResult(
                    file_id=fid,
                    file_path=row['file_path'],
                    description=row['description'] or 'No description',
                    score=similarity,
                    file_type=FileType(row['file_type']),
                    library_path=row['library_path']
                ))

        return results


# ---------------------------------------------------------------------------
# HybridRetriever  (default)
# ---------------------------------------------------------------------------

class HybridRetriever(AbstractRetriever):
    """
    Dispatches to ExactMatchRetriever or SemanticRetriever based on
    simple query-shape heuristics.
    """

    def __init__(self, database: Optional[Database] = None):
        self.db = database or Database()
        self._exact = ExactMatchRetriever(database=self.db)
        self._semantic = SemanticRetriever(database=self.db)

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Route *query* to the most appropriate search backend.

        Exact-match triggers:
          - Query contains a double-quoted phrase  →  extract and search
            the quoted substring.
          - Query contains the word "contains" or "phrase"  →  use the
            full query string as the FTS5 term.

        Everything else goes to semantic search.

        Args:
            query: User query string
            limit: Maximum results

        Returns:
            List of SearchResult
        """
        lower = query.lower()

        # --- exact-match path ---
        if '"' in query:
            # Extract first quoted substring
            start = query.index('"') + 1
            try:
                end = query.index('"', start)
                phrase = query[start:end]
                return self._exact.search(phrase, limit)
            except ValueError:
                pass  # Unmatched quote — fall through to semantic

        if 'contains' in lower or 'phrase' in lower:
            return self._exact.search(query, limit)

        # --- semantic path ---
        return self._semantic.search(query, limit)