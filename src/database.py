"""
Database operations for metadata and vector storage.
Uses SQLite with FTS5 for full-text search and sqlite-vec for
vector similarity search.
"""
import sqlite3
import hashlib
import os
import struct
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager

from src.base import FileMetadata, FileContent, FileType, ProcessingState, SearchResult
from src.config import get_config


class Database:
    """Handles all database operations for the system."""

    def __init__(self, db_path: Optional[Path] = None, vector_db_path: Optional[Path] = None):
        """
        Initialize database connections and load sqlite-vec extension.

        Args:
            db_path: Path to metadata database
            vector_db_path: Path to vector database
        """
        config = get_config()
        self.db_path = db_path or config.database_path
        self.vector_db_path = vector_db_path or config.vector_db_path
        
        # Path to sqlite-vec extension (set via environment variable in Dockerfile)
        self.vec_ext_path = os.getenv('SQLITE_VEC_PATH', '/opt/sqlite-extensions/vec0')

        self._init_metadata_db()
        self._init_vector_db()

    # ------------------------------------------------------------------ schema

    def _init_metadata_db(self):
        """Initialize metadata database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL UNIQUE,
                    original_url TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    checksum TEXT NOT NULL UNIQUE,
                    library_path TEXT,
                    processed_date TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS content (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    extracted_text TEXT,
                    description TEXT,
                    is_fully_redacted INTEGER DEFAULT 0,
                    page_count INTEGER,
                    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_state (
                    url TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    last_updated TEXT NOT NULL
                )
            """)

            # Indices for common lookups
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_path ON files(file_path)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_checksum ON files(checksum)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_file_id ON content(file_id)")

            # FTS5 virtual table for full-text search
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS content_fts
                USING fts5(file_id, extracted_text, description)
            """)

            conn.commit()

    def _init_vector_db(self):
        """
        Initialize vector database for embeddings using sqlite-vec.
        
        The extension must be loaded on every connection. The embeddings
        table uses a native vec0 column to store 384-dimensional float32
        vectors (MiniLM embedding dimension).
        """
        conn = sqlite3.connect(str(self.vector_db_path))
        conn.enable_load_extension(True)
        conn.load_extension(self.vec_ext_path)
        conn.enable_load_extension(False)
        
        cursor = conn.cursor()

        # Create table with native vec0 column type
        # vec0 stores vectors as compact binary blobs optimized for similarity search
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                file_id INTEGER PRIMARY KEY,
                embedding BLOB
            )
        """)

        conn.commit()
        conn.close()

    # --------------------------------------------------------------- helpers

    @contextmanager
    def _get_connection(self):
        """Context manager for metadata database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _get_vector_connection(self):
        """
        Get a connection to the vector database with sqlite-vec loaded.
        
        Caller is responsible for closing the connection.
        """
        conn = sqlite3.connect(str(self.vector_db_path))
        conn.enable_load_extension(True)
        conn.load_extension(self.vec_ext_path)
        conn.enable_load_extension(False)
        return conn

    @staticmethod
    def _serialize_vec(vector: List[float]) -> bytes:
        """
        Serialize a float vector to the binary format sqlite-vec expects.
        
        sqlite-vec uses little-endian float32 packed as raw bytes.
        
        Args:
            vector: List of floats
            
        Returns:
            Binary blob suitable for vec0 column
        """
        return struct.pack(f'{len(vector)}f', *vector)

    @staticmethod
    def _deserialize_vec(blob: bytes) -> List[float]:
        """
        Deserialize a vec0 blob back to a Python list of floats.
        
        Args:
            blob: Binary data from vec0 column
            
        Returns:
            List of floats
        """
        count = len(blob) // 4  # 4 bytes per float32
        return list(struct.unpack(f'{count}f', blob))

    @staticmethod
    def calculate_checksum(file_path: Path) -> str:
        """
        Calculate SHA-256 checksum of a file (streamed, no full read).

        Args:
            file_path: Path to file

        Returns:
            Hex digest string
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()

    # ---------------------------------------------------------- file records

    def add_file(self, metadata: FileMetadata) -> int:
        """
        Insert file metadata row.

        Args:
            metadata: FileMetadata instance

        Returns:
            Auto-generated file ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO files (file_path, original_url, file_type, file_size, checksum, library_path, processed_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.file_path,
                metadata.original_url,
                metadata.file_type.value,
                metadata.file_size,
                metadata.checksum,
                metadata.library_path,
                metadata.processed_date or datetime.now().isoformat()
            ))
            conn.commit()
            return cursor.lastrowid

    def file_exists(self, checksum: str) -> bool:
        """
        Check whether a file with the given checksum is already stored.

        Args:
            checksum: SHA-256 hex digest

        Returns:
            True if a matching row exists
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM files WHERE checksum = ?", (checksum,))
            return cursor.fetchone() is not None

    def get_file_by_checksum(self, checksum: str) -> Optional[FileMetadata]:
        """
        Retrieve file metadata by checksum.

        Args:
            checksum: SHA-256 hex digest

        Returns:
            FileMetadata or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM files WHERE checksum = ?", (checksum,))
            row = cursor.fetchone()

            if row:
                return FileMetadata(
                    id=row['id'],
                    file_path=row['file_path'],
                    original_url=row['original_url'],
                    file_type=FileType(row['file_type']),
                    file_size=row['file_size'],
                    checksum=row['checksum'],
                    library_path=row['library_path'],
                    processed_date=row['processed_date']
                )
            return None

    # --------------------------------------------------------------- content

    def add_content(self, content: FileContent) -> int:
        """
        Insert extracted content and update the FTS index.

        Args:
            content: FileContent instance

        Returns:
            Auto-generated content ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO content (file_id, extracted_text, description, is_fully_redacted, page_count)
                VALUES (?, ?, ?, ?, ?)
            """, (
                content.file_id,
                content.extracted_text,
                content.description,
                1 if content.is_fully_redacted else 0,
                content.page_count
            ))

            # Mirror into FTS index so full-text queries work
            if content.extracted_text or content.description:
                cursor.execute("""
                    INSERT INTO content_fts (file_id, extracted_text, description)
                    VALUES (?, ?, ?)
                """, (content.file_id, content.extracted_text or '', content.description or ''))

            conn.commit()
            return cursor.lastrowid

    # -------------------------------------------------------- processing state

    def update_processing_state(self, url: str, state: ProcessingState):
        """
        Upsert the processing state for a URL.

        Args:
            url: Source URL
            state: New ProcessingState
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO processing_state (url, state, last_updated)
                VALUES (?, ?, ?)
            """, (url, state.value, datetime.now().isoformat()))
            conn.commit()

    def get_processing_state(self, url: str) -> Optional[ProcessingState]:
        """
        Get the current processing state for a URL.

        Args:
            url: Source URL

        Returns:
            ProcessingState or None if not tracked
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT state FROM processing_state WHERE url = ?", (url,))
            row = cursor.fetchone()

            if row:
                return ProcessingState(row['state'])
            return None

    # ----------------------------------------------------------- FTS search

    def search_text(self, query: str, limit: int = 10) -> List[Tuple[int, str, str, FileType, str]]:
        """
        Full-text search across extracted_text and description via FTS5.

        Args:
            query: FTS5 query string
            limit: Maximum number of rows

        Returns:
            List of (file_id, file_path, description, file_type, library_path)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT f.id, f.file_path, c.description, f.file_type, f.library_path
                FROM content_fts cf
                JOIN content c ON cf.file_id = c.file_id
                JOIN files f ON c.file_id = f.id
                WHERE content_fts MATCH ?
                LIMIT ?
            """, (query, limit))

            results = []
            for row in cursor.fetchall():
                results.append((
                    row['id'],
                    row['file_path'],
                    row['description'] or '',
                    FileType(row['file_type']),
                    row['library_path']
                ))
            return results

    # ---------------------------------------------------------- embeddings

    def get_all_file_ids(self) -> List[int]:
        """
        Return every file ID in the metadata database.

        Returns:
            List of integer IDs
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM files")
            return [row['id'] for row in cursor.fetchall()]

    def add_embedding(self, file_id: int, embedding: List[float]):
        """
        Store (or replace) the embedding vector for a file using sqlite-vec.

        Args:
            file_id: File identifier
            embedding: Float vector as a Python list
        """
        blob = self._serialize_vec(embedding)

        conn = self._get_vector_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings (file_id, embedding)
                VALUES (?, ?)
            """, (file_id, blob))
            conn.commit()
        finally:
            conn.close()

    def get_embedding(self, file_id: int) -> Optional[List[float]]:
        """
        Retrieve the embedding vector for a file.

        Args:
            file_id: File identifier

        Returns:
            Float vector or None
        """
        conn = self._get_vector_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding FROM embeddings WHERE file_id = ?", (file_id,))
            row = cursor.fetchone()

            if row and row[0]:
                return self._deserialize_vec(row[0])
            return None
        finally:
            conn.close()

    def vector_search(self, query_embedding: List[float], limit: int = 10) -> List[Tuple[int, float]]:
        """
        Perform vector similarity search using sqlite-vec's native functions.
        
        This is much faster than fetching all embeddings and computing similarity
        in Python, especially as the database grows.
        
        Args:
            query_embedding: Query vector as a list of floats
            limit: Maximum number of results
            
        Returns:
            List of (file_id, distance) tuples, sorted by distance (lower = more similar)
        """
        query_blob = self._serialize_vec(query_embedding)
        
        conn = self._get_vector_connection()
        try:
            cursor = conn.cursor()
            
            # vec_distance_cosine returns cosine distance (0 = identical, 2 = opposite)
            # Lower distance = more similar
            cursor.execute("""
                SELECT 
                    file_id,
                    vec_distance_cosine(embedding, ?) as distance
                FROM embeddings
                WHERE embedding IS NOT NULL
                ORDER BY distance ASC
                LIMIT ?
            """, (query_blob, limit))
            
            return [(row[0], row[1]) for row in cursor.fetchall()]
        finally:
            conn.close()