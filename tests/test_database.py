"""
Tests for the Database class.

Every test receives a fresh pair of SQLite databases via the temp_db
fixture — nothing touches a real /data directory.
"""
import pytest
import tempfile
import shutil
from pathlib import Path

from src.database import Database
from src.base import FileMetadata, FileContent, FileType, ProcessingState


@pytest.fixture
def temp_db():
    """Spin up an isolated Database backed by temp files."""
    tmp = Path(tempfile.mkdtemp())
    db = Database(
        db_path=tmp / "meta.db",
        vector_db_path=tmp / "vec.db"
    )
    yield db
    shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------ file records

def test_add_file_returns_positive_id(temp_db):
    meta = FileMetadata(
        file_path="/a.txt",
        original_url="http://x.com/a.txt",
        file_type=FileType.TEXT,
        file_size=100,
        checksum="aaa111"
    )
    fid = temp_db.add_file(meta)
    assert isinstance(fid, int)
    assert fid > 0


def test_file_exists_false_before_insert(temp_db):
    assert temp_db.file_exists("bbb222") is False


def test_file_exists_true_after_insert(temp_db):
    temp_db.add_file(FileMetadata(
        file_path="/b.txt",
        original_url="http://x.com/b.txt",
        file_type=FileType.TEXT,
        file_size=50,
        checksum="bbb222"
    ))
    assert temp_db.file_exists("bbb222") is True


def test_get_file_by_checksum(temp_db):
    temp_db.add_file(FileMetadata(
        file_path="/c.pdf",
        original_url="http://x.com/c.pdf",
        file_type=FileType.PDF,
        file_size=2048,
        checksum="ccc333"
    ))
    result = temp_db.get_file_by_checksum("ccc333")
    assert result is not None
    assert result.file_path == "/c.pdf"
    assert result.file_type == FileType.PDF


def test_get_file_by_checksum_returns_none_for_missing(temp_db):
    assert temp_db.get_file_by_checksum("nonexistent") is None


# ------------------------------------------------------------ content

def _insert_file(db, checksum="x1") -> int:
    """Helper: insert a minimal file row and return its ID."""
    return db.add_file(FileMetadata(
        file_path=f"/file_{checksum}.txt",
        original_url="http://example.com",
        file_type=FileType.TEXT,
        file_size=10,
        checksum=checksum
    ))


def test_add_content_returns_positive_id(temp_db):
    fid = _insert_file(temp_db, "ct1")
    cid = temp_db.add_content(FileContent(
        file_id=fid,
        extracted_text="Hello world",
        description="A greeting"
    ))
    assert cid > 0


def test_fts_search_finds_inserted_text(temp_db):
    fid = _insert_file(temp_db, "ct2")
    temp_db.add_content(FileContent(
        file_id=fid,
        extracted_text="The quick brown fox jumps over the lazy dog",
        description="Classic pangram"
    ))
    results = temp_db.search_text("fox", limit=5)
    assert len(results) >= 1
    assert results[0][0] == fid  # file_id matches


def test_fts_search_returns_empty_for_no_match(temp_db):
    fid = _insert_file(temp_db, "ct3")
    temp_db.add_content(FileContent(
        file_id=fid,
        extracted_text="Nothing special here",
        description="Boring doc"
    ))
    results = temp_db.search_text("tractor", limit=5)
    assert results == []


# ---------------------------------------------------------- processing state

def test_processing_state_initially_none(temp_db):
    assert temp_db.get_processing_state("http://new.url") is None


def test_processing_state_round_trip(temp_db):
    url = "http://state.test/doc.pdf"

    temp_db.update_processing_state(url, ProcessingState.PROCESSING)
    assert temp_db.get_processing_state(url) == ProcessingState.PROCESSING

    temp_db.update_processing_state(url, ProcessingState.COMPLETED)
    assert temp_db.get_processing_state(url) == ProcessingState.COMPLETED


def test_processing_state_failed(temp_db):
    url = "http://state.test/bad.pdf"
    temp_db.update_processing_state(url, ProcessingState.FAILED)
    assert temp_db.get_processing_state(url) == ProcessingState.FAILED


# ------------------------------------------------------------ embeddings

def test_add_and_get_embedding(temp_db):
    fid = _insert_file(temp_db, "emb1")
    vector = [0.1, 0.2, 0.3, 0.4]

    temp_db.add_embedding(fid, vector)
    retrieved = temp_db.get_embedding(fid)

    assert retrieved is not None
    assert len(retrieved) == 4
    # Float round-trip through JSON should be exact for these values
    assert retrieved == pytest.approx(vector)


def test_get_embedding_returns_none_when_missing(temp_db):
    assert temp_db.get_embedding(99999) is None


def test_add_embedding_replaces_existing(temp_db):
    fid = _insert_file(temp_db, "emb2")
    temp_db.add_embedding(fid, [1.0, 2.0])
    temp_db.add_embedding(fid, [3.0, 4.0])  # replace

    result = temp_db.get_embedding(fid)
    assert result == pytest.approx([3.0, 4.0])


# -------------------------------------------------------- checksum helper

def test_calculate_checksum(temp_db):
    tmp = Path(tempfile.mkdtemp())
    try:
        f = tmp / "sample.txt"
        f.write_text("deterministic content")

        cs1 = Database.calculate_checksum(f)
        cs2 = Database.calculate_checksum(f)

        assert cs1 == cs2
        assert len(cs1) == 64  # SHA-256 hex length
    finally:
        shutil.rmtree(tmp)
