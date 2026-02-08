# User Guide

Complete guide to using The Librarian for document management and search.

## Table of Contents

- [Makefile Commands](#makefile-commands)
- [Processing Documents](#processing-documents)
- [Querying Documents](#querying-documents)
- [Backup and Restore](#backup-and-restore)
- [Finding Your Files](#finding-your-files)
- [Best Practices](#best-practices)

## Makefile Commands

All operations use `make` commands for simplicity:

| Command | Function |
|---------|----------|
| `make build` | Build the Docker image (first time or after updates) |
| `make up` | Start the librarian container |
| `make down` | Stop the librarian container |
| `make process-urls` | Download and index all URLs in `urls.txt` |
| `make query` | Open interactive search prompt |
| `make backup` | Create backup snapshot of databases |
| `make restore BACKUP=<path>` | Restore from a backup |
| `make test` | Run the test suite |
| `make clean` | Delete staging/temp files |
| `make clean-all` | Stop container, remove image, wipe staging |
| `make shell` | Drop into bash shell inside container (debugging) |
| `make logs` | Tail container logs |

## Processing Documents

### Adding URLs

Edit `urls.txt` and add URLs, one per line:

```
# Comments start with #
https://example.com/document.pdf
https://example.com/archive.zip
https://example.com/image.jpg

# Local files use file:// URLs
file:///app/local-document.pdf
```

**Supported URL schemes**:
- `http://` and `https://` - Remote files
- `file://` - Local files (must be mounted in Docker)

**Supported file types**:
- **Text**: `.txt`, `.md`, `.csv`, `.log`
- **Images**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`
- **PDFs**: Any PDF (scanned or digital)
- **Archives**: `.zip` (extracted automatically)

### Processing

```bash
make process-urls
```

**What happens**:
1. URLs are downloaded to `/data/staging/`
2. ZIP archives are extracted
3. Each file is:
   - Deduplicated by SHA-256 checksum
   - Classified by file type
   - Copied to `/data/library/{type}/`
   - Processed for content extraction
   - Indexed for search
4. Staging area is cleaned

**Example output**:

```
Processing 3 URL(s)...
[ModelServer] All models loaded and ready

Downloading https://example.com/report.pdf...
Processing report.pdf as pdf...
  Page 1: 1234 chars extracted
  Page 2: 987 chars extracted
OK: report.pdf

Downloading https://example.com/photos.zip...
Extracting photos.zip...
Processing photo1.jpg as image...
OK: photo1.jpg
Processing photo2.jpg as image...
Duplicate: photo2.jpg (checksum a7f3b2c9...), skipping

Successfully processed 2 files
```

### Incremental Processing

The system tracks which URLs have been processed. Run `make process-urls` multiple times:

- **Completed URLs**: Skipped automatically
- **New URLs**: Processed normally
- **Failed URLs**: Retried

To reprocess a URL, remove it from the database or delete the database entirely.

### Local Files

To process local files:

1. **Mount the file in docker-compose.yml**:

```yaml
volumes:
  - ./data:/data
  - ./urls.txt:/app/urls.txt:ro
  - ./my-files:/app/my-files:ro  # Add this
```

2. **Reference in urls.txt**:

```
file:///app/my-files/document.pdf
file:///app/my-files/archive.zip
```

3. **Restart and process**:

```bash
make down
make up
make process-urls
```

## Querying Documents

### Interactive Mode

```bash
make query
```

This opens Ms. Clarke's interactive search interface:

```
================================================================================
📚 DOCUMENT LIBRARY — Ms. Clarke, Head Librarian
================================================================================

*Ms. Clarke adjusts her reading glasses and looks up from her desk*

"Good afternoon. State your inquiry clearly, please.
And do remember: I expect precision. None of this
'find me stuff about things' nonsense."

Type 'quit' or 'exit' to leave.

================================================================================

📚 Ask the librarian: 
```

### Query Types

**Semantic Search** (default):

Natural language queries use AI to find conceptually similar documents:

```
📚 Ask the librarian: contracts with payment terms

*Ms. Clarke returns promptly, looking rather pleased*

"Ah yes. I thought so. Here's what you're after:"

Rank   Stars      Type     Filename                       Description
----------------------------------------------------------------------------------------------------
1      ★★★★★      pdf      service-agreement.pdf          Legal contract detailing payment...
2      ★★★★☆      text     vendor-terms.txt               Payment schedule and terms
3      ★★★☆☆      pdf      invoice-template.pdf           Standard invoice format

*3 item(s) found. Files are in /data/library/*
```

**Exact Match Search**:

Use quotes for exact phrase matching:

```
📚 Ask the librarian: "net 30 payment"
```

Or use the word "contains":

```
📚 Ask the librarian: contains quarterly report
```

### Understanding Results

**Star Ratings**:
- ★★★★★ Excellent match (0.8-1.0 similarity)
- ★★★★☆ Good match (0.6-0.8)
- ★★★☆☆ Fair match (0.4-0.6)
- ★★☆☆☆ Weak match (0.2-0.4)
- ★☆☆☆☆ Poor match (0.0-0.2)

Results below the threshold (default 0.25) are filtered out.

**Ms. Clarke's Mood**:
- Pleased: Top result scored ≥ 0.8
- Neutral: Top result scored 0.5-0.8
- Disapproving: Top result scored < 0.5
- Very disapproving: Nothing met threshold

### Query Tips

**Good queries**:
- "documents about dogs" ✓
- "contracts signed in 2024" ✓
- "photos of landscapes" ✓
- "meeting notes from Q4" ✓

**Less effective**:
- "stuff" (too vague)
- "find things" (no semantic meaning)
- "files" (matches everything)

**For best results**:
- Be specific but natural
- Use domain-relevant terms
- Include context (time, topic, type)

## Backup and Restore

### Creating Backups

```bash
make backup
```

Backups are stored in `/data/backups/` with timestamp:

```
data/backups/
└── backup_20260208_153000/
    ├── metadata.db
    ├── vectors.db
    └── manifest.txt
```

The manifest contains:
- Backup timestamp
- File sizes
- Checksums

### Restoring from Backup

**⚠️ WARNING**: Restoring overwrites current databases!

```bash
# Back up current state first
make backup

# Restore from specific backup
make restore BACKUP=/data/backups/backup_20260208_153000
```

**What gets restored**:
- Metadata database (file info, content, FTS index)
- Vector database (embeddings)

**What does NOT get restored**:
- Files in `/data/library/` (these persist separately)
- Processing state
- Configuration

### Backup Strategy

**Recommended schedule**:
- After major document ingestion
- Before major changes (upgrades, config changes)
- Weekly for active collections

**Manual backup**:

```bash
# Full backup including library files
cp -r data data-backup-$(date +%Y%m%d)
```

## Finding Your Files

### From Query Results

Query results show the original filename, but files are stored with checksum names in `/data/library/`.

**To locate a file**:

1. Note the filename from query results (e.g., `contract.pdf`)
2. Use shell to find it:

```bash
make shell
# Inside container:
find /data/library -name "*.pdf" | xargs grep -l "contract" /data/database/metadata.db
exit
```

Or check the database directly:

```bash
make shell
sqlite3 /data/database/metadata.db
SELECT library_path, file_path FROM files WHERE file_path LIKE '%contract%';
```

### Exporting Files

**Copy specific file out of container**:

```bash
# Find the checksum filename first
docker compose exec librarian ls /data/library/pdfs/

# Copy to host
docker cp the-librarian:/data/library/pdfs/a7f3b2c9....pdf ./contract.pdf
```

**Export all files**:

```bash
# Library directory is mounted at ./data/library/
ls -R data/library/
```

## Best Practices

### Processing

**Do**:
- Process documents in batches (add multiple URLs to `urls.txt`)
- Let it run overnight for large collections
- Check logs if processing seems stuck

**Don't**:
- Process the same URL twice (it's tracked automatically)
- Add non-text binary formats (executables, videos)
- Process extremely large files (>100MB PDFs may be slow)

### Querying

**Do**:
- Use natural language queries
- Start broad, then refine
- Try different phrasings if results are weak

**Don't**:
- Expect perfect accuracy (it's AI-based)
- Query for files you haven't indexed yet
- Use very long multi-sentence queries

### Maintenance

**Regular tasks**:
- `make backup` - Weekly or before changes
- `make clean` - When staging area gets large
- Check `make logs` - If something seems wrong

**Occasional tasks**:
- Rebuild image when models update
- Clean old backups from `/data/backups/`
- Review and update `urls.txt`

### Performance

**If processing is slow**:
- Reduce `BATCH_SIZE` in `.env` (default 10 → 5)
- Disable OCR for digital PDFs: `OCR_ENABLED=false`
- Process fewer files per run

**If queries are slow**:
- Reduce result limit: `make query` then ask for fewer results
- Use exact match for simple lookups
- Check database size (millions of files may need optimization)

**If memory is tight**:
- Lower `BATCH_SIZE`
- Ensure swap is enabled on host
- Increase Docker memory limit in docker-compose.yml

## Advanced Usage

### Custom Model Configuration

Edit `.env` to use different models:

```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VISION_MODEL=openai/clip-vit-base-patch32
```

Then rebuild:

```bash
make clean-all
make build
make up
```

### Batch Operations

Process specific file types:

```bash
# In urls.txt, group by type
# PDFs first
https://example.com/doc1.pdf
https://example.com/doc2.pdf

# Then images
https://example.com/photo1.jpg
https://example.com/photo2.jpg
```

### Debugging

**Check what's being processed**:

```bash
make logs
```

**Inspect databases**:

```bash
make shell
sqlite3 /data/database/metadata.db
.tables
SELECT COUNT(*) FROM files;
.quit
```

**Test model worker**:

```bash
make shell
python
>>> from src.workers.model_client import ModelClient
>>> # Models should already be loaded
```

## Troubleshooting

See [Troubleshooting Guide](troubleshooting.md) for common issues and solutions.