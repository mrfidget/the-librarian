# Configuration Reference

Complete reference for all configuration options in The Librarian.

## Configuration File

All settings are defined in `.env` in the project root. Copy from `.env.example`:

```bash
cp .env.example .env
```

Changes require container restart:

```bash
make down
make up
```

## Environment Variables

### Processing Configuration

#### BATCH_SIZE
**Default**: `10`  
**Range**: 1-100

Number of files processed before triggering garbage collection and memory cleanup.

**When to adjust**:
- Lower (5) if experiencing memory pressure
- Higher (20) for faster processing on systems with more RAM

```env
BATCH_SIZE=10
```

#### MAX_WORKERS
**Default**: `2`  
**Range**: 1-8

Number of concurrent worker threads for I/O operations (downloading, file operations).

**Note**: Does not affect AI model parallelism (models run in single worker process).

```env
MAX_WORKERS=2
```

#### CHUNK_SIZE
**Default**: `65536` (64KB)  
**Range**: 4096-1048576

Size of chunks when streaming file downloads.

```env
CHUNK_SIZE=65536
```

### Storage Paths

#### DATABASE_PATH
**Default**: `/data/database/metadata.db`

Location of the main SQLite database containing file metadata, content, and FTS index.

```env
DATABASE_PATH=/data/database/metadata.db
```

#### VECTOR_DB_PATH
**Default**: `/data/database/vectors.db`

Location of the vector database containing embeddings for semantic search.

```env
VECTOR_DB_PATH=/data/database/vectors.db
```

#### STAGING_PATH
**Default**: `/data/staging`

Temporary directory for downloaded files before processing. Cleaned automatically after processing.

```env
STAGING_PATH=/data/staging
```

#### LIBRARY_PATH
**Default**: `/data/library`

Permanent storage for processed files. Organized by type: `text/`, `images/`, `pdfs/`.

```env
LIBRARY_PATH=/data/library
```

#### BACKUP_PATH
**Default**: `/data/backups`

Location for database backup snapshots.

```env
BACKUP_PATH=/data/backups
```

### AI Models

#### EMBEDDING_MODEL
**Default**: `sentence-transformers/all-MiniLM-L6-v2`

HuggingFace model ID for text embeddings (semantic search).

**Options**:
- `sentence-transformers/all-MiniLM-L6-v2` (384 dim, ~90MB) - Fast, good quality
- `sentence-transformers/all-mpnet-base-v2` (768 dim, ~420MB) - Higher quality, slower
- `sentence-transformers/paraphrase-MiniLM-L3-v2` (384 dim, ~61MB) - Faster, lower quality

**Note**: Changing requires rebuild to download new model.

```env
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

#### VISION_MODEL
**Default**: `openai/clip-vit-base-patch32`

HuggingFace model ID for image classification.

**Options**:
- `openai/clip-vit-base-patch32` (~600MB) - Balanced speed/quality
- `openai/clip-vit-large-patch14` (~1.7GB) - Higher quality, much slower

**Note**: Changing requires rebuild to download new model.

```env
VISION_MODEL=openai/clip-vit-base-patch32
```

#### OCR_ENABLED
**Default**: `true`  
**Values**: `true` or `false`

Enable OCR for scanned PDFs and images with minimal text.

**When to disable**:
- Processing only born-digital PDFs (significant speed up)
- Limited CPU resources
- OCR quality not needed

```env
OCR_ENABLED=true
```

#### OCR_ENGINE
**Default**: `tesseract`  
**Values**: `tesseract` (only option currently)

OCR engine to use. Currently only Tesseract is supported.

```env
OCR_ENGINE=tesseract
```

### Backup Configuration

#### BACKUP_ENABLED
**Default**: `true`  
**Values**: `true` or `false`

Enable or disable backup functionality.

```env
BACKUP_ENABLED=true
```

#### BACKUP_SCHEDULE
**Default**: `0 2 * * 0` (2 AM every Sunday)  
**Format**: Cron expression

**Note**: Currently not used (automated backups not implemented). Reserved for future use.

```env
BACKUP_SCHEDULE=0 2 * * 0
```

### Resource Limits

#### MAX_MEMORY_MB
**Default**: `3500` (3.5 GB)  
**Range**: 2000-8000

Soft memory limit for processing. Used for monitoring, not enforced.

**Note**: Hard limits are set in docker-compose.yml (`mem_limit: 4g`).

```env
MAX_MEMORY_MB=3500
```

### Logging

#### LOG_LEVEL
**Default**: `INFO`  
**Values**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

Python logging level.

**Levels**:
- `DEBUG` - Verbose, shows all operations
- `INFO` - Normal, shows key operations
- `WARNING` - Only warnings and errors
- `ERROR` - Only errors

```env
LOG_LEVEL=INFO
```

### Search Configuration

#### SEARCH_THRESHOLD
**Default**: `0.25`  
**Range**: 0.0-1.0

Minimum similarity score for search results. Results below threshold are filtered out.

**When to adjust**:
- Lower (0.15) to see more results (less strict)
- Higher (0.40) to see only strong matches (more strict)

```env
SEARCH_THRESHOLD=0.25
```

### Docker Configuration

#### IMAGE_NAME
**Default**: `the-librarian`

Docker image name.

```env
IMAGE_NAME=the-librarian
```

#### CONTAINER_NAME
**Default**: `the-librarian`

Docker container name.

```env
CONTAINER_NAME=the-librarian
```

#### VERSION
**Default**: `1.0.0`

Docker image tag/version.

```env
VERSION=1.0.0
```

## Docker Compose Configuration

Additional settings in `docker-compose.yml`:

### Resource Limits

```yaml
mem_limit: 4g    # Hard memory limit
cpus: 4.0        # CPU limit (4 cores)
```

### Logging

```yaml
logging:
  driver: json-file
  options:
    max-size: "10m"   # Max size per log file
    max-file: "3"     # Number of rotated files
```

### Volumes

```yaml
volumes:
  - ./data:/data                    # Persistent data
  - ./urls.txt:/app/urls.txt:ro     # URL list (read-only)
```

### Environment

```yaml
environment:
  - PYTHONUNBUFFERED=1               # Immediate stdout
  - PYTHONDONTWRITEBYTECODE=1        # No .pyc files
  - TRANSFORMERS_VERBOSITY=error     # Suppress model warnings
```

## Configuration Examples

### Minimal Resource (2GB RAM)

```env
BATCH_SIZE=3
MAX_WORKERS=1
MAX_MEMORY_MB=1800
OCR_ENABLED=false
```

Update docker-compose.yml:
```yaml
mem_limit: 2g
cpus: 2.0
```

### High Performance (8GB RAM)

```env
BATCH_SIZE=20
MAX_WORKERS=4
MAX_MEMORY_MB=7500
OCR_ENABLED=true
```

Update docker-compose.yml:
```yaml
mem_limit: 8g
cpus: 8.0
```

### OCR-Heavy Workload

```env
BATCH_SIZE=5
MAX_WORKERS=2
OCR_ENABLED=true
```

**Also consider**: Using higher quality OCR model (future enhancement).

### Development Mode

```env
LOG_LEVEL=DEBUG
BATCH_SIZE=5
```

Mount code as volume in docker-compose.yml:
```yaml
volumes:
  - ./src:/app/src  # Live code updates
```

## Tuning Guide

### For Speed

1. Disable OCR: `OCR_ENABLED=false`
2. Increase batch size: `BATCH_SIZE=20`
3. Use faster embedding model (trade-off with quality)

### For Quality

1. Enable OCR: `OCR_ENABLED=true`
2. Use better embedding model: `all-mpnet-base-v2`
3. Lower search threshold: `SEARCH_THRESHOLD=0.15`

### For Low Memory

1. Reduce batch size: `BATCH_SIZE=3`
2. Lower workers: `MAX_WORKERS=1`
3. Reduce docker memory limit: `mem_limit: 2g`

### For Large Collections

1. Increase batch size: `BATCH_SIZE=20`
2. Monitor with: `make logs`
3. Consider periodic database vacuum: `sqlite3 metadata.db 'VACUUM;'`

## Validation

After changing configuration:

```bash
# Restart container
make down
make up

# Check logs for errors
make logs

# Test processing
echo "https://example.com/test.pdf" > urls.txt
make process-urls

# Test querying
make query
```

## Advanced: Model Cache Paths

These are set automatically in the Dockerfile and shouldn't normally be changed:

```dockerfile
ENV HF_HOME=/opt/hf_cache
ENV TRANSFORMERS_CACHE=/opt/hf_cache/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/opt/hf_cache/sentence_transformers
ENV HF_HUB_OFFLINE=1
```

## Troubleshooting Configuration

**Changes not taking effect**:
```bash
make down
make up
```

**Invalid values**:
Check logs: `make logs`

**Performance issues after changes**:
Revert to defaults in `.env.example`

**Can't find .env file**:
```bash
ls -la .env
cp .env.example .env
```