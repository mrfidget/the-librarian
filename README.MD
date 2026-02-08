# The Librarian

![An image of Ms. Clarke, the Head Librarian. No, Ms. Clarke does not like you.](/../main/docs/assets/Images/Ms-Clarke.png)

> "Good afternoon. State your inquiry clearly, please." — Ms. Clarke, Head Librarian

An AI-powered local document management system to retrieve, classify, store and query different formats of documents with natural language semantic search.

## Features

- Semantic Search: Natural language queries powered by sentence transformers
- Multi-Format Support: Text files, images, PDFs (scanned and digital)
- Intelligent OCR: Automatic fallback to OCR for scanned documents
- Deduplication: SHA-256 checksums prevent duplicate indexing
- Process Isolation: AI models run in separate worker process for stability
- Offline Operation: Models cached in Docker image, no runtime downloads
- Interactive CLI: Query interface with personality and visual feedback

## Quick-Start

``` {bash}

# 1. Copy environment template
cp .env.example .env

# 2. Build the Docker image (first time only)
make build

# 3. Start the librarian
make up

# 4. Add URLs to process
cp urls.txt.example urls.txt
# Edit urls.txt with your document URLs

# 5. Process documents
make process-urls

# 6. Query the collection
make query

```
<!-- 
## Setup

### Prerequisites

- ```Docker```
- ```Make```

### First-Time Setup

``` {bash}
# Copy the environment template
cp .env.example .env
# Edit .env 
nano .env
# Build the Docker image
make build
# Create your URL list
cp urls.txt.example urls.txt

```

## Architecture

``` {mermaid}

---
config:
  theme: 'base'
  themeVariables:
    primaryColor: '#e3f2fd'
    primaryTextColor: '#1565c0'
    primaryBorderColor: '#7C00'
    lineColor: '#F8B229'
    secondaryColor: '#bbdefb'
    tertiaryColor: '#90caf9'
---
mindmap
    root((The Librarian))
            ((Downloader))
            ((Extractor))
                ((zip files))
            ((Classifier))
                ((Type of files)) 
                    ((Text))
                    ((Image))
                    ((PDF)
                    ((unknown))    
            ((Processor))
                ((Text))
                ((Image))
                ((PDF))
            ((Indexer))
                ((Vector indexer))
            ((Retriever))
                ((Query the Librarian via CLI))
```

## Operation

### Makefile Commands — Quick Reference
| Command | Function |
|-------|-----|
| `make build` | Build (or rebuild) the librarian image. |
| `make up` | Start the librarian container. |
| `make process-urls` | Download and index everything listed in urls.txt. |
| `make query` | Open the interactive search prompt. |
| `make backup` | Snapshot the databases to /data/backups. |
| `make restore` | Restore from a backup (details below). |
| `make test` | Run the test suite (will run ephemeral). |
| `make clean` | Delete temporary staging files only. |
| `make clean-all` | Remove staging files and the Docker image. |
| `make shell` | Shell with the project folder mounted read-write (dev use). |
| `make logs` | Tail the container logs. |

### Processing Documents

- Edit urls.txt — one URL per line. Lines starting with # are ignored.

- Then kick off processing:

``` {bash}
make process-urls
```

### Querying The Librarian

``` {bash}
make query
```

### Backup & Restore

``` {bash}
# Create a backup at any time:
make backup
# Backups land in data/backups/ as timestamped folders (e.g. backup_20260201_140000). Each folder contains the two database files and a manifest.txt.

# To restore
make restore BACKUP=/data/backups/backup_20260201_140000
# !NOTE!: Restoring overwrites the current databases. Back up first if you want to keep your current state.

```

### Key Configuration Values (.env)
| Variable | Description |
|-------|-----|
| BATCH_SIZE | How many files are processed before a memory-cleanup cycle. Lower this if you see memory pressure. Default: 10.|
| OCR_ENABLED | Set to false to skip OCR on scanned PDFs. Saves significant CPU time. Default: true.|
| BACKUP_ENABLED | Set to false to disable backup commands. Default: true.|
| DATABASE_PATH | Where the metadata SQLite database lives. Default: /data/database/metadata.db.|
| VECTOR_DB_PATH | Where the embeddings database lives. Default: /data/database/vectors.db.|
| VECTOR_DB_PATH | Where the embeddings database lives. Default: /data/database/vectors.db.|
| VECTOR_DB_PATH | Where the embeddings database lives. Default: /data/database/vectors.db.| -->
