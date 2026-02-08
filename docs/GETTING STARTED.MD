# Getting Started

This guide walks you through installing and running The Librarian for the first time.

## Prerequisites

Before you begin, ensure you have:

- **Docker** (version 20.10 or higher)
- **Docker Compose** (version 2.0 or higher)
- **Make** (usually pre-installed on Linux/Mac)
- **4GB RAM minimum** (8GB recommended)
- **10GB free disk space**

### Verify Prerequisites

```bash
docker --version        # Should show 20.10+
docker compose version  # Should show 2.0+
make --version         # Any recent version
```

## Installation

### 1. Clone or Download the Repository

```bash
git clone [your-repo-url]
cd the-librarian
```

### 2. Configure Environment

Copy the example environment file and customize it:

```bash
cp .env.example .env
```

Edit `.env` if you want to change defaults (optional for first run):

```bash
nano .env  # or vim, or any text editor
```

See [Configuration](configuration.md) for all available options.

### 3. Build the Docker Image

This downloads and caches all AI models (~2.5GB). First build takes 10-15 minutes:

```bash
make build
```

The build process:
- Installs system dependencies
- Downloads PyTorch (CPU-only)
- Downloads CLIP and MiniLM models
- Builds sqlite-vec extension
- Creates the runtime image

### 4. Start The Librarian

```bash
make up
```

This starts the container in the background and loads AI models into memory (~30 seconds).

Check that it's running:

```bash
docker compose ps
# Should show "the-librarian" with status "Up"
```

## First Document Processing

### 1. Create URL List

```bash
cp urls.txt.example urls.txt
nano urls.txt
```

Add URLs to process, one per line:

```
https://example.com/document.pdf
https://example.com/archive.zip
file:///app/local-file.pdf
```

**Note**: For local files, use `file:///app/` prefix and mount the file in docker-compose.yml.

### 2. Process Documents

```bash
make process-urls
```

You'll see output like:

```
Processing 3 URL(s)...
[ModelServer] All models loaded and ready
Downloading https://example.com/document.pdf...
Processing document.pdf as pdf...
  Page 1: Minimal text (12 chars), applying OCR...
OK: document.pdf
Successfully processed 1 files
```

### 3. Query Your Collection

```bash
make query
```

This opens the interactive query interface:

```
================================================================================
The Librarian — Ms. Clarke, Head Librarian
================================================================================

*Ms. Clarke adjusts her reading glasses and looks up from her desk*

"Good afternoon. State your inquiry clearly, please.
And do remember: I expect precision. None of this
'find me stuff about things' nonsense."

Type 'quit' or 'exit' to leave.

================================================================================

Ask Ms. Clarke: 
```

Try a query like:

```
Ask Ms. Clarke: documents about contracts

*Ms. Clarke returns promptly, looking rather pleased*

"Ah yes. I thought so. Here's what you're after:"

Rank   Stars      Type     Filename                       Description
----------------------------------------------------------------------------------------------------
1      ★★★★★      pdf      service-agreement.pdf          Legal document discussing contract...
2      ★★★★☆      text     terms-draft.txt                Draft contract for review

*2 item(s) found. Files are in /data/library/*
```

## Next Steps

- **[User Guide](user-guide.md)** - Learn how to use all features
- **[Configuration](configuration.md)** - Customize behavior and performance
- **[Architecture](architecture.md)** - Understand how it works

## Common First-Time Issues

### "No such command: docker compose"

You might have an older Docker version. Try `docker-compose` (with hyphen):

```bash
docker-compose --version
```

If this works, you'll need to replace `docker compose` with `docker-compose` in the Makefile.

### "Models downloading at runtime"

The offline cache didn't work. Rebuild completely:

```bash
make clean-all
make build
```

### "Container immediately exits"

Check logs:

```bash
make logs
```

Usually this means the model server crashed during startup. Check available RAM.

### "Out of memory"

Reduce `BATCH_SIZE` in `.env`:

```
BATCH_SIZE=5
```

Then restart:

```bash
make down
make up
```
