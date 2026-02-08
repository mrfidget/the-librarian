# Troubleshooting Guide

Solutions to common problems when using The Librarian.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Processing Issues](#processing-issues)
- [Query Issues](#query-issues)
- [Performance Issues](#performance-issues)
- [Database Issues](#database-issues)
- [Docker Issues](#docker-issues)
- [Model Issues](#model-issues)

## Installation Issues

### Docker Compose Command Not Found

**Symptom**:
```
make: docker compose: Command not found
```

**Solution**:

You have an older Docker version. Use `docker-compose` (with hyphen):

```bash
# Check version
docker-compose --version

# If this works, update Makefile
# Replace all "docker compose" with "docker-compose"
```

Or upgrade Docker to 20.10+.

### Build Fails with "No Space Left on Device"

**Symptom**:
```
ERROR: failed to solve: write /var/lib/docker/...: no space left on device
```

**Solution**:

```bash
# Clean up Docker
docker system prune -a

# Check disk space
df -h

# If still low, increase Docker disk limit in Docker Desktop settings
```

### Build Takes Forever / Hangs

**Symptom**: Build stuck downloading models

**Solution**:

Check network connection and retry:

```bash
# Cancel build
Ctrl+C

# Try again
make build

# If still stuck, check if you can reach HuggingFace
curl -I https://huggingface.co
```

### Permission Denied Errors

**Symptom**:
```
Permission denied: './data'
```

**Solution**:

```bash
# Fix ownership
sudo chown -R $USER:$USER data/

# Or run with sudo (not recommended)
sudo make up
```

## Processing Issues

### "Models Downloading at Runtime"

**Symptom**: Progress bars appear when processing, not just at startup

**Solution**:

The offline cache didn't work. Rebuild completely:

```bash
make clean-all
make build
make up
```

Check that these environment variables are set in Dockerfile:
```dockerfile
ENV HF_HUB_OFFLINE=1
ENV HF_HOME=/opt/hf_cache
```

### "No Such Column: library_path"

**Symptom**:
```
sqlite3.OperationalError: no such column: f.library_path
```

**Solution**:

Database schema is outdated. Delete and recreate:

```bash
make down
rm -rf data/database/*
make up
make process-urls
```

**To preserve data**: Export to backup first:

```bash
make backup
# Note the backup path
make down
rm -rf data/database/*
make up
# Re-process all URLs (references preserved in library/)
```

### "Duplicate File" but It's Not a Duplicate

**Symptom**: Different files marked as duplicates

**Solution**:

Files have identical content (same SHA-256). This is correct behavior. If they're truly different:

1. Check files aren't byte-identical:
```bash
diff file1.pdf file2.pdf
```

2. If genuinely different and being marked duplicate, it's a bug. File an issue with:
   - Original URLs
   - File sizes
   - Checksums from logs

### OCR Not Working

**Symptom**: Scanned PDFs have no extracted text

**Check OCR is enabled**:

```bash
# In .env
OCR_ENABLED=true
```

**Check Tesseract is installed**:

```bash
make shell
tesseract --version
# Should show version 4.x or 5.x
```

**Check logs for OCR attempts**:

```bash
make logs | grep "OCR"
# Should see "Minimal text (N chars), applying OCR..."
```

### Processing Hangs

**Symptom**: Processing stuck on one file

**Solution**:

Check logs for last file processed:

```bash
make logs
```

If it's a specific file causing issues:

1. Remove that URL from `urls.txt`
2. Continue processing other files
3. Investigate problem file separately

Common causes:
- Corrupted file (redownload)
- Extremely large file (>500MB PDFs)
- Infinite loop in PDF extractor (file a bug report)

### Out of Memory During Processing

**Symptom**:
```
Killed
```
or container crashes without error

**Solution**:

Reduce batch size:

```env
# .env
BATCH_SIZE=3
```

Restart:

```bash
make down
make up
make process-urls
```

For very large files, process fewer at once:

```bash
# Edit urls.txt to have only 5-10 URLs at a time
make process-urls
```

## Query Issues

### "No Model Client Available"

**Symptom**:
```
Warning: No model client available for semantic search
```

**Solution**:

Model worker process crashed or didn't start. Check logs:

```bash
make logs | grep ModelServer
```

Should see:
```
[ModelServer] Loading models...
[ModelServer] All models loaded and ready
```

If not there, restart:

```bash
make down
make up
```

### Search Returns Nothing

**Symptom**: All queries return "no results found"

**Causes**:

1. **Nothing indexed yet**:
```bash
# Check if database has files
make shell
sqlite3 /data/database/metadata.db "SELECT COUNT(*) FROM files;"
```

2. **Threshold too high**:
```env
# .env - Lower threshold
SEARCH_THRESHOLD=0.15
```

3. **Embeddings not created**:
```bash
# Check vector database
make shell
sqlite3 /data/database/vectors.db "SELECT COUNT(*) FROM embeddings;"
# Should match file count
```

### Search Results Don't Make Sense

**Symptom**: Irrelevant results ranking highly

**Explanation**: Semantic search is based on AI embeddings, not keyword matching.

**Solutions**:

1. **Use exact match for keywords**:
```
📚 Ask the librarian: "exact phrase here"
```

2. **Be more specific**:
```
# Bad
"documents"

# Good  
"legal contracts from 2024"
```

3. **Lower threshold to see why results ranked as they did**:
```env
SEARCH_THRESHOLD=0.10
```

### Ms. Clarke Not Appearing

**Symptom**: Plain text output instead of formatted interface

**Cause**: Not in interactive mode

**Solution**:

```bash
# Correct
make query

# This will give plain output
docker compose exec librarian python main.py query --query "test"
```

## Performance Issues

### Slow Processing

**Common causes**:

1. **OCR on digital PDFs**:
   - Check logs for "applying OCR" on pages that shouldn't need it
   - Increase `OCR_TEXT_THRESHOLD` if too many pages trigger OCR

2. **Large batch size**:
```env
BATCH_SIZE=5  # Reduce from 10
```

3. **Insufficient resources**:
```bash
# Check container stats
docker stats the-librarian
```

### Slow Queries

**Causes**:

1. **Model not cached** (rare after first startup):
   - Check logs - model should load once at startup

2. **Large database**:
   - Vacuum database periodically:
```bash
make shell
sqlite3 /data/database/metadata.db "VACUUM;"
sqlite3 /data/database/vectors.db "VACUUM;"
```

3. **High result limit**:
   - Reduce limit in queries (default is 10)

### Memory Usage Growing

**Symptom**: Container using more RAM over time

**Solutions**:

1. **Restart container** (temporary fix):
```bash
make down
make up
```

2. **Reduce batch size** (prevent growth):
```env
BATCH_SIZE=5
```

3. **Check for memory leaks** (file issue if persistent):
```bash
docker stats the-librarian
# Watch over time
```

## Database Issues

### Database Locked

**Symptom**:
```
sqlite3.OperationalError: database is locked
```

**Solution**:

Another process has the database open:

```bash
# Check what's accessing it
make shell
lsof /data/database/metadata.db

# Force close container and restart
make down
docker rm -f the-librarian
make up
```

### Corrupted Database

**Symptom**: Various SQL errors, inconsistent results

**Solution**:

1. **Try repair**:
```bash
make shell
sqlite3 /data/database/metadata.db
.recover
```

2. **Restore from backup**:
```bash
make restore BACKUP=/data/backups/backup_YYYYMMDD_HHMMSS
```

3. **Rebuild** (last resort):
```bash
make backup  # Save library files
make down
rm -rf data/database/*
make up
make process-urls  # Reprocess all URLs
```

### Wrong Schema Version

**Symptom**: Missing columns, unexpected fields

**Solution**:

Schema mismatch. Delete database and rebuild:

```bash
make down
rm -rf data/database/*
make up
```

## Docker Issues

### Container Exits Immediately

**Symptom**:
```
docker compose ps
# Shows "Exited (1)"
```

**Check logs**:

```bash
make logs
```

Common causes:
- Model server crash
- Permission issues
- Missing dependencies

### Can't Connect to Container

**Symptom**:
```
Error: No such container: the-librarian
```

**Solution**:

```bash
# Check if running
docker compose ps

# If not, start it
make up

# Check again
docker compose ps
```

### Port Already in Use

**Symptom** (if exposing ports):
```
Error: port is already allocated
```

**Solution**:

```bash
# Find what's using the port
sudo lsof -i :8000

# Kill it or change port in docker-compose.yml
```

### Volume Mount Fails

**Symptom**:
```
Error: invalid mount config
```

**Solution**:

Check docker-compose.yml paths:

```yaml
# Should be relative or absolute
volumes:
  - ./data:/data  # Relative OK
  - /absolute/path/data:/data  # Absolute OK
  - ~/data:/data  # May not work, use $HOME/data
```

## Model Issues

### "Models Not Found" at Runtime

**Symptom**:
```
OSError: Model 'sentence-transformers/all-MiniLM-L6-v2' not found
```

**Solution**:

Cache path mismatch. Rebuild with explicit paths:

```bash
make clean-all
make build
```

Verify in logs:
```
[ModelServer] Loading models...
[ModelServer] All models loaded and ready
```

### Wrong Model Loaded

**Symptom**: Unexpected model behavior

**Check which model is loaded**:

```bash
make shell
echo $EMBEDDING_MODEL
echo $VISION_MODEL
```

Should match `.env` settings.

### Model Worker Crash

**Symptom**:
```
[ModelServer] Fatal error: ...
```

**Solution**:

1. **Check available memory**:
```bash
docker stats the-librarian
# Should have >1GB available
```

2. **Restart container**:
```bash
make down
make up
```

3. **If persistent**, file issue with:
   - Full error from logs
   - System specs
   - Model being used

## Getting More Help

### Collect Diagnostic Info

```bash
# System info
docker --version
docker compose version
make logs > librarian-logs.txt

# Database info
make shell
sqlite3 /data/database/metadata.db "SELECT COUNT(*) FROM files;" > db-stats.txt
sqlite3 /data/database/vectors.db "SELECT COUNT(*) FROM embeddings;" >> db-stats.txt
exit

# Config
cat .env
```

### Check Known Issues

- GitHub Issues: [your-repo-url]/issues
- Search closed issues for similar problems

### Report a Bug

Include:
1. What you were trying to do
2. What happened instead
3. Error messages (full text)
4. Logs (`make logs`)
5. System info (OS, Docker version, RAM)
6. Configuration (`.env` file, sanitized)

### Community Help

- GitHub Discussions
- Stack Overflow (tag: document-librarian)
- Discord (if available)

## Preventive Maintenance

### Regular Tasks

```bash
# Weekly
make backup

# Monthly
make shell
sqlite3 /data/database/metadata.db "VACUUM;"
sqlite3 /data/database/vectors.db "VACUUM;"

# After major updates
make clean-all
make build
make up
```

### Monitoring

```bash
# Check disk space
df -h data/

# Check database sizes
du -sh data/database/

# Check memory usage
docker stats the-librarian
```

### Log Rotation

Logs rotate automatically (configured in docker-compose.yml):
- Max size: 10MB per file
- Max files: 3
- Total: 30MB max

To view old logs:
```bash
docker compose logs --tail=1000 librarian
```