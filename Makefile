
.PHONY: build up down process-urls query backup restore test clean clean-all shell logs help


help:
	@echo "Document Librarian - Makefile Commands"
	@echo ""
	@echo "  make build          - Build the Docker image"
	@echo "  make up             - Start the librarian container"
	@echo "  make down           - Stop and remove the librarian container"
	@echo "  make process-urls   - Download and index everything in urls.txt"
	@echo "  make query          - Open an interactive query session"
	@echo "  make backup         - Snapshot databases to data/backups/"
	@echo "  make restore        - Restore from a backup (set BACKUP=path)"
	@echo "  make test           - Run the test suite"
	@echo "  make shell          - Drop into a bash shell inside the container"
	@echo "  make logs           - Tail the container logs"
	@echo "  make clean          - Delete staging/temp files inside the container"
	@echo "  make clean-all      - Stop container, remove image, wipe staging/"
	@echo ""

# --- image & lifecycle -----------------------------------------------------

build:
	docker compose build

up:
	docker compose up -d librarian

down:
	docker compose down librarian

# --- workload --------------------------------------------------------------
# These all exec into the already-running librarian container so the
# embedding model stays warm between runs.

process-urls:
	@if [ ! -f urls.txt ]; then \
		echo "Error: urls.txt not found"; \
		echo "Create urls.txt with one URL per line"; \
		exit 1; \
	fi
	docker compose exec librarian python main.py process --url-file /app/urls.txt

query:
	docker compose exec -it librarian python main.py query --interactive

backup:
	docker compose exec librarian python main.py backup

restore:
	@if [ -z "$(BACKUP)" ]; then \
		echo "Error: BACKUP variable not set"; \
		echo "Usage: make restore BACKUP=/data/backups/backup_YYYYMMDD_HHMMSS"; \
		exit 1; \
	fi
	docker compose exec librarian python main.py backup --restore $(BACKUP)

# --- tests -----------------------------------------------------------------
# Runs the ephemeral librarian-test service — starts, runs pytest, exits.

test:
	docker compose run --rm librarian-test

# --- housekeeping ----------------------------------------------------------

shell:
	docker compose exec -it librarian /bin/bash

logs:
	docker compose logs -f librarian

clean:
	docker compose exec librarian python -c \
		"from src.orchestrator import Orchestrator; o = Orchestrator(); o._cleanup_staging()"

clean-all: down
	docker compose down --rmi all
	rm -rf data/staging/*
	@echo "Stopped container, removed image, wiped staging/"
