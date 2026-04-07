# NOTE: Just used to simplify running commands :)

.PHONY: help setup lint data

help:
	@echo "Available commands:"
	@echo ""
	@echo "  help   : Show this help message"
	@echo "  setup  : Create virtual environment and install dependencies"
	@echo "  data   : Download Kaggle dataset to data/raw"
	@echo "  lint   : Run lint" 
	@echo ""

setup:
	uv venv --python 3.12
	uv sync

lint:
	uv run ruff check src

data:
	uv run python scripts/download_data.py