# NOTE: Just used to simplify running commands :)

.PHONY: help setup lint

help:
	@echo "Available commands:"
	@echo ""
	@echo "  help   : Show this help message"
	@echo "  setup  : Create virtual environment and install dependencies"
	@echo "  lint   : Run lint" 
	@echo ""

setup:
	uv venv --python 3.12
	uv sync

lint:
	uv run ruff check src