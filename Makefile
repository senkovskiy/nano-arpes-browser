# Makefile
.PHONY: install lint lint-fix format typecheck check test test-cov run clean build

# Install all dependencies
install:
	uv sync --all-extras

# Run linter
lint:
	uv run ruff check src tests

# Run linter with auto-fix
lint-fix:
	uv run ruff check src tests --fix

# Format code
format:
	uv run ruff format src tests

# Run type checking
typecheck:
	uv run mypy src

# Run all checks
check: lint typecheck
	@echo "All checks passed!"

# Run tests
test:
	uv run pytest

# Run tests with coverage
test-cov:
	uv run pytest --cov=src/nano_arpes_browser --cov-report=html

# Run the application
run:
	uv run python -m src.gui.app

# Build package
build:
	uv run python -m build

# Publish to PyPI
publish:
	uv run twine upload dist/*

# Clean build artifacts (cross-platform using Python)
clean:
	uv run python -c "import shutil; import os; [shutil.rmtree(p, ignore_errors=True) for p in ['build', 'dist', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov']]"
	uv run python -c "import pathlib; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
	uv run python -c "import shutil; import pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"

# Show help
help:
	@echo "Available targets:"
	@echo "  install   - Install all dependencies"
	@echo "  lint      - Run linter"
	@echo "  lint-fix  - Run linter with auto-fix"
	@echo "  format    - Format code"
	@echo "  typecheck - Run type checker"
	@echo "  check     - Run all checks"
	@echo "  test      - Run tests"
	@echo "  test-cov  - Run tests with coverage"
	@echo "  run       - Run the application"
	@echo "  build     - Build package"
	@echo "  publish   - Publish to PyPI"
	@echo "  clean     - Clean build artifacts"