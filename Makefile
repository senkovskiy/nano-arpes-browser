# Makefile
.PHONY: install lint lint-fix format typecheck check test test-cov run clean build package publish verify-install release-check help

# Install all dependencies
install:
	uv sync --group dev

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
	QT_QPA_PLATFORM=offscreen uv run pytest

# Run tests with coverage
test-cov:
	uv run pytest --cov=src/nano_arpes_browser --cov-report=html

# Run the application
run:
	uv run nano-arpes-browser

# Build package
build:
	uv run python -m build

# Clean, build, and validate package artifacts
package: clean build
	uv run twine check dist/*

# Run checks expected before a release
release-check: lint test package
	git status --short

# Publish existing dist artifacts to PyPI
publish:
	uv publish dist/*

# Reinstall latest PyPI release as a CLI smoke test
verify-install:
	uv tool install nano-arpes-browser --force
	nano-arpes-browser --help

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
	@echo "  package   - Clean, build, and validate dist"
	@echo "  release-check - Run release checks"
	@echo "  publish   - Publish existing dist to PyPI"
	@echo "  verify-install - Reinstall from PyPI and run CLI smoke test"
	@echo "  clean     - Clean build artifacts"
