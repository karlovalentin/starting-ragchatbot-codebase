#!/bin/bash

# Quality check script for the RAG chatbot project
set -e

echo "🔍 Running code quality checks..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Please run from project root."
    exit 1
fi

echo ""
echo "📝 Formatting code with Black..."
uv run black backend/ main.py

echo ""
echo "🔧 Running Ruff linting..."
uv run ruff check backend/ main.py

echo ""
echo "🔧 Running Ruff import sorting..."
uv run ruff check --select I --fix backend/ main.py

echo ""
echo "🔍 Running type checking with MyPy..."
uv run mypy backend/ main.py

echo ""
echo "🧪 Running tests..."
uv run python -m pytest

echo ""
echo "✅ All quality checks passed!"