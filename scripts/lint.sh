#!/bin/bash

# Lint code with Ruff
set -e

echo "🔧 Running Ruff linting..."
uv run ruff check backend/ main.py

echo ""
echo "✅ Linting complete!"