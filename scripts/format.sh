#!/bin/bash

# Format code with Black and fix imports with Ruff
set -e

echo "📝 Formatting code with Black..."
uv run black backend/ main.py

echo ""
echo "🔧 Fixing imports with Ruff..."
uv run ruff check --select I --fix backend/ main.py

echo ""
echo "✅ Code formatting complete!"