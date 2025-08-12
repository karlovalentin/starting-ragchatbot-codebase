#!/bin/bash

# Type checking with MyPy
set -e

echo "🔍 Running type checking with MyPy..."
uv run mypy backend/ main.py

echo ""
echo "✅ Type checking complete!"