#!/bin/bash

# Run pre-commit hooks on all files
set -e

echo "🔧 Running pre-commit hooks on all files..."
uv run pre-commit run --all-files

echo ""
echo "✅ Pre-commit hooks completed!"