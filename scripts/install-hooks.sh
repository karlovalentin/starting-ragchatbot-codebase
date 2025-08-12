#!/bin/bash

# Install pre-commit hooks
set -e

echo "🔧 Installing pre-commit hooks..."
uv run pre-commit install

echo ""
echo "✅ Pre-commit hooks installed!"
echo ""
echo "To run hooks manually on all files: ./scripts/run-hooks.sh"