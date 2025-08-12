# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a full-stack RAG (Retrieval-Augmented Generation) system that enables users to query course materials and receive intelligent, context-aware responses. The application uses:

- **Backend**: Python FastAPI with modular RAG architecture
- **Vector Database**: ChromaDB for semantic search and document storage
- **AI Generation**: Anthropic's Claude API for response generation
- **Frontend**: Static HTML/CSS/JavaScript served by FastAPI
- **Document Processing**: Automated chunking and embedding of course materials

### Core Components

The backend follows a modular architecture with clear separation of concerns:

- `rag_system.py` - Main orchestrator that coordinates all components
- `vector_store.py` - ChromaDB integration for semantic search
- `ai_generator.py` - Claude API integration for response generation
- `document_processor.py` - Text chunking and course material processing
- `session_manager.py` - Conversation history management
- `search_tools.py` - Tool-based search functionality
- `models.py` - Data models for courses, chunks, and responses
- `config.py` - Configuration management with environment variables

## Development Commands

### Installation and Setup
```bash
# Install dependencies using uv
uv sync

# Set up environment variables
# Create .env file with:
# ANTHROPIC_API_KEY=your_api_key_here
```

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh
./run.sh

# Manual start
cd backend
uv run uvicorn app:app --reload --port 8000
```

### Testing and Development
```bash
# Run the application in development mode
cd backend
uv run uvicorn app:app --reload --port 8000

# The application serves:
# - Web interface at http://localhost:8000
# - API documentation at http://localhost:8000/docs
```

## Configuration

The system uses environment variables loaded from a `.env` file:

- `ANTHROPIC_API_KEY` - Required for Claude API access
- Default model: `claude-sonnet-4-20250514`
- Embedding model: `all-MiniLM-L6-v2` (sentence-transformers)
- Document chunks: 800 characters with 100 character overlap
- ChromaDB path: `./chroma_db`

## API Endpoints

- `POST /api/query` - Process user queries against course materials
- `GET /api/courses` - Get course analytics and statistics
- `GET /` - Serve frontend static files

## Document Processing

Course materials are automatically loaded from the `docs/` directory on startup. The system:

1. Processes text files into semantic chunks
2. Creates embeddings using sentence-transformers
3. Stores vectors in ChromaDB for semantic search
4. Maintains conversation history for context-aware responses

## Development Notes

- The application uses FastAPI's automatic startup event to load documents
- CORS is enabled for development with wildcard origins
- Static files are served with no-cache headers for development
- Session management tracks conversation history for better context
- The vector store maintains separate collections for course metadata and content chunks

## Tool and Workflow Memories

- Use `uv` to run python files or add any dependencies

## Code Quality Tools

The project includes comprehensive code quality tools:

### Available Scripts
- `./scripts/quality.sh` - Run all quality checks (format, lint, type check, test)
- `./scripts/format.sh` - Format code with Black and fix imports
- `./scripts/lint.sh` - Run Ruff linting
- `./scripts/typecheck.sh` - Run MyPy type checking
- `./scripts/install-hooks.sh` - Install pre-commit hooks
- `./scripts/run-hooks.sh` - Run pre-commit hooks on all files

### Tools Configuration
- **Black**: Code formatting (88 character line length)
- **Ruff**: Fast linting and import sorting
- **MyPy**: Static type checking
- **Pre-commit**: Automated quality checks on commit

### Development Workflow
```bash
# Install pre-commit hooks (run once)
./scripts/install-hooks.sh

# Format code before committing
./scripts/format.sh

# Run all quality checks
./scripts/quality.sh
```

## Vector Database Configuration

- The vector database has two collections:
  - `course_catalog`: stores course titles for name resolution
    - Metadata for each course includes:
      - title
      - instructor
      - course_link
      - lesson_count
      - lessons_json (list of lessons with lesson_number, lesson_title, lesson_link)
  - `course_content`: stores text chunks for semantic search
    - Metadata for each chunk includes:
      - course_title
      - lesson_number
      - chunk_index