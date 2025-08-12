"""Testing utilities and helper functions for RAG system tests."""

import os
import shutil
import tempfile
from typing import Any
from unittest.mock import Mock, patch

import chromadb
from chromadb.config import Settings
from models import Course, CourseChunk
from vector_store import SearchResults


class TestVectorStore:
    """Test-specific vector store with in-memory ChromaDB"""

    def __init__(self):
        # Use in-memory ChromaDB for testing
        self.temp_dir = tempfile.mkdtemp()
        self.client = chromadb.PersistentClient(
            path=self.temp_dir, settings=Settings(anonymized_telemetry=False)
        )

        # Create mock embedding function that returns simple vectors
        self.embedding_function = self._create_mock_embedding_function()

        # Create collections
        self.course_catalog = self.client.get_or_create_collection(
            name="course_catalog", embedding_function=self.embedding_function
        )
        self.course_content = self.client.get_or_create_collection(
            name="course_content", embedding_function=self.embedding_function
        )

    def _create_mock_embedding_function(self):
        """Create a mock embedding function for testing"""

        class MockEmbeddingFunction:
            def __call__(self, input):
                # Return simple fixed-size embeddings for testing
                if isinstance(input, list):
                    return [[0.1] * 384 for _ in input]  # 384-dimensional vectors
                return [0.1] * 384

        return MockEmbeddingFunction()

    def add_test_data(self, courses: list[Course], chunks: list[CourseChunk]):
        """Add test data to the vector store"""
        # Add course metadata
        for course in courses:
            import json

            lessons_metadata = []
            for lesson in course.lessons:
                lessons_metadata.append(
                    {
                        "lesson_number": lesson.lesson_number,
                        "lesson_title": lesson.title,
                        "lesson_link": lesson.lesson_link,
                    }
                )

            self.course_catalog.add(
                documents=[course.title],
                metadatas=[
                    {
                        "title": course.title,
                        "instructor": course.instructor,
                        "course_link": course.course_link,
                        "lessons_json": json.dumps(lessons_metadata),
                        "lesson_count": len(course.lessons),
                    }
                ],
                ids=[course.title],
            )

        # Add course content
        if chunks:
            documents = [chunk.content for chunk in chunks]
            metadatas = [
                {
                    "course_title": chunk.course_title,
                    "lesson_number": chunk.lesson_number,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ]
            ids = [
                f"{chunk.course_title.replace(' ', '_')}_{chunk.chunk_index}"
                for chunk in chunks
            ]

            self.course_content.add(documents=documents, metadatas=metadatas, ids=ids)

    def cleanup(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def create_mock_search_results(
    documents: list[str],
    metadata: list[dict[str, Any]],
    distances: list[float] | None = None,
    error: str | None = None,
) -> SearchResults:
    """Create SearchResults object for testing"""
    if distances is None:
        distances = [0.1] * len(documents)

    return SearchResults(
        documents=documents, metadata=metadata, distances=distances, error=error
    )


def assert_search_results_valid(results: SearchResults):
    """Assert that search results are in valid format"""
    assert results is not None
    assert isinstance(results.documents, list)
    assert isinstance(results.metadata, list)
    assert isinstance(results.distances, list)

    # If not empty, lengths should match
    if results.documents:
        assert len(results.documents) == len(results.metadata)
        assert len(results.documents) == len(results.distances)


def assert_tool_definition_valid(tool_definition: dict[str, Any]):
    """Assert that a tool definition is valid for Anthropic API"""
    required_fields = ["name", "description", "input_schema"]
    for field in required_fields:
        assert (
            field in tool_definition
        ), f"Tool definition missing required field: {field}"

    assert isinstance(tool_definition["name"], str)
    assert isinstance(tool_definition["description"], str)
    assert isinstance(tool_definition["input_schema"], dict)

    # Check input schema structure
    schema = tool_definition["input_schema"]
    assert schema.get("type") == "object"
    assert "properties" in schema
    assert isinstance(schema["properties"], dict)


def mock_anthropic_response(
    text: str, stop_reason: str = "end_turn", tool_calls: list[dict] | None = None
):
    """Create a mock Anthropic API response"""
    mock_response = Mock()

    if tool_calls:
        mock_response.content = []
        for tool_call in tool_calls:
            mock_content = Mock()
            mock_content.type = "tool_use"
            mock_content.name = tool_call["name"]
            mock_content.input = tool_call["input"]
            mock_content.id = tool_call.get("id", "tool_123")
            mock_response.content.append(mock_content)
        mock_response.stop_reason = "tool_use"
    else:
        mock_content = Mock()
        mock_content.text = text
        mock_response.content = [mock_content]
        mock_response.stop_reason = stop_reason

    return mock_response


class MockAnthropicClient:
    """Mock Anthropic client for testing"""

    def __init__(self, responses: list[dict[str, Any]] = None):
        self.responses = responses or []
        self.call_count = 0
        self.messages = Mock()
        self.messages.create = Mock(side_effect=self._create_response)

    def _create_response(self, **kwargs):
        """Mock the messages.create method"""
        if self.call_count < len(self.responses):
            response_data = self.responses[self.call_count]
            self.call_count += 1

            mock_response = Mock()
            mock_response.content = []

            if "text" in response_data:
                mock_content = Mock()
                mock_content.text = response_data["text"]
                mock_response.content = [mock_content]
                mock_response.stop_reason = response_data.get("stop_reason", "end_turn")

            elif "tool_calls" in response_data:
                for tool_call in response_data["tool_calls"]:
                    mock_content = Mock()
                    mock_content.type = "tool_use"
                    mock_content.name = tool_call["name"]
                    mock_content.input = tool_call["input"]
                    mock_content.id = tool_call.get("id", f"tool_{self.call_count}")
                    mock_response.content.append(mock_content)
                mock_response.stop_reason = "tool_use"

            return mock_response

        # Default response if no more responses configured
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = "Default test response"
        mock_response.content = [mock_content]
        mock_response.stop_reason = "end_turn"
        return mock_response


def capture_tool_calls(tool_manager):
    """Capture and return tool calls made during testing"""
    original_execute = tool_manager.execute_tool
    captured_calls = []

    def mock_execute(tool_name: str, **kwargs):
        captured_calls.append({"tool_name": tool_name, "kwargs": kwargs})
        return original_execute(tool_name, **kwargs)

    tool_manager.execute_tool = mock_execute
    return captured_calls


def create_test_course_document(filename: str, content: str) -> str:
    """Create a temporary course document file for testing"""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return file_path


def cleanup_test_files(*file_paths):
    """Clean up test files and directories"""
    for file_path in file_paths:
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def patch_config_for_testing():
    """Context manager to patch config for testing"""
    return patch(
        "config.config",
        Mock(
            ANTHROPIC_API_KEY="test-key",
            ANTHROPIC_MODEL="test-model",
            EMBEDDING_MODEL="test-embedding-model",
            CHUNK_SIZE=100,
            CHUNK_OVERLAP=20,
            MAX_RESULTS=3,
            MAX_HISTORY=2,
            CHROMA_PATH=tempfile.mkdtemp(),
        ),
    )


def assert_no_errors_in_logs(caplog):
    """Assert that no ERROR level logs were captured"""
    errors = [record for record in caplog.records if record.levelname == "ERROR"]
    assert not errors, f"Unexpected errors in logs: {[r.message for r in errors]}"


def assert_contains_keywords(
    text: str, keywords: list[str], case_sensitive: bool = False
):
    """Assert that text contains all specified keywords"""
    if not case_sensitive:
        text = text.lower()
        keywords = [k.lower() for k in keywords]

    for keyword in keywords:
        assert keyword in text, f"Text does not contain expected keyword: '{keyword}'"
