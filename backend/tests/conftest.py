"""Pytest configuration and shared fixtures for RAG system tests."""

import os
import sys
import tempfile
from unittest.mock import Mock

import pytest

# Add backend directory to Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults, VectorStore


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing"""
    config = Config()
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 100
    config.CHUNK_OVERLAP = 20
    config.MAX_RESULTS = 3
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = tempfile.mkdtemp()  # Use temp directory for tests
    return config


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    lessons = [
        Lesson(
            lesson_number=1,
            title="Introduction to Testing",
            lesson_link="https://example.com/lesson1",
        ),
        Lesson(
            lesson_number=2,
            title="Advanced Testing Concepts",
            lesson_link="https://example.com/lesson2",
        ),
    ]

    course = Course(
        title="Test Course",
        instructor="Test Instructor",
        course_link="https://example.com/course",
        lessons=lessons,
    )

    return course


@pytest.fixture
def sample_course_chunks(sample_course):
    """Create sample course chunks for testing"""
    chunks = [
        CourseChunk(
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0,
            content="This is the first chunk of lesson 1 content about testing fundamentals.",
        ),
        CourseChunk(
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=1,
            content="This is the second chunk of lesson 1 with more testing information.",
        ),
        CourseChunk(
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=2,
            content="This is lesson 2 content about advanced testing concepts and methodologies.",
        ),
    ]
    return chunks


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock_store = Mock(spec=VectorStore)

    # Configure mock search to return sample results
    mock_results = SearchResults(
        documents=["Sample document content", "Another document"],
        metadata=[
            {"course_title": "Test Course", "lesson_number": 1},
            {"course_title": "Test Course", "lesson_number": 2},
        ],
        distances=[0.1, 0.2],
        error=None,
    )
    mock_store.search.return_value = mock_results
    mock_store._resolve_course_name.return_value = "Test Course"
    mock_store.get_lesson_link.return_value = "https://example.com/lesson1"

    return mock_store


@pytest.fixture
def mock_empty_vector_store():
    """Create a mock vector store that returns empty results"""
    mock_store = Mock(spec=VectorStore)

    empty_results = SearchResults(documents=[], metadata=[], distances=[], error=None)
    mock_store.search.return_value = empty_results
    mock_store._resolve_course_name.return_value = None

    return mock_store


@pytest.fixture
def mock_error_vector_store():
    """Create a mock vector store that returns errors"""
    mock_store = Mock(spec=VectorStore)

    error_results = SearchResults(
        documents=[], metadata=[], distances=[], error="Search error occurred"
    )
    mock_store.search.return_value = error_results
    mock_store._resolve_course_name.side_effect = Exception("Vector store error")

    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    mock_client = Mock()

    # Mock successful response without tools
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "This is a test response"
    mock_response.stop_reason = "end_turn"

    mock_client.messages.create.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_anthropic_client_with_tool_use():
    """Create a mock Anthropic client that uses tools"""
    mock_client = Mock()

    # Mock tool use response
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].type = "tool_use"
    mock_response.content[0].name = "search_course_content"
    mock_response.content[0].input = {"query": "test query"}
    mock_response.content[0].id = "tool_123"
    mock_response.stop_reason = "tool_use"

    # Mock final response after tool execution
    mock_final_response = Mock()
    mock_final_response.content = [Mock()]
    mock_final_response.content[0].text = "Response after tool execution"

    mock_client.messages.create.side_effect = [mock_response, mock_final_response]

    return mock_client


@pytest.fixture
def course_search_tool(mock_vector_store):
    """Create a CourseSearchTool with mock vector store"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def course_outline_tool(mock_vector_store):
    """Create a CourseOutlineTool with mock vector store"""
    return CourseOutlineTool(mock_vector_store)


@pytest.fixture
def tool_manager(course_search_tool, course_outline_tool):
    """Create a ToolManager with registered tools"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    manager.register_tool(course_outline_tool)
    return manager

@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for API testing"""
    mock_system = Mock(spec=RAGSystem)
    
    # Mock session manager
    mock_session_manager = Mock()
    mock_session_manager.create_session.return_value = "test-session-123"
    mock_system.session_manager = mock_session_manager
    
    # Mock query method
    mock_system.query.return_value = (
        "Test response",
        [{"text": "Test source", "url": "https://example.com/test"}]
    )
    
    # Mock analytics method
    mock_system.get_course_analytics.return_value = {
        "total_courses": 1,
        "course_titles": ["Test Course"]
    }
    
    # Mock document loading method
    mock_system.add_course_folder.return_value = (1, 3)  # 1 course, 3 chunks
    
    return mock_system

@pytest.fixture
def mock_session_manager():
    """Create a mock session manager for testing"""
    mock_manager = Mock()
    mock_manager.create_session.return_value = "test-session-123"
    mock_manager.get_session.return_value = {"history": []}
    mock_manager.add_to_session.return_value = None
    return mock_manager

@pytest.fixture
def api_test_data():
    """Common test data for API testing"""
    return {
        "valid_query": {
            "query": "What is testing?",
            "session_id": None
        },
        "query_with_session": {
            "query": "Follow up question",
            "session_id": "existing-session-456"
        },
        "empty_query": {
            "query": "",
            "session_id": None
        },
        "long_query": {
            "query": "test " * 1000,
            "session_id": None
        },
        "expected_response": {
            "answer": "Test response about the query",
            "sources": [
                {"text": "Source chunk 1", "url": "https://example.com/lesson1"},
                {"text": "Source chunk 2", "url": "https://example.com/lesson2"}
            ],
            "session_id": "test-session-123"
        },
        "expected_analytics": {
            "total_courses": 3,
            "course_titles": ["Course 1", "Course 2", "Course 3"]
        }
    }

@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic API response for testing"""
    response_data = {
        "content": [{
            "text": "This is a test response from the AI model",
            "type": "text"
        }],
        "model": "claude-sonnet-4-20250514",
        "role": "assistant",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50
        }
    }
    return response_data

@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_dirs():
    """Clean up temporary directories after tests"""
    yield
    # Cleanup happens automatically for temporary directories
    pass
