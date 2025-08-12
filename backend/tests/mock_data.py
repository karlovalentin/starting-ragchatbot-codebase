"""Mock data and utilities for testing RAG system components."""

from typing import Any

from models import Course, CourseChunk, Lesson

# Sample course data for testing
SAMPLE_COURSES = [
    {
        "title": "Introduction to Machine Learning",
        "instructor": "Dr. Andrew Ng",
        "course_link": "https://example.com/ml-course",
        "lessons": [
            {
                "lesson_number": 1,
                "title": "What is Machine Learning?",
                "lesson_link": "https://example.com/ml/lesson1",
            },
            {
                "lesson_number": 2,
                "title": "Supervised Learning",
                "lesson_link": "https://example.com/ml/lesson2",
            },
            {
                "lesson_number": 3,
                "title": "Linear Regression",
                "lesson_link": "https://example.com/ml/lesson3",
            },
        ],
    },
    {
        "title": "Building with MCP",
        "instructor": "Anthropic Team",
        "course_link": "https://example.com/mcp-course",
        "lessons": [
            {
                "lesson_number": 1,
                "title": "MCP Introduction",
                "lesson_link": "https://example.com/mcp/lesson1",
            },
            {
                "lesson_number": 2,
                "title": "Tool Development",
                "lesson_link": "https://example.com/mcp/lesson2",
            },
        ],
    },
]

# Sample course content chunks
SAMPLE_CHUNKS = [
    {
        "course_title": "Introduction to Machine Learning",
        "lesson_number": 1,
        "chunk_index": 0,
        "content": "Machine learning is a subset of artificial intelligence that focuses on developing algorithms that can learn and make decisions from data without being explicitly programmed.",
    },
    {
        "course_title": "Introduction to Machine Learning",
        "lesson_number": 1,
        "chunk_index": 1,
        "content": "The key concept in machine learning is that systems can automatically learn and improve from experience. This involves training models on data to recognize patterns.",
    },
    {
        "course_title": "Introduction to Machine Learning",
        "lesson_number": 2,
        "chunk_index": 2,
        "content": "Supervised learning uses labeled training data to learn a mapping function from inputs to outputs. Examples include classification and regression tasks.",
    },
    {
        "course_title": "Building with MCP",
        "lesson_number": 1,
        "chunk_index": 0,
        "content": "The Model Context Protocol (MCP) enables secure connections between AI applications and data sources. MCP makes it easy for developers to build AI-powered applications.",
    },
]

# Sample search queries and expected results
TEST_QUERIES = [
    {
        "query": "What is machine learning?",
        "expected_course": "Introduction to Machine Learning",
        "expected_lesson": 1,
        "should_find_results": True,
    },
    {
        "query": "How does MCP work?",
        "expected_course": "Building with MCP",
        "expected_lesson": 1,
        "should_find_results": True,
    },
    {
        "query": "Supervised learning examples",
        "expected_course": "Introduction to Machine Learning",
        "expected_lesson": 2,
        "should_find_results": True,
    },
    {
        "query": "Quantum physics fundamentals",
        "expected_course": None,
        "expected_lesson": None,
        "should_find_results": False,
    },
]

# Mock Anthropic API responses
MOCK_API_RESPONSES = {
    "simple_response": {
        "content": [{"text": "This is a simple response without tool usage"}],
        "stop_reason": "end_turn",
    },
    "tool_use_response": {
        "content": [
            {
                "type": "tool_use",
                "name": "search_course_content",
                "input": {"query": "machine learning"},
                "id": "tool_123",
            }
        ],
        "stop_reason": "tool_use",
    },
    "outline_tool_response": {
        "content": [
            {
                "type": "tool_use",
                "name": "get_course_outline",
                "input": {"course_name": "Machine Learning"},
                "id": "tool_456",
            }
        ],
        "stop_reason": "tool_use",
    },
    "final_response_after_tool": {
        "content": [
            {
                "text": "Based on the course material, machine learning is a field of AI focused on algorithms that learn from data."
            }
        ],
        "stop_reason": "end_turn",
    },
}


def create_mock_course(course_data: dict) -> Course:
    """Create a Course object from mock data"""
    lessons = [
        Lesson(
            lesson_number=lesson["lesson_number"],
            title=lesson["title"],
            lesson_link=lesson["lesson_link"],
        )
        for lesson in course_data["lessons"]
    ]

    return Course(
        title=course_data["title"],
        instructor=course_data["instructor"],
        course_link=course_data["course_link"],
        lessons=lessons,
    )


def create_mock_chunks(chunks_data: list[dict]) -> list[CourseChunk]:
    """Create CourseChunk objects from mock data"""
    return [
        CourseChunk(
            course_title=chunk["course_title"],
            lesson_number=chunk["lesson_number"],
            chunk_index=chunk["chunk_index"],
            content=chunk["content"],
        )
        for chunk in chunks_data
    ]


def get_expected_search_results(query: str) -> dict[str, Any]:
    """Get expected search results for a test query"""
    for test_query in TEST_QUERIES:
        if test_query["query"] == query:
            return test_query
    return {"should_find_results": False}


# Course document sample for document processor testing
SAMPLE_COURSE_DOCUMENT = """Course Title: Introduction to Testing
Course Link: https://example.com/testing-course
Course Instructor: Test Expert

Lesson 1: Testing Fundamentals  
Lesson Link: https://example.com/testing/lesson1
Testing is a critical part of software development. It helps ensure that code works as expected and prevents bugs from reaching users. There are many different types of testing, including unit tests, integration tests, and end-to-end tests.

Lesson 2: Unit Testing
Lesson Link: https://example.com/testing/lesson2  
Unit testing focuses on testing individual components or functions in isolation. This allows developers to verify that each piece of code works correctly on its own, making it easier to identify and fix bugs.

Lesson 3: Integration Testing
Lesson Link: https://example.com/testing/lesson3
Integration testing verifies that different components work together correctly. This is important because even if individual units work properly, they may not function correctly when combined.
"""

# Error scenarios for testing error handling
ERROR_SCENARIOS = [
    {
        "name": "vector_store_connection_error",
        "error_type": "ConnectionError",
        "error_message": "Unable to connect to vector database",
    },
    {
        "name": "api_key_error",
        "error_type": "AuthenticationError",
        "error_message": "Invalid API key",
    },
    {
        "name": "model_error",
        "error_type": "ModelError",
        "error_message": "Model is currently unavailable",
    },
    {
        "name": "empty_query_error",
        "error_type": "ValidationError",
        "error_message": "Query cannot be empty",
    },
]
