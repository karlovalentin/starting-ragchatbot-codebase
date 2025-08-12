"""Unit tests for CourseSearchTool to diagnose search functionality issues."""


from search_tools import CourseSearchTool, ToolManager
from test_utils import (
    assert_tool_definition_valid,
    create_mock_search_results,
)
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test the CourseSearchTool execute method and functionality"""

    def test_tool_definition_valid(self, course_search_tool):
        """Test that tool definition is valid for Anthropic API"""
        tool_def = course_search_tool.get_tool_definition()
        assert_tool_definition_valid(tool_def)

        assert tool_def["name"] == "search_course_content"
        assert "query" in tool_def["input_schema"]["properties"]
        assert "course_name" in tool_def["input_schema"]["properties"]
        assert "lesson_number" in tool_def["input_schema"]["properties"]
        assert tool_def["input_schema"]["required"] == ["query"]

    def test_execute_with_successful_search(self, mock_vector_store):
        """Test execute method with successful search results"""
        # Setup mock vector store to return results
        mock_results = create_mock_search_results(
            documents=["This is content about machine learning fundamentals."],
            metadata=[{"course_title": "ML Course", "lesson_number": 1}],
            distances=[0.1],
        )
        mock_vector_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("machine learning")

        # Verify results
        assert isinstance(result, str)
        assert len(result) > 0
        assert "ML Course" in result
        assert "Lesson 1" in result

        # Verify vector store was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="machine learning", course_name=None, lesson_number=None
        )

    def test_execute_with_course_filter(self, mock_vector_store):
        """Test execute method with course name filter"""
        mock_results = create_mock_search_results(
            documents=["MCP protocol information"],
            metadata=[{"course_title": "Building with MCP", "lesson_number": 2}],
            distances=[0.05],
        )
        mock_vector_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("protocol", course_name="MCP")

        assert "Building with MCP" in result
        assert "Lesson 2" in result

        # Verify search was called with course filter
        mock_vector_store.search.assert_called_once_with(
            query="protocol", course_name="MCP", lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store):
        """Test execute method with lesson number filter"""
        mock_results = create_mock_search_results(
            documents=["Linear regression concepts"],
            metadata=[{"course_title": "ML Course", "lesson_number": 3}],
            distances=[0.2],
        )
        mock_vector_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("regression", lesson_number=3)

        assert "ML Course" in result
        assert "Lesson 3" in result

        mock_vector_store.search.assert_called_once_with(
            query="regression", course_name=None, lesson_number=3
        )

    def test_execute_with_both_filters(self, mock_vector_store):
        """Test execute method with both course and lesson filters"""
        mock_results = create_mock_search_results(
            documents=["Advanced MCP concepts"],
            metadata=[{"course_title": "Building with MCP", "lesson_number": 2}],
            distances=[0.15],
        )
        mock_vector_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("advanced", course_name="MCP", lesson_number=2)

        assert "Building with MCP" in result
        assert "Lesson 2" in result

        mock_vector_store.search.assert_called_once_with(
            query="advanced", course_name="MCP", lesson_number=2
        )

    def test_execute_with_empty_results(self, mock_empty_vector_store):
        """Test execute method when search returns no results"""
        tool = CourseSearchTool(mock_empty_vector_store)
        result = tool.execute("nonexistent topic")

        assert "No relevant content found" in result
        assert isinstance(result, str)

        mock_empty_vector_store.search.assert_called_once()

    def test_execute_with_empty_results_and_filters(self, mock_empty_vector_store):
        """Test execute method with empty results and filters"""
        tool = CourseSearchTool(mock_empty_vector_store)
        result = tool.execute(
            "topic", course_name="Nonexistent Course", lesson_number=99
        )

        assert "No relevant content found" in result
        assert "Nonexistent Course" in result
        assert "lesson 99" in result

    def test_execute_with_vector_store_error(self, mock_error_vector_store):
        """Test execute method when vector store returns error"""
        tool = CourseSearchTool(mock_error_vector_store)
        result = tool.execute("any query")

        assert "Search error occurred" in result
        assert isinstance(result, str)

        mock_error_vector_store.search.assert_called_once()

    def test_execute_tracks_sources(self, mock_vector_store):
        """Test that execute method tracks sources for UI"""
        mock_results = create_mock_search_results(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/lesson1",
            "https://example.com/lesson2",
        ]

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")

        # Check that sources were tracked
        assert len(tool.last_sources) == 2

        source1 = tool.last_sources[0]
        assert source1["text"] == "Course A - Lesson 1"
        assert source1["url"] == "https://example.com/lesson1"

        source2 = tool.last_sources[1]
        assert source2["text"] == "Course B - Lesson 2"
        assert source2["url"] == "https://example.com/lesson2"

    def test_format_results_structure(self, mock_vector_store):
        """Test internal _format_results method structure"""
        mock_results = create_mock_search_results(
            documents=["Sample content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
        )

        tool = CourseSearchTool(mock_vector_store)
        formatted = tool._format_results(mock_results)

        # Check format structure
        assert "[Test Course - Lesson 1]" in formatted
        assert "Sample content" in formatted
        assert isinstance(formatted, str)

    def test_format_results_without_lesson_number(self, mock_vector_store):
        """Test formatting when lesson number is not available"""
        mock_results = create_mock_search_results(
            documents=["Course overview"],
            metadata=[{"course_title": "Test Course"}],  # No lesson_number
            distances=[0.1],
        )

        tool = CourseSearchTool(mock_vector_store)
        formatted = tool._format_results(mock_results)

        assert "[Test Course]" in formatted  # Should not include lesson info
        assert "Course overview" in formatted

    def test_multiple_documents_formatting(self, mock_vector_store):
        """Test formatting of multiple search results"""
        mock_results = create_mock_search_results(
            documents=["First result", "Second result"],
            metadata=[
                {"course_title": "Course 1", "lesson_number": 1},
                {"course_title": "Course 2", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
        )

        tool = CourseSearchTool(mock_vector_store)
        formatted = tool._format_results(mock_results)

        # Check both results are included
        assert "[Course 1 - Lesson 1]" in formatted
        assert "First result" in formatted
        assert "[Course 2 - Lesson 2]" in formatted
        assert "Second result" in formatted

        # Check they're separated properly
        assert (
            formatted.count("\n\n") >= 1
        )  # Should have double newlines between results


class TestToolManager:
    """Test ToolManager functionality with CourseSearchTool"""

    def test_tool_registration(self, course_search_tool):
        """Test tool registration in ToolManager"""
        manager = ToolManager()
        manager.register_tool(course_search_tool)

        assert "search_course_content" in manager.tools

        # Test tool definitions
        definitions = manager.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_tool_execution_through_manager(self, mock_vector_store):
        """Test executing tool through ToolManager"""
        mock_results = create_mock_search_results(
            documents=["Test content"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.1],
        )
        mock_vector_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_vector_store)
        manager = ToolManager()
        manager.register_tool(tool)

        result = manager.execute_tool(
            "search_course_content", query="test query", course_name="Test Course"
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_tool_execution_unknown_tool(self, tool_manager):
        """Test executing unknown tool returns error"""
        result = tool_manager.execute_tool("unknown_tool", query="test")

        assert "Tool 'unknown_tool' not found" in result

    def test_get_last_sources(self, mock_vector_store):
        """Test retrieving sources from ToolManager"""
        mock_results = create_mock_search_results(
            documents=["Content"],
            metadata=[{"course_title": "Course", "lesson_number": 1}],
            distances=[0.1],
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        tool = CourseSearchTool(mock_vector_store)
        manager = ToolManager()
        manager.register_tool(tool)

        # Execute search to populate sources
        manager.execute_tool("search_course_content", query="test")

        # Get sources
        sources = manager.get_last_sources()
        assert len(sources) == 1
        assert sources[0]["text"] == "Course - Lesson 1"
        assert sources[0]["url"] == "https://example.com/lesson1"

    def test_reset_sources(self, mock_vector_store):
        """Test resetting sources in ToolManager"""
        mock_results = create_mock_search_results(
            documents=["Content"],
            metadata=[{"course_title": "Course", "lesson_number": 1}],
            distances=[0.1],
        )
        mock_vector_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_vector_store)
        manager = ToolManager()
        manager.register_tool(tool)

        # Execute search and verify sources exist
        manager.execute_tool("search_course_content", query="test")
        assert len(manager.get_last_sources()) > 0

        # Reset and verify sources are cleared
        manager.reset_sources()
        assert len(manager.get_last_sources()) == 0


class TestErrorScenarios:
    """Test various error scenarios that might cause 'query failed'"""

    def test_invalid_query_parameter(self, mock_vector_store):
        """Test with invalid query parameter"""
        tool = CourseSearchTool(mock_vector_store)

        # Test empty query
        result = tool.execute("")
        assert isinstance(result, str)
        # Should still call vector store even with empty query
        mock_vector_store.search.assert_called_once()

    def test_invalid_lesson_number(self, mock_vector_store):
        """Test with invalid lesson number"""
        mock_results = create_mock_search_results(
            documents=[], metadata=[], distances=[]
        )
        mock_vector_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test", lesson_number=-1)

        assert "No relevant content found" in result
        mock_vector_store.search.assert_called_once_with(
            query="test", course_name=None, lesson_number=-1
        )

    def test_vector_store_exception(self, mock_vector_store):
        """Test when vector store raises exception"""
        mock_vector_store.search.side_effect = Exception("Database connection failed")

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")

        # The tool should handle the exception and return error message
        assert isinstance(result, str)
        # This will help us see what the actual error handling does
        print(f"Error result: {result}")

    def test_malformed_metadata(self, mock_vector_store):
        """Test with malformed metadata from vector store"""
        # Create results with missing required metadata fields
        mock_results = SearchResults(
            documents=["Content"],
            metadata=[{}],  # Empty metadata
            distances=[0.1],
            error=None,
        )
        mock_vector_store.search.return_value = mock_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test")

        # Should handle gracefully and still return formatted result
        assert isinstance(result, str)
        assert len(result) > 0
