"""End-to-end tests for RAGSystem to diagnose complete pipeline issues."""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
from mock_data import (
    SAMPLE_COURSE_DOCUMENT,
)
from rag_system import RAGSystem
from test_utils import (
    cleanup_test_files,
    create_test_course_document,
    patch_config_for_testing,
)


class TestRAGSystemInitialization:
    """Test RAGSystem initialization and component setup"""

    def test_rag_system_initialization(self, mock_config):
        """Test that RAGSystem initializes all components correctly"""
        with patch_config_for_testing():
            rag_system = RAGSystem(mock_config)

            assert rag_system.document_processor is not None
            assert rag_system.vector_store is not None
            assert rag_system.ai_generator is not None
            assert rag_system.session_manager is not None
            assert rag_system.tool_manager is not None
            assert rag_system.search_tool is not None
            assert rag_system.outline_tool is not None

    def test_tools_registered(self, mock_config):
        """Test that tools are properly registered"""
        with patch_config_for_testing():
            rag_system = RAGSystem(mock_config)

            # Check that tools are registered
            tool_definitions = rag_system.tool_manager.get_tool_definitions()
            assert len(tool_definitions) == 2

            tool_names = [tool["name"] for tool in tool_definitions]
            assert "search_course_content" in tool_names
            assert "get_course_outline" in tool_names


class TestRAGSystemQuery:
    """Test the main query method of RAGSystem"""

    def test_query_without_session(self, mock_config):
        """Test query method without session ID"""
        with patch_config_for_testing():
            with patch("anthropic.Anthropic") as mock_anthropic:
                # Mock simple response without tool use
                mock_response = Mock()
                mock_response.content = [Mock()]
                mock_response.content[0].text = "This is a general knowledge answer."
                mock_response.stop_reason = "end_turn"
                mock_anthropic.return_value.messages.create.return_value = mock_response

                rag_system = RAGSystem(mock_config)
                answer, sources = rag_system.query("What is 2+2?")

                assert answer == "This is a general knowledge answer."
                assert isinstance(sources, list)

    def test_query_with_session(self, mock_config):
        """Test query method with session ID"""
        with patch_config_for_testing():
            with patch("anthropic.Anthropic") as mock_anthropic:
                mock_response = Mock()
                mock_response.content = [Mock()]
                mock_response.content[0].text = "Response with session context."
                mock_response.stop_reason = "end_turn"
                mock_anthropic.return_value.messages.create.return_value = mock_response

                rag_system = RAGSystem(mock_config)

                # Create session
                session_id = rag_system.session_manager.create_session()

                answer, sources = rag_system.query("Test question", session_id)

                assert answer == "Response with session context."
                assert isinstance(sources, list)

                # Verify session was updated
                history = rag_system.session_manager.get_conversation_history(
                    session_id
                )
                assert "Test question" in history

    def test_query_with_tool_use(self, mock_config):
        """Test query that triggers tool usage"""
        with patch_config_for_testing():
            with patch("anthropic.Anthropic") as mock_anthropic:
                # First response: tool use
                tool_response = Mock()
                tool_response.stop_reason = "tool_use"
                tool_response.content = [Mock()]
                tool_response.content[0].type = "tool_use"
                tool_response.content[0].name = "search_course_content"
                tool_response.content[0].input = {"query": "machine learning"}
                tool_response.content[0].id = "tool_123"

                # Second response: final answer
                final_response = Mock()
                final_response.content = [Mock()]
                final_response.content[0].text = (
                    "Based on the course materials, machine learning is..."
                )
                final_response.stop_reason = "end_turn"

                mock_anthropic.return_value.messages.create.side_effect = [
                    tool_response,
                    final_response,
                ]

                # Mock the vector store to return results
                with patch.object(RAGSystem, "__init__", lambda x, config: None):
                    rag_system = RAGSystem(None)

                    # Manually set up components with mocks
                    rag_system.config = mock_config
                    rag_system.session_manager = Mock()
                    rag_system.session_manager.get_conversation_history.return_value = (
                        None
                    )
                    rag_system.session_manager.add_exchange.return_value = None

                    # Set up mock vector store
                    mock_vector_store = Mock()
                    mock_search_results = Mock()
                    mock_search_results.error = None
                    mock_search_results.documents = ["ML content here"]
                    mock_search_results.metadata = [
                        {"course_title": "ML Course", "lesson_number": 1}
                    ]
                    mock_search_results.is_empty.return_value = False
                    mock_vector_store.search.return_value = mock_search_results
                    mock_vector_store.get_lesson_link.return_value = (
                        "https://example.com/lesson1"
                    )

                    # Set up tools
                    from search_tools import CourseSearchTool, ToolManager

                    rag_system.search_tool = CourseSearchTool(mock_vector_store)
                    rag_system.tool_manager = ToolManager()
                    rag_system.tool_manager.register_tool(rag_system.search_tool)

                    # Set up AI generator
                    from ai_generator import AIGenerator

                    rag_system.ai_generator = AIGenerator("test-key", "test-model")

                    answer, sources = rag_system.query("What is machine learning?")

                    assert (
                        answer
                        == "Based on the course materials, machine learning is..."
                    )
                    assert isinstance(sources, list)

                    # Verify the search tool was used
                    mock_vector_store.search.assert_called_once()

    def test_query_error_handling(self, mock_config):
        """Test query method error handling"""
        with patch_config_for_testing():
            with patch("anthropic.Anthropic") as mock_anthropic:
                mock_anthropic.return_value.messages.create.side_effect = Exception(
                    "API Error"
                )

                rag_system = RAGSystem(mock_config)

                # The error should propagate up
                with pytest.raises(Exception) as exc_info:
                    rag_system.query("Test question")

                assert "API Error" in str(exc_info.value)

    def test_sources_handling(self, mock_config):
        """Test that sources are properly handled and reset"""
        with patch_config_for_testing():
            with patch("anthropic.Anthropic") as mock_anthropic:
                # Mock tool use that would generate sources
                tool_response = Mock()
                tool_response.stop_reason = "tool_use"
                tool_response.content = [Mock()]
                tool_response.content[0].type = "tool_use"
                tool_response.content[0].name = "search_course_content"
                tool_response.content[0].input = {"query": "test"}
                tool_response.content[0].id = "tool_123"

                final_response = Mock()
                final_response.content = [Mock()]
                final_response.content[0].text = "Final answer"
                final_response.stop_reason = "end_turn"

                mock_anthropic.return_value.messages.create.side_effect = [
                    tool_response,
                    final_response,
                ]

                # Set up RAG system with mock components
                with patch.object(RAGSystem, "__init__", lambda x, config: None):
                    rag_system = RAGSystem(None)
                    rag_system.config = mock_config
                    rag_system.session_manager = Mock()
                    rag_system.session_manager.get_conversation_history.return_value = (
                        None
                    )
                    rag_system.session_manager.add_exchange.return_value = None

                    # Mock tool manager with sources
                    rag_system.tool_manager = Mock()
                    rag_system.tool_manager.get_tool_definitions.return_value = [
                        {"name": "search_course_content"}
                    ]
                    rag_system.tool_manager.execute_tool.return_value = "Search results"
                    rag_system.tool_manager.get_last_sources.return_value = [
                        {"text": "Course 1", "url": "https://example.com"}
                    ]
                    rag_system.tool_manager.reset_sources.return_value = None

                    from ai_generator import AIGenerator

                    rag_system.ai_generator = AIGenerator("test-key", "test-model")

                    answer, sources = rag_system.query("Test question")

                    assert answer == "Final answer"
                    assert len(sources) == 1
                    assert sources[0]["text"] == "Course 1"

                    # Verify sources were reset after retrieval
                    rag_system.tool_manager.reset_sources.assert_called_once()


class TestRAGSystemDocumentProcessing:
    """Test document processing functionality"""

    def test_add_course_document(self, mock_config):
        """Test adding a single course document"""
        with patch_config_for_testing():
            # Create a temporary course document
            temp_file = create_test_course_document(
                "test_course.txt", SAMPLE_COURSE_DOCUMENT
            )

            try:
                rag_system = RAGSystem(mock_config)

                course, chunk_count = rag_system.add_course_document(temp_file)

                assert course is not None
                assert course.title == "Introduction to Testing"
                assert course.instructor == "Test Expert"
                assert chunk_count > 0

            finally:
                cleanup_test_files(temp_file, os.path.dirname(temp_file))

    def test_add_course_folder(self, mock_config):
        """Test adding multiple course documents from folder"""
        with patch_config_for_testing():
            # Create temporary folder with course documents
            temp_dir = tempfile.mkdtemp()

            try:
                # Create multiple test files
                file1 = os.path.join(temp_dir, "course1.txt")
                file2 = os.path.join(temp_dir, "course2.txt")

                with open(file1, "w") as f:
                    f.write(SAMPLE_COURSE_DOCUMENT)

                with open(file2, "w") as f:
                    f.write(
                        SAMPLE_COURSE_DOCUMENT.replace("Testing", "Advanced Testing")
                    )

                rag_system = RAGSystem(mock_config)

                courses, chunks = rag_system.add_course_folder(temp_dir)

                assert courses >= 1  # Should process at least one valid course
                assert chunks > 0

            finally:
                cleanup_test_files(temp_dir)

    def test_add_course_document_error_handling(self, mock_config):
        """Test error handling when adding invalid document"""
        with patch_config_for_testing():
            rag_system = RAGSystem(mock_config)

            # Try to add non-existent file
            course, chunk_count = rag_system.add_course_document(
                "/nonexistent/file.txt"
            )

            assert course is None
            assert chunk_count == 0


class TestRAGSystemAnalytics:
    """Test analytics and statistics functionality"""

    def test_get_course_analytics(self, mock_config):
        """Test getting course analytics"""
        with patch_config_for_testing():
            rag_system = RAGSystem(mock_config)

            # Mock the vector store methods
            rag_system.vector_store.get_course_count = Mock(return_value=3)
            rag_system.vector_store.get_existing_course_titles = Mock(
                return_value=["Course 1", "Course 2", "Course 3"]
            )

            analytics = rag_system.get_course_analytics()

            assert analytics["total_courses"] == 3
            assert len(analytics["course_titles"]) == 3
            assert "Course 1" in analytics["course_titles"]


class TestRAGSystemRealData:
    """Test RAGSystem with real data to identify actual issues"""

    def test_with_real_vector_store_empty(self, mock_config):
        """Test RAG system behavior with empty vector store"""
        with patch_config_for_testing():
            with patch("anthropic.Anthropic") as mock_anthropic:
                # Tool use response
                tool_response = Mock()
                tool_response.stop_reason = "tool_use"
                tool_response.content = [Mock()]
                tool_response.content[0].type = "tool_use"
                tool_response.content[0].name = "search_course_content"
                tool_response.content[0].input = {"query": "machine learning"}
                tool_response.content[0].id = "tool_123"

                # Final response
                final_response = Mock()
                final_response.content = [Mock()]
                final_response.content[0].text = "I couldn't find relevant information."
                final_response.stop_reason = "end_turn"

                mock_anthropic.return_value.messages.create.side_effect = [
                    tool_response,
                    final_response,
                ]

                # Use real vector store (which will be empty)
                rag_system = RAGSystem(mock_config)

                # This should reveal what happens with empty vector store
                answer, sources = rag_system.query("What is machine learning?")

                # Capture what actually happens
                print(f"Answer with empty vector store: {answer}")
                print(f"Sources: {sources}")

                assert answer is not None
                assert isinstance(sources, list)

    def test_with_actual_course_data(self, mock_config):
        """Test with actual course data loaded"""
        with patch_config_for_testing():
            with patch("anthropic.Anthropic") as mock_anthropic:
                # Mock responses
                tool_response = Mock()
                tool_response.stop_reason = "tool_use"
                tool_response.content = [Mock()]
                tool_response.content[0].type = "tool_use"
                tool_response.content[0].name = "search_course_content"
                tool_response.content[0].input = {"query": "testing fundamentals"}
                tool_response.content[0].id = "tool_456"

                final_response = Mock()
                final_response.content = [Mock()]
                final_response.content[0].text = (
                    "Testing is important for software quality."
                )
                final_response.stop_reason = "end_turn"

                mock_anthropic.return_value.messages.create.side_effect = [
                    tool_response,
                    final_response,
                ]

                # Create RAG system and add real data
                rag_system = RAGSystem(mock_config)

                # Create and add test course document
                temp_file = create_test_course_document(
                    "test_course.txt", SAMPLE_COURSE_DOCUMENT
                )

                try:
                    course, chunks = rag_system.add_course_document(temp_file)
                    print(
                        f"Added course: {course.title if course else 'None'}, Chunks: {chunks}"
                    )

                    # Now test query
                    answer, sources = rag_system.query("What are testing fundamentals?")

                    print(f"Answer with data: {answer}")
                    print(f"Sources with data: {sources}")

                    assert answer == "Testing is important for software quality."

                finally:
                    cleanup_test_files(temp_file, os.path.dirname(temp_file))


class TestRAGSystemErrorDiagnosis:
    """Specific tests to diagnose the 'query failed' issue"""

    def test_diagnose_query_failed_issue(self, mock_config):
        """Comprehensive test to diagnose why queries are failing"""
        with patch_config_for_testing():
            print("\\n=== DIAGNOSING QUERY FAILED ISSUE ===")

            # Test each component individually
            rag_system = RAGSystem(mock_config)

            # 1. Test vector store directly
            print("1. Testing vector store...")
            try:
                # Test if vector store can perform search

                test_results = rag_system.vector_store.search("test query")
                print(f"   Vector store search result: {test_results}")
                print(f"   Error: {test_results.error}")
                print(
                    f"   Documents: {len(test_results.documents) if test_results.documents else 0}"
                )
            except Exception as e:
                print(f"   Vector store error: {e}")

            # 2. Test search tool directly
            print("2. Testing search tool...")
            try:
                search_result = rag_system.search_tool.execute("test query")
                print(
                    f"   Search tool result: {search_result[:100]}..."
                    if len(search_result) > 100
                    else search_result
                )
            except Exception as e:
                print(f"   Search tool error: {e}")

            # 3. Test tool manager
            print("3. Testing tool manager...")
            try:
                tool_defs = rag_system.tool_manager.get_tool_definitions()
                print(f"   Tool definitions count: {len(tool_defs)}")
                for tool_def in tool_defs:
                    print(f"   Tool: {tool_def['name']}")

                # Test tool execution
                manager_result = rag_system.tool_manager.execute_tool(
                    "search_course_content", query="test query"
                )
                print(
                    f"   Tool manager execution result: {manager_result[:100]}..."
                    if len(manager_result) > 100
                    else manager_result
                )
            except Exception as e:
                print(f"   Tool manager error: {e}")

            # 4. Test AI generator (mocked)
            print("4. Testing AI generator...")
            try:
                with patch("anthropic.Anthropic") as mock_anthropic:
                    mock_response = Mock()
                    mock_response.content = [Mock()]
                    mock_response.content[0].text = "Test AI response"
                    mock_response.stop_reason = "end_turn"
                    mock_anthropic.return_value.messages.create.return_value = (
                        mock_response
                    )

                    ai_result = rag_system.ai_generator.generate_response(
                        "test query",
                        tools=rag_system.tool_manager.get_tool_definitions(),
                        tool_manager=rag_system.tool_manager,
                    )
                    print(f"   AI generator result: {ai_result}")
            except Exception as e:
                print(f"   AI generator error: {e}")

            # 5. Test full pipeline (mocked)
            print("5. Testing full pipeline...")
            try:
                with patch("anthropic.Anthropic") as mock_anthropic:
                    mock_response = Mock()
                    mock_response.content = [Mock()]
                    mock_response.content[0].text = "Full pipeline test response"
                    mock_response.stop_reason = "end_turn"
                    mock_anthropic.return_value.messages.create.return_value = (
                        mock_response
                    )

                    answer, sources = rag_system.query("What is testing?")
                    print(f"   Full pipeline answer: {answer}")
                    print(f"   Full pipeline sources: {sources}")
            except Exception as e:
                print(f"   Full pipeline error: {e}")

            print("=== END DIAGNOSIS ===\\n")
