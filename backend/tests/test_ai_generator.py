"""Integration tests for AIGenerator to diagnose tool calling issues."""

from unittest.mock import Mock, patch

import anthropic
import pytest
from ai_generator import AIGenerator
from search_tools import CourseSearchTool, ToolManager
from test_utils import (
    mock_anthropic_response,
)


class TestAIGenerator:
    """Test AIGenerator functionality and tool calling mechanism"""

    def test_initialization(self):
        """Test AIGenerator initialization"""
        generator = AIGenerator("test-api-key", "test-model")

        assert generator.model == "test-model"
        assert generator.base_params["model"] == "test-model"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    def test_generate_response_without_tools(self):
        """Test generating response without tools"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_response = mock_anthropic_response("Simple response without tools")
            mock_anthropic.return_value.messages.create.return_value = mock_response

            generator = AIGenerator("test-key", "test-model")
            result = generator.generate_response("What is AI?")

            assert result == "Simple response without tools"

            # Verify API was called correctly
            mock_anthropic.return_value.messages.create.assert_called_once()
            call_args = mock_anthropic.return_value.messages.create.call_args

            assert call_args[1]["messages"][0]["content"] == "What is AI?"
            assert "tools" not in call_args[1]  # No tools should be passed

    def test_generate_response_with_tools_no_tool_use(self):
        """Test generating response with tools available but not used"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_response = mock_anthropic_response("Response without using tools")
            mock_anthropic.return_value.messages.create.return_value = mock_response

            # Create mock tools
            mock_tools = [{"name": "test_tool", "description": "Test tool"}]
            mock_tool_manager = Mock()

            generator = AIGenerator("test-key", "test-model")
            result = generator.generate_response(
                "General question", tools=mock_tools, tool_manager=mock_tool_manager
            )

            assert result == "Response without using tools"

            # Verify tools were passed to API
            call_args = mock_anthropic.return_value.messages.create.call_args
            assert call_args[1]["tools"] == mock_tools
            assert call_args[1]["tool_choice"] == {"type": "auto"}

            # Tool manager should not have been called
            mock_tool_manager.execute_tool.assert_not_called()

    def test_generate_response_with_tool_use(self):
        """Test generating response that uses tools"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            # First response: tool use
            tool_use_response = Mock()
            tool_use_response.stop_reason = "tool_use"
            tool_use_response.content = [Mock()]
            tool_use_response.content[0].type = "tool_use"
            tool_use_response.content[0].name = "search_course_content"
            tool_use_response.content[0].input = {"query": "machine learning"}
            tool_use_response.content[0].id = "tool_123"

            # Second response: final answer
            final_response = mock_anthropic_response(
                "Based on the search results, machine learning is..."
            )

            mock_anthropic.return_value.messages.create.side_effect = [
                tool_use_response,
                final_response,
            ]

            # Create mock tool manager
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = (
                "Search results about machine learning"
            )

            mock_tools = [
                {
                    "name": "search_course_content",
                    "description": "Search course content",
                }
            ]

            generator = AIGenerator("test-key", "test-model")
            result = generator.generate_response(
                "What is machine learning?",
                tools=mock_tools,
                tool_manager=mock_tool_manager,
            )

            assert result == "Based on the search results, machine learning is..."

            # Verify tool was executed
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content", query="machine learning"
            )

            # Verify two API calls were made (initial + follow-up)
            assert mock_anthropic.return_value.messages.create.call_count == 2

    def test_tool_execution_flow(self):
        """Test the complete tool execution flow"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            # Mock tool use response
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_response.content = [Mock()]
            tool_response.content[0].type = "tool_use"
            tool_response.content[0].name = "search_course_content"
            tool_response.content[0].input = {
                "query": "test query",
                "course_name": "Test Course",
            }
            tool_response.content[0].id = "tool_456"

            # Mock final response
            final_response = mock_anthropic_response(
                "Final response after tool execution"
            )

            mock_anthropic.return_value.messages.create.side_effect = [
                tool_response,
                final_response,
            ]

            # Create real tool manager with mock tool
            mock_vector_store = Mock()
            mock_vector_store.search.return_value = Mock()
            mock_vector_store.search.return_value.error = None
            mock_vector_store.search.return_value.documents = ["Test content"]
            mock_vector_store.search.return_value.metadata = [
                {"course_title": "Test", "lesson_number": 1}
            ]
            mock_vector_store.search.return_value.is_empty.return_value = False

            search_tool = CourseSearchTool(mock_vector_store)
            tool_manager = ToolManager()
            tool_manager.register_tool(search_tool)

            generator = AIGenerator("test-key", "test-model")
            result = generator.generate_response(
                "Search for test content",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager,
            )

            assert result == "Final response after tool execution"

            # Verify the vector store was called through the tool
            mock_vector_store.search.assert_called_once_with(
                query="test query", course_name="Test Course", lesson_number=None
            )

    def test_conversation_history_handling(self):
        """Test that conversation history is included in system prompt"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_response = mock_anthropic_response("Response with history")
            mock_anthropic.return_value.messages.create.return_value = mock_response

            generator = AIGenerator("test-key", "test-model")

            history = "User: Previous question\\nAssistant: Previous answer"
            result = generator.generate_response(
                "Current question", conversation_history=history
            )

            assert result == "Response with history"

            # Verify history was included in system prompt
            call_args = mock_anthropic.return_value.messages.create.call_args
            system_content = call_args[1]["system"]
            assert "Previous conversation:" in system_content
            assert history in system_content

    def test_api_error_handling(self):
        """Test handling of Anthropic API errors"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_anthropic.return_value.messages.create.side_effect = Exception(
                "API Error"
            )

            generator = AIGenerator("test-key", "test-model")

            with pytest.raises(Exception) as exc_info:
                generator.generate_response("Test query")

            assert "API Error" in str(exc_info.value)

    def test_tool_execution_error_handling(self):
        """Test handling of tool execution errors"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            # Tool use response
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_response.content = [Mock()]
            tool_response.content[0].type = "tool_use"
            tool_response.content[0].name = "search_course_content"
            tool_response.content[0].input = {"query": "test"}
            tool_response.content[0].id = "tool_789"

            # Final response
            final_response = mock_anthropic_response("Response after tool error")

            mock_anthropic.return_value.messages.create.side_effect = [
                tool_response,
                final_response,
            ]

            # Mock tool manager that returns error
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = (
                "Tool execution failed with error"
            )

            generator = AIGenerator("test-key", "test-model")
            result = generator.generate_response(
                "Test query",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager,
            )

            assert result == "Response after tool error"

            # Verify error was passed back to the API
            second_call_args = (
                mock_anthropic.return_value.messages.create.call_args_list[1]
            )
            messages = second_call_args[1]["messages"]

            # Should have user message with tool results
            tool_result_message = messages[-1]
            assert tool_result_message["role"] == "user"
            assert "Tool execution failed with error" in str(
                tool_result_message["content"]
            )


class TestAIGeneratorIntegration:
    """Integration tests with real tool components"""

    def test_integration_with_working_search_tool(self):
        """Test AIGenerator integration with working search tool"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            # Mock tool use decision
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_response.content = [Mock()]
            tool_response.content[0].type = "tool_use"
            tool_response.content[0].name = "search_course_content"
            tool_response.content[0].input = {"query": "machine learning basics"}
            tool_response.content[0].id = "tool_test"

            # Mock final response
            final_response = mock_anthropic_response(
                "Machine learning is a field of AI..."
            )

            mock_anthropic.return_value.messages.create.side_effect = [
                tool_response,
                final_response,
            ]

            # Set up working search tool
            mock_vector_store = Mock()
            mock_search_results = Mock()
            mock_search_results.error = None
            mock_search_results.documents = [
                "ML is a subset of AI that focuses on learning from data."
            ]
            mock_search_results.metadata = [
                {"course_title": "ML Course", "lesson_number": 1}
            ]
            mock_search_results.is_empty.return_value = False

            mock_vector_store.search.return_value = mock_search_results
            mock_vector_store.get_lesson_link.return_value = (
                "https://example.com/lesson1"
            )

            # Create tool and manager
            search_tool = CourseSearchTool(mock_vector_store)
            tool_manager = ToolManager()
            tool_manager.register_tool(search_tool)

            # Test the integration
            generator = AIGenerator("test-key", "test-model")
            result = generator.generate_response(
                "What is machine learning?",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager,
            )

            assert result == "Machine learning is a field of AI..."

            # Verify the search tool was called correctly
            mock_vector_store.search.assert_called_once_with(
                query="machine learning basics", course_name=None, lesson_number=None
            )

    def test_integration_with_failing_search_tool(self):
        """Test AIGenerator integration when search tool fails"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            # Mock tool use decision
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_response.content = [Mock()]
            tool_response.content[0].type = "tool_use"
            tool_response.content[0].name = "search_course_content"
            tool_response.content[0].input = {"query": "test"}
            tool_response.content[0].id = "tool_fail"

            # Mock final response
            final_response = mock_anthropic_response(
                "I apologize, but I encountered an error..."
            )

            mock_anthropic.return_value.messages.create.side_effect = [
                tool_response,
                final_response,
            ]

            # Set up failing search tool
            mock_vector_store = Mock()
            mock_search_results = Mock()
            mock_search_results.error = "Database connection failed"

            mock_vector_store.search.return_value = mock_search_results

            # Create tool and manager
            search_tool = CourseSearchTool(mock_vector_store)
            tool_manager = ToolManager()
            tool_manager.register_tool(search_tool)

            # Test the integration
            generator = AIGenerator("test-key", "test-model")
            result = generator.generate_response(
                "Test query",
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager,
            )

            assert result == "I apologize, but I encountered an error..."

            # Verify the error was propagated through the system
            second_call = mock_anthropic.return_value.messages.create.call_args_list[1]
            messages = second_call[1]["messages"]
            tool_result = messages[-1]["content"][0]["content"]
            assert "Database connection failed" in tool_result


class TestSystemPromptBehavior:
    """Test system prompt and tool decision making"""

    def test_system_prompt_includes_tool_guidance(self):
        """Test that system prompt includes proper tool usage guidance"""
        generator = AIGenerator("test-key", "test-model")

        # The system prompt should contain guidance about tool usage
        assert "search_course_content" in generator.SYSTEM_PROMPT
        assert "get_course_outline" in generator.SYSTEM_PROMPT
        assert "Tool Usage:" in generator.SYSTEM_PROMPT

    def test_tool_choice_auto_when_tools_provided(self):
        """Test that tool_choice is set to auto when tools are provided"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_response = mock_anthropic_response("Test response")
            mock_anthropic.return_value.messages.create.return_value = mock_response

            generator = AIGenerator("test-key", "test-model")

            mock_tools = [{"name": "test_tool"}]
            generator.generate_response("Test query", tools=mock_tools)

            call_args = mock_anthropic.return_value.messages.create.call_args
            assert call_args[1]["tool_choice"] == {"type": "auto"}

    def test_no_tool_choice_when_no_tools(self):
        """Test that tool_choice is not set when no tools provided"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_response = mock_anthropic_response("Test response")
            mock_anthropic.return_value.messages.create.return_value = mock_response

            generator = AIGenerator("test-key", "test-model")
            generator.generate_response("Test query")

            call_args = mock_anthropic.return_value.messages.create.call_args
            assert "tool_choice" not in call_args[1]


class TestSequentialToolCalling:
    """Test sequential tool calling functionality"""

    def test_sequential_two_tool_calls(self):
        """Test that Claude can make 2 sequential tool calls"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            # Round 1: Tool use response
            round1_response = Mock()
            round1_response.stop_reason = "tool_use"
            round1_response.content = [Mock()]
            round1_response.content[0].type = "tool_use"
            round1_response.content[0].name = "get_course_outline"
            round1_response.content[0].input = {
                "course_name": "Machine Learning Course"
            }
            round1_response.content[0].id = "tool_round1"

            # Round 2: Tool use response
            round2_response = Mock()
            round2_response.stop_reason = "tool_use"
            round2_response.content = [Mock()]
            round2_response.content[0].type = "tool_use"
            round2_response.content[0].name = "search_course_content"
            round2_response.content[0].input = {
                "query": "neural networks",
                "course_name": "Deep Learning",
            }
            round2_response.content[0].id = "tool_round2"

            # Final response after max rounds
            final_response = mock_anthropic_response(
                "Based on both searches, here's a comprehensive comparison..."
            )

            mock_anthropic.return_value.messages.create.side_effect = [
                round1_response,  # Initial call with tools
                round2_response,  # Round 2 call with tools
                final_response,  # Final call without tools (forced)
            ]

            # Create mock tool manager
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.side_effect = [
                "Course outline results for Machine Learning",
                "Search results about neural networks in Deep Learning course",
            ]

            mock_tools = [
                {"name": "get_course_outline", "description": "Get course outline"},
                {
                    "name": "search_course_content",
                    "description": "Search course content",
                },
            ]

            generator = AIGenerator("test-key", "test-model")
            result = generator.generate_response(
                "Compare lesson 4 of Machine Learning course with Deep Learning content on neural networks",
                tools=mock_tools,
                tool_manager=mock_tool_manager,
            )

            # Verify final response
            assert (
                result == "Based on both searches, here's a comprehensive comparison..."
            )

            # Verify both tools were executed
            assert mock_tool_manager.execute_tool.call_count == 2
            mock_tool_manager.execute_tool.assert_any_call(
                "get_course_outline", course_name="Machine Learning Course"
            )
            mock_tool_manager.execute_tool.assert_any_call(
                "search_course_content",
                query="neural networks",
                course_name="Deep Learning",
            )

            # Verify 3 API calls were made (initial + round 2 + final)
            assert mock_anthropic.return_value.messages.create.call_count == 3

    def test_sequential_terminates_after_one_round(self):
        """Test that system terminates when Claude doesn't use tools in round 2"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            # Round 1: Tool use response
            round1_response = Mock()
            round1_response.stop_reason = "tool_use"
            round1_response.content = [Mock()]
            round1_response.content[0].type = "tool_use"
            round1_response.content[0].name = "search_course_content"
            round1_response.content[0].input = {"query": "machine learning basics"}
            round1_response.content[0].id = "tool_round1"

            # Round 2: Text response (no tools)
            round2_response = mock_anthropic_response(
                "Based on the search results, machine learning is..."
            )

            mock_anthropic.return_value.messages.create.side_effect = [
                round1_response,  # Initial call with tools
                round2_response,  # Round 2 call - Claude chooses not to use tools
            ]

            # Create mock tool manager
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = (
                "Search results about machine learning basics"
            )

            mock_tools = [
                {
                    "name": "search_course_content",
                    "description": "Search course content",
                }
            ]

            generator = AIGenerator("test-key", "test-model")
            result = generator.generate_response(
                "What is machine learning?",
                tools=mock_tools,
                tool_manager=mock_tool_manager,
            )

            # Verify response
            assert result == "Based on the search results, machine learning is..."

            # Verify only one tool was executed
            assert mock_tool_manager.execute_tool.call_count == 1

            # Verify 2 API calls were made (initial + round 2 with no tools used)
            assert mock_anthropic.return_value.messages.create.call_count == 2

    def test_max_rounds_enforced(self):
        """Test that system enforces 2-round limit"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            # Round 1: Tool use response
            round1_response = Mock()
            round1_response.stop_reason = "tool_use"
            round1_response.content = [Mock()]
            round1_response.content[0].type = "tool_use"
            round1_response.content[0].name = "search_course_content"
            round1_response.content[0].input = {"query": "search term 1"}
            round1_response.content[0].id = "tool_round1"

            # Round 2: Tool use response
            round2_response = Mock()
            round2_response.stop_reason = "tool_use"
            round2_response.content = [Mock()]
            round2_response.content[0].type = "tool_use"
            round2_response.content[0].name = "search_course_content"
            round2_response.content[0].input = {"query": "search term 2"}
            round2_response.content[0].id = "tool_round2"

            # Final forced response
            final_response = mock_anthropic_response(
                "Final answer after 2 rounds of tools"
            )

            mock_anthropic.return_value.messages.create.side_effect = [
                round1_response,  # Initial call
                round2_response,  # Round 2 call
                final_response,  # Final forced call without tools
            ]

            # Create mock tool manager
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.side_effect = [
                "Results from search 1",
                "Results from search 2",
            ]

            mock_tools = [
                {
                    "name": "search_course_content",
                    "description": "Search course content",
                }
            ]

            generator = AIGenerator("test-key", "test-model")
            result = generator.generate_response(
                "Complex query requiring multiple searches",
                tools=mock_tools,
                tool_manager=mock_tool_manager,
            )

            # Verify final response
            assert result == "Final answer after 2 rounds of tools"

            # Verify both tools were executed (2 rounds max)
            assert mock_tool_manager.execute_tool.call_count == 2

            # Verify 3 API calls (initial + round 2 + final without tools)
            assert mock_anthropic.return_value.messages.create.call_count == 3

            # Verify final call had no tools
            final_call_args = (
                mock_anthropic.return_value.messages.create.call_args_list[2]
            )
            assert "tools" not in final_call_args[1]

    def test_tool_execution_error_terminates_sequence(self):
        """Test graceful handling of tool execution failures"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            # Round 1: Tool use response
            round1_response = Mock()
            round1_response.stop_reason = "tool_use"
            round1_response.content = [Mock()]
            round1_response.content[0].type = "tool_use"
            round1_response.content[0].name = "search_course_content"
            round1_response.content[0].input = {"query": "test"}
            round1_response.content[0].id = "tool_round1"

            mock_anthropic.return_value.messages.create.return_value = round1_response

            # Mock tool manager that raises exception
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.side_effect = Exception(
                "Database connection failed"
            )

            mock_tools = [
                {
                    "name": "search_course_content",
                    "description": "Search course content",
                }
            ]

            generator = AIGenerator("test-key", "test-model")
            result = generator.generate_response(
                "Test query", tools=mock_tools, tool_manager=mock_tool_manager
            )

            # Verify error is handled gracefully
            assert "I encountered an error while searching for information" in result
            assert "Database connection failed" in result

            # Verify tool was attempted
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content", query="test"
            )

            # Verify only 1 API call was made (no additional rounds after tool failure)
            assert mock_anthropic.return_value.messages.create.call_count == 1

    def test_complex_query_workflow(self):
        """Test the example use case: course comparison requiring outline then search"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            # Round 1: Get course outline
            round1_response = Mock()
            round1_response.stop_reason = "tool_use"
            round1_response.content = [Mock()]
            round1_response.content[0].type = "tool_use"
            round1_response.content[0].name = "get_course_outline"
            round1_response.content[0].input = {
                "course_name": "Introduction to Machine Learning"
            }
            round1_response.content[0].id = "tool_outline"

            # Round 2: Search based on outline results
            round2_response = Mock()
            round2_response.stop_reason = "tool_use"
            round2_response.content = [Mock()]
            round2_response.content[0].type = "tool_use"
            round2_response.content[0].name = "search_course_content"
            round2_response.content[0].input = {
                "query": "supervised learning algorithms"
            }
            round2_response.content[0].id = "tool_search"

            # Final response
            final_response = mock_anthropic_response(
                "Here are courses that discuss supervised learning algorithms, which is the topic of lesson 4..."
            )

            mock_anthropic.return_value.messages.create.side_effect = [
                round1_response,
                round2_response,
                final_response,
            ]

            # Mock tool manager with realistic responses
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.side_effect = [
                "Course: Introduction to Machine Learning\nLesson 4: Supervised Learning Algorithms",
                "Found 3 courses discussing supervised learning algorithms: Advanced ML, Data Science Fundamentals, AI Bootcamp",
            ]

            mock_tools = [
                {"name": "get_course_outline", "description": "Get course outline"},
                {
                    "name": "search_course_content",
                    "description": "Search course content",
                },
            ]

            generator = AIGenerator("test-key", "test-model")
            result = generator.generate_response(
                "Find courses that discuss the same topic as lesson 4 of Introduction to Machine Learning",
                tools=mock_tools,
                tool_manager=mock_tool_manager,
            )

            # Verify comprehensive response
            assert (
                "Here are courses that discuss supervised learning algorithms" in result
            )

            # Verify sequential tool execution
            assert mock_tool_manager.execute_tool.call_count == 2
            mock_tool_manager.execute_tool.assert_any_call(
                "get_course_outline", course_name="Introduction to Machine Learning"
            )
            mock_tool_manager.execute_tool.assert_any_call(
                "search_course_content", query="supervised learning algorithms"
            )

            # Verify 3 API calls total
            assert mock_anthropic.return_value.messages.create.call_count == 3

    def test_backward_compatibility_preserved(self):
        """Test that existing single tool usage still works correctly"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            # Single tool use response
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_response.content = [Mock()]
            tool_response.content[0].type = "tool_use"
            tool_response.content[0].name = "search_course_content"
            tool_response.content[0].input = {"query": "machine learning"}
            tool_response.content[0].id = "tool_single"

            # Round 2: Direct text response (no additional tools)
            final_response = mock_anthropic_response(
                "Machine learning is a field of AI that..."
            )

            mock_anthropic.return_value.messages.create.side_effect = [
                tool_response,
                final_response,
            ]

            # Create mock tool manager
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = (
                "ML is a subset of AI focused on learning from data"
            )

            mock_tools = [
                {
                    "name": "search_course_content",
                    "description": "Search course content",
                }
            ]

            generator = AIGenerator("test-key", "test-model")
            result = generator.generate_response(
                "What is machine learning?",
                tools=mock_tools,
                tool_manager=mock_tool_manager,
            )

            # Verify response
            assert result == "Machine learning is a field of AI that..."

            # Verify single tool execution (backward compatibility)
            assert mock_tool_manager.execute_tool.call_count == 1
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content", query="machine learning"
            )

            # Verify 2 API calls (same as before)
            assert mock_anthropic.return_value.messages.create.call_count == 2


class TestErrorPropagation:
    """Test how errors are propagated through the system"""

    def test_anthropic_api_key_error(self):
        """Test handling of API key errors"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_anthropic.return_value.messages.create.side_effect = (
                anthropic.AuthenticationError(
                    "Invalid API key", response=Mock(), body={}
                )
            )

            generator = AIGenerator("invalid-key", "test-model")

            with pytest.raises(anthropic.AuthenticationError):
                generator.generate_response("Test query")

    def test_anthropic_rate_limit_error(self):
        """Test handling of rate limit errors"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            mock_anthropic.return_value.messages.create.side_effect = (
                anthropic.RateLimitError(
                    "Rate limit exceeded", response=Mock(), body={}
                )
            )

            generator = AIGenerator("test-key", "test-model")

            with pytest.raises(anthropic.RateLimitError):
                generator.generate_response("Test query")

    def test_tool_manager_none_error(self):
        """Test when tool_manager is None but tools are provided"""
        with patch("anthropic.Anthropic") as mock_anthropic:
            # Tool use response
            tool_response = Mock()
            tool_response.stop_reason = "tool_use"
            tool_response.content = [Mock()]
            tool_response.content[0].type = "tool_use"
            tool_response.content[0].name = "test_tool"
            tool_response.content[0].input = {}
            tool_response.content[0].id = "tool_id"

            mock_anthropic.return_value.messages.create.return_value = tool_response

            generator = AIGenerator("test-key", "test-model")

            # This should not raise an error, but return the initial response
            result = generator.generate_response(
                "Test query",
                tools=[{"name": "test_tool"}],
                tool_manager=None,  # No tool manager provided
            )

            # Should return the tool use content directly since no tool manager to handle it
            # This might be part of the issue - need to see what actually happens
            assert result is not None
