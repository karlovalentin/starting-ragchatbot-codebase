from typing import Any

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to search and outline tools for course information.

Tool Usage:
- **search_course_content**: Use for questions about specific course content or detailed educational materials
- **get_course_outline**: Use for questions about course structure, lesson lists, or course overview
- **Sequential tool capability**: You can use tools up to 2 times across multiple rounds
- Use tools strategically: initial search, then refined search if needed for comprehensive answers
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Use search_course_content tool first, then answer
- **Course outline/structure questions**: Use get_course_outline tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the outline"

For outline queries, always include:
- Course title
- Course link (if available)
- Complete lesson list with lesson numbers and titles

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: str | None = None,
        tools: list | None = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle sequential tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._execute_sequential_rounds(
                response, api_params, tool_manager, tools
            )

        # Return direct response
        return response.content[0].text

    def _execute_sequential_rounds(
        self, initial_response, base_params: dict[str, Any], tool_manager, tools
    ):
        """
        Execute sequential tool calling rounds (up to 2 rounds maximum).

        Args:
            initial_response: The initial response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            tools: Available tools for subsequent rounds

        Returns:
            Final response text after sequential tool execution
        """
        MAX_ROUNDS = 2
        round_number = 1

        # Start with existing messages
        messages = base_params["messages"].copy()
        current_response = initial_response

        while round_number <= MAX_ROUNDS:
            # Execute tools for current round
            tool_execution_result = self._execute_tools_for_round(
                current_response, messages, tool_manager
            )

            # Termination condition: Tool execution failed
            if tool_execution_result["failed"]:
                return self._handle_tool_failure(tool_execution_result["error"])

            # Update messages with tool execution results
            messages = tool_execution_result["updated_messages"]

            # Termination condition: Reached maximum rounds
            if round_number >= MAX_ROUNDS:
                # Final call without tools to force conclusion
                return self._make_final_call_without_tools(
                    messages, base_params["system"]
                )

            # Prepare for next round - make API call with tools still available
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"],
                "tools": tools,
                "tool_choice": {"type": "auto"},
            }

            try:
                current_response = self.client.messages.create(**api_params)
            except Exception as e:
                return self._handle_tool_failure(str(e))

            # Termination condition: No tool use in response
            if current_response.stop_reason != "tool_use":
                return current_response.content[0].text

            round_number += 1

        # Fallback (should not reach here)
        return self._make_final_call_without_tools(messages, base_params["system"])

    def _execute_tools_for_round(self, response, messages, tool_manager):
        """
        Execute all tool calls for the current round and update conversation context.

        Args:
            response: The response containing tool use requests
            messages: Current message list
            tool_manager: Manager to execute tools

        Returns:
            Dict with execution results and updated messages
        """
        try:
            # Add AI's tool use response to conversation
            messages.append({"role": "assistant", "content": response.content})

            # Execute all tool calls and collect results
            tool_results = []
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )

            # Add tool results to conversation
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            return {
                "failed": False,
                "updated_messages": messages,
                "tool_results": tool_results,
            }

        except Exception as e:
            return {"failed": True, "error": str(e), "updated_messages": messages}

    def _make_final_call_without_tools(self, messages, system_content):
        """
        Make final API call without tools to force Claude to provide conclusion.

        Args:
            messages: Full conversation context
            system_content: System prompt

        Returns:
            Final response text
        """
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
            # Explicitly no tools to force conclusion
        }

        try:
            final_response = self.client.messages.create(**final_params)
            return final_response.content[0].text
        except Exception as e:
            return f"Error generating final response: {e}"

    def _handle_tool_failure(self, error):
        """
        Handle tool execution failures gracefully.

        Args:
            error: Error message

        Returns:
            User-friendly error message
        """
        return f"I encountered an error while searching for information: {error}"
