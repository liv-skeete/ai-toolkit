"""
title: AMM_Memory_Retrieval_Enhancement
description: Memory Retrieval & Enhancement Module for Open WebUI - Retrieves relevant memories and enhances prompts using LLM-based scoring
author: Cody
version: 0.5.18
date: 2025-04-13
changes:
- Corrected memory parsing (_format_memories_for_context) for `Category Name:\n- Bullet` format
- Removed code fences from examples in associated prompts
- Added valve parameter set truncation length
- Sol reworked prompt to stop examples being used as memories
- restructured guidance section to prevent hallucinations while maintaining permissiveness
- added clear verification requirements for memories before inclusion
- strengthened anti-hallucination language with explicit reminders
- emphasized accuracy over helpfulness in memory selection
- prior changes:
  - made relevance prompt more permissive to return more contextually relevant memories
  - reduced strict constraints and negative language in the relevance prompt
- improved debug logging to show up to 50 lines of system prompt for better visibility
- added relevance_threshold valve to set minimum relevance score for memories
- added verbose_logging valve to toggle between concise and verbose logging
- standardized logging strategy to match MIS module
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

import aiohttp
from open_webui.models.memories import Memories
from open_webui.models.users import Users
from pydantic import BaseModel, Field

# Logger configuration
logger = logging.getLogger("amm_mre")
logger.propagate = False
logger.setLevel(logging.INFO)

# Configure handler once
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    # Explicitly set handler level to match logger
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)


class Filter:
    """Memory Retrieval & Enhancement module for retrieving relevant memories and enhancing prompts."""

    class Valves(BaseModel):  
        # Enable/Disable Function
        enabled: bool = Field(
            default=True,
            description="Enable memory retrieval & enhancement",
        )
        # Set processing priority
        priority: int = Field(
            default=2,
            description="Priority level for the filter operations.",
        )  
        # UI settings
        show_status: bool = Field(
            default=True, description="Show memory retrieval status in chat"
        )
        # Debug settings
        verbose_logging: bool = Field(
            default=False,
            description="Enable verbose logging",
        )
        max_log_lines: int = Field(
            default=10,
            description="Maximum number of lines to show in verbose logs for multi-line content",
        )

        # API configuration
        api_provider: Literal["OpenAI API", "Ollama API"] = Field(
            default="OpenAI API",
            description="Choose LLM API provider for memory processing",
        )
        # OpenAI settings
        openai_api_url: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI API endpoint",
        )
        openai_api_key: str = Field(
            default=os.getenv("OPENAI_API_KEY", ""),
            description="OpenAI API key",
        )
        openai_model: str = Field(
            default="gpt-4o-mini",
            description="OpenAI model to use for memory processing",
        )
        # Ollama settings
        ollama_api_url: str = Field(
            default="http://ollama:11434",
            description="Ollama API URL",
        )
        ollama_model: str = Field(
            default="qwen2.5:latest",
            description="Ollama model to use for memory processing",
        )
        ollama_context_size: int = Field(
            default=32768,
            description="Context size (n_ctx) for Ollama model",
        )
        # Common API settings
        request_timeout: int = Field(
            default=30,
            description="Timeout for API requests in seconds",
        )
        max_retries: int = Field(
            default=2,
            description="Maximum number of retries for API calls",
        )
        retry_delay: float = Field(
            default=1.0,
            description="Delay between retries (seconds)",
        )
        temperature: float = Field(
            default=0.3,
            description="Temperature for API calls",
        )
        max_tokens: int = Field(
            default=750,
            description="Maximum tokens for API calls",
        )
        # Memory retrieval settings
        relevance_threshold: float = Field(
            default=0.5,
            description="Minimum relevance score (0.0-1.0) for memories to be included",
        )
        # Prompt storage with no default value (required field)
        relevance_prompt: str = Field(
            ...,  # This makes the field required with no default
            description="Memory relevance prompt with preserved formatting"
        )
        # Other memory retrieval aspects are handled by LLM intelligence
        # No additional explicit parameters needed as the LLM intelligently consolidates memories

    # No static prompt constant - prompt is stored in valves

    def __init__(self) -> None:
        """
        Initialize the Memory Retrieval & Enhancement module.
        """
        logger.info("Initializing Memory Retrieval & Enhancement module")
        
        # Initialize with empty prompt - must be set via update_valves
        try:
            self.valves = self.Valves(
                relevance_prompt=""  # Empty string to start - must be set via update_valves
            )
            logger.warning("Relevance prompt is empty - module will not function until prompt is set")
        except Exception as e:
            logger.error(f"Failed to initialize valves: {e}")
            raise
            
        self.session = aiohttp.ClientSession()

    async def close(self) -> None:
        """Close the aiohttp session."""
        await self.session.close()

    def update_valves(self, new_valves: Dict[str, Any]) -> None:
        """
        Update valve settings.

        Args:
            new_valves: Dictionary of valve settings to update
        """
        logger.info("Updating valves")
        
        for key, value in new_valves.items():
            if hasattr(self.valves, key):
                # For prompt fields, log a truncated version
                if key.endswith('_prompt') and isinstance(value, str):
                    preview = value[:50] + "..." if len(value) > 50 else value
                    logger.info(f"Updating {key} with: {preview}")
                    setattr(self.valves, key, value)
                else:
                    logger.info(f"Updating {key} with: {value}")
                    setattr(self.valves, key, value)

    async def _get_relevant_memories_llm(
        self, current_message: str, db_memories: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Get memories relevant to the current context using LLM-based scoring.

        Args:
            current_message: The current user message
            db_memories: List of memories from the database

        Returns:
            List of dictionaries containing relevant memories with their scores
        """
        if self.valves.verbose_logging:
            logger.info("Getting relevant memories using LLM-based scoring")

        # Extract memory contents
        memory_contents = [
            mem.content
            for mem in db_memories
            if hasattr(mem, "content") and mem.content
        ]

        # If no valid memory contents, return empty list
        if not memory_contents:
            if self.valves.verbose_logging:
                logger.info("No valid memory contents found in database")
            return []

        # Format memories for the prompt
        formatted_memories = "\n".join([f"- {mem}" for mem in memory_contents])

        if self.valves.verbose_logging:
            logger.info(f"Processing {len(memory_contents)} memories from database")

        # Check if prompt is empty and fail fast
        if not self.valves.relevance_prompt:
            logger.error("Relevance prompt is empty - cannot process memories")
            raise ValueError("Relevance prompt is empty - module cannot function")
            
        # Format the system prompt using the valve-stored prompt
        system_prompt = self.valves.relevance_prompt.format(
            current_message=current_message,
            memories=formatted_memories,
        )

        if self.valves.verbose_logging:
            # Log a truncated version of the system prompt
            truncated_prompt = self._truncate_log_lines(system_prompt)
            logger.info(f"System prompt (truncated):\n{truncated_prompt}")

        # Query the appropriate API based on the provider setting
        if self.valves.api_provider == "OpenAI API":
            if self.valves.verbose_logging:
                logger.info(
                    f"Querying OpenAI API with model: {self.valves.openai_model}"
                )
            response = await self.query_openai_api(system_prompt, current_message)
        else:  # Ollama API
            if self.valves.verbose_logging:
                logger.info(
                    f"Querying Ollama API with model: {self.valves.ollama_model}"
                )
            response = await self.query_ollama_api(system_prompt, current_message)

        # Log the raw response for debugging
        if self.valves.verbose_logging:
            truncated_response = self._truncate_log_lines(response)
            logger.info(f"Raw API response: {truncated_response}")

        # Prepare and parse the response
        prepared_response = self._prepare_json_response(response)
        try:
            memory_data = json.loads(prepared_response)

            # Basic validation and processing
            valid_memories = []
            if isinstance(memory_data, list):
                for mem in memory_data:
                    if isinstance(mem, dict) and "text" in mem and "score" in mem:
                        try:
                            score = float(mem["score"])
                            valid_memories.append(
                                {"text": str(mem["text"]), "score": score}
                            )
                        except (ValueError, TypeError):
                            if self.valves.verbose_logging:
                                logger.warning(f"Invalid memory format: {mem}")
                            pass

            # Sort by score in descending order
            valid_memories.sort(key=lambda x: x["score"], reverse=True)

            # Count total memories before threshold filtering
            total_memories = len(valid_memories)

            # Filter memories based on the relevance threshold
            threshold_filtered_memories = [
                mem
                for mem in valid_memories
                if mem["score"] >= self.valves.relevance_threshold
            ]

            # Log the filtering results - ALWAYS LOG THIS (not conditional)
            filtered_count = total_memories - len(threshold_filtered_memories)
            if filtered_count > 0:
                logger.info(
                    f"Found {total_memories} memories, filtered {filtered_count} below threshold {self.valves.relevance_threshold:.2f}"
                )
            else:
                logger.info(f"Found {total_memories} relevant memories")

            # Log details of memories in debug mode
            if self.valves.verbose_logging and threshold_filtered_memories:
                for i, mem in enumerate(threshold_filtered_memories):
                    logger.info(
                        f"Memory {i+1}: {mem['text']} (score: {mem['score']:.2f})"
                    )

                # Log filtered out memories in debug mode
                if filtered_count > 0:
                    logger.info(
                        f"Memories filtered out by threshold {self.valves.relevance_threshold:.2f}:"
                    )
                    for i, mem in enumerate(
                        [
                            m
                            for m in valid_memories
                            if m["score"] < self.valves.relevance_threshold
                        ]
                    ):
                        logger.info(
                            f"Filtered {i+1}: {mem['text']} (score: {mem['score']:.2f})"
                        )

            return threshold_filtered_memories

        except Exception as e:
            logger.error(f"Error processing LLM response: {str(e)}")
            if self.valves.verbose_logging:
                logger.error(f"Failed JSON content: {prepared_response}")
            return []

    async def get_relevant_memories(
        self,
        current_message: str,
        user_id: str,
        db_memories: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get memories relevant to the current context using LLM-based scoring.

        Args:
            current_message: The current user message
            user_id: The user ID
            db_memories: List of memories from the database

        Returns:
            List of dictionaries containing relevant memories with their scores
        """
        if not self.valves.enabled or not db_memories:
            return []

        try:
            # Use LLM for memory retrieval
            return await self._get_relevant_memories_llm(current_message, db_memories)
        except Exception as e:
            logger.error(f"Error in get_relevant_memories: {str(e)}")
            return []

    def _format_memories_for_context(
        self, relevant_memories: List[Dict[str, Any]]
    ) -> tuple[str, str]:
        """
        Format relevant memories for inclusion in the context.
        Preserves the category-based format (category headers and bullet points).

        Args:
            relevant_memories: List of dictionaries containing relevant memories with their scores

        Returns:
            Tuple containing:
            - Formatted string of memories for inclusion in the context (without scores)
            - Formatted string of memories with scores for logging
        """
        if not relevant_memories:
            return "", ""

        formatted_memories = ["User Information (sorted by relevance):"]
        formatted_memories_with_scores = [
            "User Information (sorted by relevance):"
        ]

        # Group memories by category to maintain the structured format
        category_memories = {}
        category_memories_with_scores = {}

        for memory in relevant_memories:
            text = memory["text"]
            score = memory["score"]
            
            # Check if the memory follows the standard "Category Name:\n- Bullet" format
            lines = text.strip().split('\n')
            if len(lines) > 1 and lines[0].strip().endswith(":") and lines[1].strip().startswith("- "):
                # Attempt to parse standard format
                try:
                    category_header = lines[0].strip() # e.g., "User Profile:"
                    
                    # Initialize category if not exists
                    if category_header not in category_memories:
                        category_memories[category_header] = []
                        category_memories_with_scores[category_header] = []
                    
                    # Extract bullet points (lines starting with "- ")
                    found_bullets = False
                    for line in lines[1:]:
                        line_stripped = line.strip()
                        if line_stripped.startswith("- "):
                            point = line_stripped[2:].strip() # Get content after "- "
                            # Check for sub-bullets (indented) - basic handling
                            indent = len(line) - len(line.lstrip(' '))
                            prefix = "  " * (indent // 2) + "🔎 - " # Add indentation and emoji back
                            category_memories[category_header].append(f"{prefix}{point}") # Keep original format for context
                            category_memories_with_scores[category_header].append(f"{prefix}{point} (score: {score:.2f})")
                            found_bullets = True
                    
                    # If header found but no bullets parsed (e.g., format error), treat content as single bullet
                    if not found_bullets:
                         logger.warning(f"Memory had category header but no bullets parsed: '{text[:50]}...'")
                         point = "\n".join(lines[1:]).strip() # Use rest of lines as content
                         if point: # Avoid adding empty bullets
                             category_memories[category_header].append(f"- {point}") # Keep original format for context
                             category_memories_with_scores[category_header].append(f"🔎 - {point} (score: {score:.2f})")

                except Exception as e: # Catch potential errors during parsing
                    logger.warning(f"Error parsing formatted memory '{text[:50]}...': {e}")
                    # Fallback to simple format if any parsing error occurs
                    formatted_memories.append(f"- {text}") # Append original text as single bullet
                    formatted_memories_with_scores.append(f"🔎 - {text} (score: {score:.2f})")
            else:
                # Handle memories not in the standard markdown category-based format
                # Append them directly as simple bullet points
                formatted_memories.append(f"- {text}") # Keep original format for context
                formatted_memories_with_scores.append(f"🔎 - {text} (score: {score:.2f})")
        
        # Add the categorized memories to the formatted output
        for category, points in category_memories.items():
            formatted_memories.append(f"\n{category}")
            formatted_memories.extend(points)
        
        for category, points in category_memories_with_scores.items():
            formatted_memories_with_scores.append(f"\n{category}")
            formatted_memories_with_scores.extend(points)

        return "\n".join(formatted_memories), "\n".join(formatted_memories_with_scores)

    def _truncate_log_lines(self, text: str, max_lines: int = None) -> str:
        """
        Truncate a multi-line string to a maximum number of lines.
        
        Args:
            text: The text to truncate
            max_lines: Maximum number of lines (defaults to self.valves.max_log_lines)
            
        Returns:
            Truncated text with indicator of how many lines were omitted
        """
        if not max_lines:
            max_lines = self.valves.max_log_lines
            
        lines = text.split('\n')
        if len(lines) <= max_lines:
            return text
            
        truncated = lines[:max_lines]
        omitted = len(lines) - max_lines
        truncated.append(f"... [truncated, {omitted} more lines omitted]")
        return '\n'.join(truncated)

    def _update_message_context(self, body: dict, formatted_memories: str) -> None:
        """
        Update message context with relevant memories.

        Args:
            body: The message body
            formatted_memories: Formatted string of memories
        """
        if not formatted_memories:
            return

        if "messages" in body and len(body["messages"]) > 0:
            # Insert the memories as a system message at the beginning of the conversation
            body["messages"].insert(
                0, {"role": "system", "content": formatted_memories}
            )

    async def _safe_emit(
        self,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
        data: Dict[str, Any],
    ) -> None:
        """
        Safely emit an event, handling missing emitter.

        Args:
            event_emitter: The event emitter function
            data: The data to emit
        """
        if not event_emitter:
            logger.debug("Event emitter not available")
            return

        try:
            await event_emitter(data)
        except Exception as e:
            logger.error(f"Error in event emitter: {e}")

    async def _delayed_clear_status(
        self,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
        delay_seconds: float = 2.0,
    ) -> None:
        """
        Clear status after a delay, without blocking the main execution flow.

        Args:
            event_emitter: The event emitter function
            delay_seconds: Delay in seconds before clearing
        """
        if not event_emitter:
            return

        try:
            # Wait for the specified delay
            await asyncio.sleep(delay_seconds)

            # Clear the status
            await self._safe_emit(
                event_emitter,
                {
                    "type": "status",
                    "data": {
                        "description": "",
                        "done": True,
                    },
                },
            )
        except Exception as e:
            logger.error(f"Error in delayed status clearing: {e}")

    async def _send_citation(
        self,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
        url: str,
        title: str,
        content: str,
    ) -> None:
        """
        Send a citation event with memory information.

        Args:
            event_emitter: The event emitter function
            url: Citation URL identifier
            title: Citation title
            content: Citation content
        """
        if not event_emitter:
            return

        await self._safe_emit(
            event_emitter,
            {
                "type": "citation",
                "data": {
                    "document": [content],
                    "metadata": [{"source": url, "html": False}],
                    "source": {"name": title},
                },
            },
        )

    def _prepare_json_response(self, response_text: str) -> str:
        """
        Prepare API response for JSON parsing with minimal cleaning.

        Args:
            response_text: The raw response text from the API

        Returns:
            A minimally cleaned string ready for JSON parsing
        """
        if not response_text or not response_text.strip():
            return "[]"  # Return empty array for empty responses

        # Remove leading/trailing whitespace and markdown code block markers
        cleaned = re.sub(r"```json|```", "", response_text.strip())

        return cleaned

    async def _query_api(self, provider: str, messages: List[Dict[str, Any]]) -> str:
        """
        Query LLM API with retry logic.

        Args:
            provider: The API provider ("OpenAI API" or "Ollama API")
            messages: Array of message objects with role and content

        Returns:
            The API response content as a string
        """
        # Prepare request based on provider
        if provider == "OpenAI API":
            url = f"{self.valves.openai_api_url}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.valves.openai_api_key}",
            }
            payload = {
                "model": self.valves.openai_model,
                "messages": messages,
                "temperature": self.valves.temperature,
                "max_tokens": self.valves.max_tokens,
            }
        else:  # Ollama API
            url = f"{self.valves.ollama_api_url.rstrip('/')}/api/chat"
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": self.valves.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.valves.temperature,
                    "num_ctx": self.valves.ollama_context_size,
                },
            }

        # Try request with retries
        for attempt in range(self.valves.max_retries + 1):
            try:
                async with self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.valves.request_timeout,
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

                    # Extract content based on provider
                    if provider == "OpenAI API":
                        return str(data["choices"][0]["message"]["content"])
                    else:  # Ollama API
                        return str(data["message"]["content"])

            except Exception as e:
                if attempt < self.valves.max_retries:
                    logger.error(
                        f"API error (attempt {attempt+1}/{self.valves.max_retries+1}): {str(e)}"
                    )
                    await asyncio.sleep(self.valves.retry_delay)
                else:
                    logger.error(f"Max retries reached: {str(e)}")
                    return ""

    async def query_openai_api(self, system_prompt: str, prompt: str) -> str:
        """
        Query OpenAI API with system prompt and user prompt.

        Args:
            system_prompt: The system prompt
            prompt: The user prompt

        Returns:
            The API response content as a string
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return await self._query_api("OpenAI API", messages)

    async def query_ollama_api(self, system_prompt: str, prompt: str) -> str:
        """
        Query Ollama API with system prompt and user prompt.

        Args:
            system_prompt: The system prompt
            prompt: The user prompt

        Returns:
            The API response content as a string
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return await self._query_api("Ollama API", messages)

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process incoming messages (called by Open WebUI).
        Enhances the message context with relevant memories.

        Args:
            body: The message body
            __event_emitter__: Optional event emitter for notifications
            __user__: Optional user information

        Returns:
            The processed message body
        """
        # Basic validation
        if (
            not self.valves.enabled
            or not body
            or not isinstance(body, dict)
            or not __user__
        ):
            return body

        # Process only if we have messages
        if not body.get("messages"):
            return body

        try:
            logger.info("Processing inlet request for user %s", __user__["id"])

            # Check if there are any user messages
            user_messages = [m for m in body["messages"] if m.get("role") == "user"]
            if not user_messages:
                if self.valves.verbose_logging:
                    logger.info("No user messages found in request")
                return body

            # Get the last user message
            last_user_message = user_messages[-1].get("content", "")
            if not last_user_message:
                if self.valves.verbose_logging:
                    logger.info("Last user message is empty")
                return body

            if self.valves.verbose_logging:
                truncated_message = (
                    last_user_message[:50] + "..."
                    if len(last_user_message) > 50
                    else last_user_message
                )
                logger.info("Processing user message: %s", truncated_message)

            # Get user object and memories
            user = Users.get_user_by_id(__user__["id"])
            if not user:
                if self.valves.verbose_logging:
                    logger.info("User not found in database")
                return body

            db_memories = Memories.get_memories_by_user_id(__user__["id"])
            if not db_memories:
                if self.valves.verbose_logging:
                    logger.info("No memories found for user %s", __user__["id"])
                return body

            logger.info(
                "Found %d memories for user %s", len(db_memories), __user__["id"]
            )

            # Emit status update if enabled
            if self.valves.show_status and __event_emitter__:
                if self.valves.verbose_logging:
                    logger.info("Emitting status: Retrieving relevant memories...")
                await self._safe_emit(
                    __event_emitter__,
                    {
                        "type": "status",
                        "data": {
                            "description": "💭 Retrieving relevant memories...",
                            "done": False,
                        },
                    },
                )

            # Get relevant memories using LLM-based scoring
            relevant_memories = await self.get_relevant_memories(
                last_user_message, __user__["id"], db_memories
            )

            # If no relevant memories, emit status and return body unchanged
            if not relevant_memories:
                logger.info("No relevant memories found for the current message")

                # Emit completion status if enabled
                if self.valves.show_status and __event_emitter__:
                    if self.valves.verbose_logging:
                        logger.info("Emitting status: No relevant memories found")
                    await self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": "☑ No relevant memories found",
                                "done": True,
                            },
                        },
                    )

                    # Create a background task to clear the status after delay
                    asyncio.create_task(self._delayed_clear_status(__event_emitter__))

                return body

            # Format memories and update context
            formatted_memories, formatted_memories_with_scores = (
                self._format_memories_for_context(relevant_memories)
            )

            logger.info(
                "Formatted %d relevant memories for context", len(relevant_memories)
            )

            # Update message context
            self._update_message_context(body, formatted_memories)

            # Log what's being sent to the assistant
            if self.valves.verbose_logging:
                # In verbose mode, show full memory content with scores
                truncated_memories = self._truncate_log_lines(formatted_memories_with_scores)
                logger.info(f"Sending to assistant: {truncated_memories}")
            else:
                # In normal mode, just show count of memories
                logger.info(f"Sending to assistant: {len(relevant_memories)} relevant memories")

            # Send citation with memories sent to assistant
            if __event_emitter__:
                if self.valves.verbose_logging:
                    logger.info("Sending citation with memories")
                await self._send_citation(
                    __event_emitter__,
                    url="module://mre/memories",
                    title="Memories Read",
                    content=formatted_memories_with_scores,
                )

            # Emit completion status if enabled
            if self.valves.show_status and __event_emitter__:
                if self.valves.verbose_logging:
                    logger.info(
                        "Emitting status: Retrieved %d relevant memories",
                        len(relevant_memories),
                    )
                await self._safe_emit(
                    __event_emitter__,
                    {
                        "type": "status",
                        "data": {
                            "description": f"☑ Retrieved {len(relevant_memories)} relevant memories (threshold: {self.valves.relevance_threshold:.1f})",
                            "done": True,
                        },
                    },
                )

                # Create a background task to clear the status after delay
                asyncio.create_task(self._delayed_clear_status(__event_emitter__))

        except Exception as e:
            logger.error(f"Error in inlet: {str(e)}")
            if self.valves.verbose_logging:
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")

        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> dict:
        """Pass-through method for outgoing messages."""
        return body
