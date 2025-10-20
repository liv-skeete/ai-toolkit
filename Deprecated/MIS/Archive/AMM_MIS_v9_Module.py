"""
title: AMM_Memory_Identification_Storage
description: Memory Identification & Storage Module for Open WebUI - Identifies and stores memories from user messages
author: Cody
version: 0.9.29
date: 2025-04-09
changes:
- Added valve parameter set truncation length
- Improved async function
- Abstracted prompts
- Removed hardcode threshold fallbacks
- updated logging strategy to use isolated logger with standardized formatting
- renamed debug_mode valve to verbose_logging for consistency with MRE module
- prior changes:
  - fixed KeyError by adding missing memory_relevance_threshold parameter to system prompt formatting
  - simplified reminder handling by removing completion-based deletion
  - focused reminder deletion on explicit deletion requests only
  - removed unnecessary complexity from prompt directives
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple

import aiohttp
from open_webui.models.memories import Memories
from open_webui.models.users import Users
from pydantic import BaseModel, Field

# Logger configuration
logger = logging.getLogger("amm_mis")
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
    """Memory Identification & Storage module for identifying and storing memories from user messages."""

    class Valves(BaseModel):
        # Enable/Disable Function
        enabled: bool = Field(
            default=True,
            description="Enable memory identification & storage",
        )
        # Set processing priority
        priority: int = Field(
            default=3,
            description="Priority level for the filter operations.",
        )
        # UI settings
        show_status: bool = Field(
            default=True,
            description="Show memory operation status in chat",
        )
        # Debug settings
        verbose_logging: bool = Field(
            default=False,
            description="Enable detailed diagnostic logging",
        )
        max_log_lines: int = Field(
            default=10,
            description="Maximum number of lines to show in verbose logs for multi-line content",
        )
        # Background processing settings
        background_processing: bool = Field(
            default=True,
            description="Process memories in the background",
        )
        # Stage 2: Memory Integration settings
        memory_integration: bool = Field(
            default=True,
            description="Enable memory integration stage",
        )
        # API configuration
        api_provider: Literal["OpenAI API", "Ollama API"] = Field(
            default="Ollama API",
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
            default=500,
            description="Maximum tokens for API calls",
        )
        # Stage 1: Memory Identification settings
        memory_importance_threshold: float = Field(
            default=0.5,
            description="Minimum importance threshold for identifying potential memories (0.0-1.0)",
        )
        memory_relevance_threshold: float = Field(
            default=0.5,
            description="Minimum relevance threshold for relating new information to existing memories (0.0-1.0)",
        )
        # Message history settings
        recent_messages_count: int = Field(
            default=2,
            description="Number of recent user messages to analyze for memory (1-10)",
        )
        # Prompt storage with no default values (required fields)
        identification_prompt: str = Field(
            ...,  # This makes the field required with no default
            description="Memory identification prompt with preserved formatting",
        )
        integration_prompt: str = Field(
            ...,  # This makes the field required with no default
            description="Memory integration prompt with preserved formatting",
        )

    # No static prompt constants - prompts are stored in valves

    def __init__(self) -> None:
        """Initialize the Memory Identification & Storage module."""
        logger.info("Initializing Memory Identification & Storage module")

        # Initialize with empty prompts - must be set via update_valves
        try:
            self.valves = self.Valves(
                identification_prompt="",  # Empty string to start - must be set via update_valves
                integration_prompt="",  # Empty string to start - must be set via update_valves
            )
            logger.warning(
                "Prompts are empty - module will not function until prompts are set"
            )
        except Exception as e:
            logger.error(f"Failed to initialize valves: {e}")
            raise

        # Configure aiohttp session with optimized connection pooling
        connector = aiohttp.TCPConnector(
            limit=10,  # Limit total number of connections
            limit_per_host=5,  # Limit connections per host
            enable_cleanup_closed=True,  # Clean up closed connections
            force_close=False,  # Keep connections alive between requests
            ttl_dns_cache=300,  # Cache DNS results for 5 minutes
        )
        self.session = aiohttp.ClientSession(connector=connector)

        self.memory_operations = []
        self.background_tasks = set()  # Track background tasks
        logger.info(
            "MIS module initialized with API provider: %s", self.valves.api_provider
        )

    async def close(self) -> None:
        """Close the aiohttp session and cancel any running background tasks."""
        logger.info("Closing MIS module session")

        # Cancel all running background tasks
        for task in self.background_tasks:
            task.cancel()

        # Wait for all tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        # Close the session
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
                if key.endswith("_prompt") and isinstance(value, str):
                    preview = value[:50] + "..." if len(value) > 50 else value
                    logger.info(f"Updating {key} with: {preview}")
                    setattr(self.valves, key, value)
                else:
                    logger.info(f"Updating {key} with: {value}")
                    setattr(self.valves, key, value)

    def _format_memories(self, db_memories: List[Any]) -> str:
        """
        Format existing memories for inclusion in the prompt.
        Uses a clear, structured format that separates memory content from IDs.

        Args:
            db_memories: List of memory objects from the database

        Returns:
            Formatted string of memories
        """
        if not db_memories:
            return "No existing memories."

        # Create a mapping of memory contents to their IDs
        memory_map = {}
        for mem in db_memories:
            if hasattr(mem, "id") and hasattr(mem, "content") and mem.content:
                memory_map[mem.content] = mem.id

        if not memory_map:
            return "No valid existing memories."

        # Format memories as a numbered list with clear separation between content and ID
        formatted = ["Available memories:"]
        for i, (content, mem_id) in enumerate(memory_map.items(), 1):
            formatted.append(f"{i}. {content} (ID: {mem_id})")

        return "\n".join(formatted)

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

        lines = text.split("\n")
        if len(lines) <= max_lines:
            return text

        truncated = lines[:max_lines]
        omitted = len(lines) - max_lines
        truncated.append(f"... [truncated, {omitted} more lines omitted]")
        return "\n".join(truncated)

    async def _identify_potential_memories(
        self, current_message: str
    ) -> List[Dict[str, Any]]:
        """
        Stage 1: Identify potential memories from the current message based on importance.

        Args:
            current_message: The current user message

        Returns:
            List of potential memories with content and importance scores
        """
        # Essential logging (always shown)
        logger.info("Stage 1: Identifying potential memories from message")

        # Check if prompt is empty and fail fast
        if not self.valves.identification_prompt:
            logger.error("Identification prompt is empty - cannot process memories")
            raise ValueError("Identification prompt is empty - module cannot function")

        # Format the system prompt using the valve-stored prompt
        system_prompt = self.valves.identification_prompt.format(
            current_message=current_message
        )

        # Verbose logging (only when verbose logging is enabled)
        if self.valves.verbose_logging:
            # Log a truncated version of the system prompt
            truncated_prompt = self._truncate_log_lines(system_prompt)
            logger.info("System prompt for Stage 1 (truncated):\n%s", truncated_prompt)

        # Query the appropriate API
        if self.valves.api_provider == "OpenAI API":
            response = await self.query_openai_api(system_prompt, current_message)
        else:  # Ollama API
            response = await self.query_ollama_api(system_prompt, current_message)

        # Verbose logging (only when verbose logging is enabled)
        if self.valves.verbose_logging:
            truncated_response = self._truncate_log_lines(response)
            logger.info("Stage 1: Raw API response: %s", truncated_response)

        # Parse the response
        potential_memories = self._parse_json_response(response)

        # Log a summary of potential memories
        if potential_memories:
            above_threshold = sum(
                1
                for mem in potential_memories
                if mem.get("importance", 0) >= self.valves.memory_importance_threshold
            )
            below_threshold = len(potential_memories) - above_threshold

            # Essential logging (always shown)
            logger.info(
                "Found %d potential memories (%d above threshold, %d below)",
                len(potential_memories),
                above_threshold,
                below_threshold,
            )

            # Verbose logging (only when verbose logging is enabled)
            if self.valves.verbose_logging:
                # Log details of memories that meet the threshold
                for mem in potential_memories:
                    importance = mem.get("importance", 0)
                    if importance >= self.valves.memory_importance_threshold:
                        logger.info(
                            "Memory content: %s (score: %.2f)",
                            mem.get("content", ""),
                            importance,
                        )
        else:
            # Essential logging (always shown)
            logger.info("No potential memories identified")

        # Return all potential memories, regardless of threshold
        # (Filtering will be done later if needed)
        return potential_memories

    async def _integrate_potential_memories(
        self,
        current_message: str,
        potential_memories: List[Dict[str, Any]],
        db_memories: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Determine how to integrate potential memories with existing memories.

        Args:
            current_message: The current user message
            potential_memories: List of potential memories from Stage 1
            db_memories: List of memories from the database
            allow_new_memories: Whether to allow creating new memories (default: True)

        Returns:
            List of memory operations (NEW/UPDATE/DELETE)
        """
        # Essential logging (always shown)
        logger.info("Stage 2: Integrating potential memories with existing memories")

        # Format existing memories
        existing_memories = self._format_memories(db_memories)
        # Verbose logging (only when verbose logging is enabled)
        if self.valves.verbose_logging:
            truncated_memories = self._truncate_log_lines(existing_memories)
            logger.info("Formatted existing memories for LLM:\n%s", truncated_memories)

        # Format potential memories for inclusion in the prompt
        # We know potential_memories is not empty due to early return in _process_memories
        potential_mem_lines = []
        for mem in potential_memories:
            content = mem.get("content", "")
            importance = mem.get("importance", 0)
            potential_mem_lines.append(f"- {content} (importance: {importance:.2f})")
        formatted_potential_memories = "\n".join(potential_mem_lines)

        # Check if prompt is empty and fail fast
        if not self.valves.integration_prompt:
            logger.error("Integration prompt is empty - cannot process memories")
            raise ValueError("Integration prompt is empty - module cannot function")

        # Format the system prompt using the valve-stored prompt
        system_prompt = self.valves.integration_prompt.format(
            current_message=current_message,
            potential_memories=formatted_potential_memories,
            existing_memories=existing_memories,
            memory_relevance_threshold=self.valves.memory_relevance_threshold,
        )

        # Verbose logging (only when verbose logging is enabled)
        if self.valves.verbose_logging:
            # Log a truncated version of the system prompt
            truncated_prompt = self._truncate_log_lines(system_prompt)
            logger.info("System prompt for Stage 2 (truncated):\n%s", truncated_prompt)

        # Query the appropriate API
        if self.valves.api_provider == "OpenAI API":
            response = await self.query_openai_api(system_prompt, current_message)
        else:  # Ollama API
            response = await self.query_ollama_api(system_prompt, current_message)

        # Verbose logging (only when verbose logging is enabled)
        if self.valves.verbose_logging:
            truncated_response = self._truncate_log_lines(response)
            logger.info("Stage 2: Raw API response: %s", truncated_response)

        # Parse the response
        memory_operations = self._parse_json_response(response)

        # No need to filter out NEW operations since potential_memories is never empty
        # and we've removed the allow_new_memories parameter

        # Log a summary of memory operations
        if memory_operations:
            op_types = {}
            for op in memory_operations:
                op_type = op.get("operation", "UNKNOWN")
                op_types[op_type] = op_types.get(op_type, 0) + 1

            # Create summary string
            summary = ", ".join(
                [f"{count} {op_type}" for op_type, count in op_types.items()]
            )
            # Essential logging (always shown)
            logger.info("Memory operations determined: %s", summary)

            # Verbose logging (only when verbose logging is enabled)
            if self.valves.verbose_logging:
                # Log details of each operation with relevance scores and reasoning
                for op in memory_operations:
                    operation_type = op.get("operation", "UNKNOWN")
                    importance = op.get("importance", 0)
                    content = op.get("content", "")
                    mem_id = op.get("id", "N/A") if operation_type != "NEW" else "new"
                    relevance = (
                        op.get("relevance", "N/A") if operation_type != "NEW" else "N/A"
                    )
                    reasoning = op.get("reasoning", "No reasoning provided")

                    if operation_type in ["UPDATE", "DELETE"]:
                        logger.info(
                            "%s memory [id: %s]: %s (importance: %.2f, relevance: %s)",
                            operation_type,
                            mem_id,
                            content,
                            importance,
                            (
                                relevance
                                if isinstance(relevance, str)
                                else f"{relevance:.2f}"
                            ),
                        )
                        logger.info("Reasoning: %s", reasoning)
                    else:
                        logger.info(
                            "%s memory: %s (importance: %.2f)",
                            operation_type,
                            content,
                            importance,
                        )
                        logger.info("Reasoning: %s", reasoning)
        else:
            # Essential logging (always shown)
            logger.info("No memory operations determined")

        # Filter NEW operations by importance threshold, but keep all UPDATE/DELETE operations
        filtered_operations = [
            op
            for op in memory_operations
            if (
                op.get("operation") != "NEW"
                or op.get("importance", 0) >= self.valves.memory_importance_threshold
            )
        ]

        # No longer validating UPDATE operations based on relevance threshold
        # The LLM is responsible for applying the appropriate threshold based on the prompt

        # Sort by combined score (importance + relevance for UPDATE/DELETE, just importance for NEW)
        def get_combined_score(op):
            importance = op.get("importance", 0)
            if op.get("operation") in ["UPDATE", "DELETE"]:
                relevance = op.get("relevance", 0)
                # Convert relevance to float if it's a string
                if isinstance(relevance, str):
                    try:
                        relevance = float(relevance)
                    except ValueError:
                        relevance = 0
                # Weight relevance more heavily for UPDATE/DELETE operations
                return (importance * 0.4) + (relevance * 0.6)
            else:
                return importance

        filtered_operations.sort(key=get_combined_score, reverse=True)

        return filtered_operations

    async def _process_memories(
        self, current_message: str, user_id: str, db_memories: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Process memories through both stages: identification and integration.

        Args:
            current_message: The current user message
            user_id: The user ID
            db_memories: List of memories from the database

        Returns:
            List of memory operations with importance scores
        """
        logger.info("Processing memories for user %s", user_id)

        # Stage 1: Identify potential memories based on importance
        potential_memories = await self._identify_potential_memories(current_message)

        # Log the results of Stage 1 and return early if no memories found
        if not potential_memories:
            logger.info("Stage 1: Did not identify any potential memories")
            return []  # Early return when no potential memories

        # Check if any memories are above the importance threshold
        important_memories = [
            mem
            for mem in potential_memories
            if mem.get("importance", 0.0) >= self.valves.memory_importance_threshold
        ]

        # Log results with information about importance
        logger.info(
            "Stage 1: Identified %d potential memories (%d above threshold)",
            len(potential_memories),
            len(important_memories),
        )

        # Early return if no important memories
        if not important_memories:
            logger.info(
                "No memories above threshold %.2f, skipping Stage 2",
                self.valves.memory_importance_threshold,
            )
            return []  # Early return when no memories above threshold

        # If Stage 2 is disabled or there are no existing memories, convert potential memories directly to NEW operations
        if not self.valves.memory_integration or not db_memories:
            return [
                {
                    "operation": "NEW",
                    "content": mem.get("content", ""),
                    "importance": mem.get("importance", 0.0),
                }
                for mem in potential_memories
                if mem.get("importance", 0.0) >= self.valves.memory_importance_threshold
            ]

        # Stage 2: Integrate potential memories with existing ones
        memory_operations = await self._integrate_potential_memories(
            current_message,  # Pass the raw message
            potential_memories,  # Now guaranteed to be non-empty
            db_memories,
        )

        # Log the results of Stage 2
        logger.info("Stage 2: Determined %d memory operations", len(memory_operations))

        return memory_operations

    async def _query_api(self, provider: str, messages: List[Dict[str, Any]]) -> str:
        """
        Query LLM API with optimized non-blocking patterns and connection pooling.

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

        # Use a timeout to prevent hanging requests
        timeout = aiohttp.ClientTimeout(total=self.valves.request_timeout)

        # Try request with retries using non-blocking patterns
        for attempt in range(self.valves.max_retries + 1):
            try:
                # Use the existing session with proper timeout
                async with self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                    raise_for_status=True,  # Automatically raise for HTTP errors
                ) as response:
                    # Use non-blocking JSON parsing
                    data = await response.json()

                    # Extract content based on provider
                    if provider == "OpenAI API":
                        return str(data["choices"][0]["message"]["content"])
                    else:  # Ollama API
                        return str(data["message"]["content"])

            except asyncio.TimeoutError:
                logger.error(
                    f"API request timed out (attempt {attempt+1}/{self.valves.max_retries+1})"
                )
                if attempt < self.valves.max_retries:
                    # Use non-blocking sleep
                    await asyncio.sleep(self.valves.retry_delay)
                else:
                    logger.error("Max retries reached: Request timed out")
                    return ""
            except aiohttp.ClientResponseError as e:
                logger.error(
                    f"API HTTP error {e.status} (attempt {attempt+1}/{self.valves.max_retries+1}): {e.message}"
                )
                if attempt < self.valves.max_retries:
                    await asyncio.sleep(self.valves.retry_delay)
                else:
                    logger.error(f"Max retries reached: HTTP error {e.status}")
                    return ""
            except Exception as e:
                logger.error(
                    f"API error (attempt {attempt+1}/{self.valves.max_retries+1}): {str(e)}"
                )
                if attempt < self.valves.max_retries:
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

    def _parse_json_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse the JSON response from the LLM.

        Args:
            response_text: The response text from the LLM

        Returns:
            List of parsed objects (potential memories or memory operations)
        """
        if not response_text or not response_text.strip():
            return []

        try:
            # Parse the JSON, removing any markdown code block markers if present
            cleaned = re.sub(r"```json|```", "", response_text.strip())
            data = json.loads(cleaned)

            # Validate the data
            if not isinstance(data, list):
                logger.error("Response is not a JSON array")
                return []

            # Validate each item
            valid_items = []
            for item in data:
                if not isinstance(item, dict):
                    continue

                # For Stage 1 (potential memories)
                if "content" in item and "importance" in item:
                    # Ensure importance is a float
                    try:
                        if "importance" not in item:
                            logger.error(
                                "Missing importance field in memory item: %s", item
                            )
                            continue
                        item["importance"] = float(item["importance"])
                    except (ValueError, TypeError):
                        logger.error(
                            "Invalid importance value '%s' in memory item. Item skipped: %s",
                            item.get("importance"),
                            item,
                        )
                        continue

                    valid_items.append(item)
                    continue

                # For Stage 2 (memory operations)
                if "operation" in item and item["operation"] in [
                    "NEW",
                    "UPDATE",
                    "DELETE",
                ]:
                    if "content" not in item:
                        continue

                    if item["operation"] in ["UPDATE", "DELETE"] and "id" not in item:
                        # For UPDATE and DELETE, we need an ID
                        continue

                    # Ensure importance is a float
                    if "importance" not in item:
                        logger.error(
                            "Missing importance field in memory operation: %s", item
                        )
                        continue

                    try:
                        item["importance"] = float(item["importance"])
                    except (ValueError, TypeError):
                        logger.error(
                            "Invalid importance value '%s' in memory operation. Operation skipped: %s",
                            item.get("importance"),
                            item,
                        )
                        continue

                    valid_items.append(item)

            return valid_items

        except json.JSONDecodeError as e:
            logger.error("Error parsing JSON response: %s", e)
            logger.error("Failed JSON content: %s", cleaned)
            return []

    async def _process_memory_operations(
        self, operations: List[Dict[str, Any]], user: Any
    ) -> List[Dict[str, Any]]:
        """
        Process memory operations.

        Args:
            operations: List of memory operations
            user: The user object

        Returns:
            List of processed operations with status
        """
        # Add logging for operations being processed
        logger.info(
            f"Processing {len(operations)} memory operations: {[op.get('operation') for op in operations]}"
        )
        results = []

        for op in operations:
            try:
                operation = op.get("operation", "UNKNOWN")
                content = op.get("content", "")
                importance = op.get(
                    "importance", 0.0
                )  # Standardized default importance

                if operation == "NEW":
                    # Create a new memory
                    memory_id = await self._create_memory(content, user)
                    results.append(
                        {
                            "operation": "NEW",
                            "content": content,
                            "importance": importance,
                            "id": memory_id,
                            "success": bool(memory_id),
                            "status": "Created" if memory_id else "Failed to create",
                        }
                    )

                elif operation == "UPDATE":
                    # Update an existing memory
                    memory_id = op.get("id")
                    if not memory_id:
                        # Log error and fail gracefully
                        logger.error("UPDATE operation missing required ID field")
                        results.append(
                            {
                                "operation": "UPDATE",
                                "content": content,
                                "importance": importance,
                                "success": False,
                                "status": "Failed: Missing memory ID",
                            }
                        )
                        continue

                    # Process the update with the provided ID
                    success = await self._update_memory(memory_id, content, user)
                    results.append(
                        {
                            "operation": "UPDATE",
                            "content": content,
                            "importance": importance,
                            "id": memory_id,
                            "success": success,
                            "status": "Updated" if success else "Failed to update",
                        }
                    )

                elif operation == "DELETE":
                    # Delete an existing memory
                    memory_id = op.get("id")
                    if not memory_id:
                        # Log error and fail gracefully
                        logger.error("DELETE operation missing required ID field")
                        results.append(
                            {
                                "operation": "DELETE",
                                "content": content,
                                "importance": importance,
                                "success": False,
                                "status": "Failed: Missing memory ID",
                            }
                        )
                        continue

                    success = await self._delete_memory(memory_id, user)
                    results.append(
                        {
                            "operation": "DELETE",
                            "content": content,
                            "importance": importance,
                            "id": memory_id,
                            "success": success,
                            "status": "Deleted" if success else "Failed to delete",
                        }
                    )

                else:
                    # Handle unknown operation type
                    results.append(
                        {
                            "operation": operation,
                            "content": content,
                            "importance": importance,
                            "success": False,
                            "status": f"Failed: Unknown operation type '{operation}'",
                        }
                    )

            except Exception as e:
                logger.error("Error processing memory operation: %s", e)
                results.append(
                    {
                        "operation": op.get("operation", "UNKNOWN"),
                        "content": op.get("content", ""),
                        "importance": op.get("importance", 0),
                        "success": False,
                        "status": f"Error: {str(e)}",
                    }
                )

        # Add summary logging for operation results
        successful_ops = sum(1 for op in results if op.get("success", False))
        logger.info(
            f"Completed {successful_ops}/{len(results)} memory operations successfully"
        )

        # Log details of failed operations
        for op in results:
            if not op.get("success", False):
                logger.error(
                    f"Failed {op.get('operation')} operation: {op.get('status')}"
                )

        # Return results outside the for loop
        return results

    async def _create_memory(self, content: str, user: Any) -> Optional[str]:
        """
        Create a new memory.

        Args:
            content: The memory content
            user: The user object

        Returns:
            The memory ID if successful, None otherwise
        """
        try:
            result = Memories.insert_new_memory(user_id=str(user.id), content=content)
            memory_id = result.id if hasattr(result, "id") else None
            if memory_id:
                # Essential logging (always shown)
                logger.info("Created memory with ID: %s", memory_id)
                # Verbose logging (only when verbose logging is enabled)
                if self.valves.verbose_logging:
                    logger.info("Memory content: %s", content)
                return str(memory_id)
            else:
                logger.error("Failed to create memory: No ID returned")
                return None
        except Exception as e:
            logger.error("Error creating memory: %s", e)
            return None

    async def _update_memory(self, memory_id: str, content: str, user: Any) -> bool:
        """
        Update an existing memory.

        Args:
            memory_id: The memory ID
            content: The new memory content
            user: The user object

        Returns:
            True if successful, False otherwise
        """
        try:
            memory = Memories.get_memory_by_id(memory_id)
            if not memory or str(memory.user_id) != str(user.id):
                return False

            result = Memories.update_memory_by_id(memory_id, content=content)

            # Add this check - only log success and return True if result indicates success
            if result:
                # Essential logging (always shown)
                logger.info("Updated memory with ID: %s", memory_id)
                # Verbose logging (only when verbose logging is enabled)
                if self.valves.verbose_logging:
                    logger.info("New content: %s", content)
                return True
            else:
                # Add this error log to capture failed updates
                logger.error(
                    f"Memory update failed: update_memory_by_id returned {result}"
                )
                return False
        except Exception as e:
            logger.error("Error updating memory %s: %s", memory_id, e)
            return False

    async def _delete_memory(self, memory_id: str, user: Any) -> bool:
        """
        Delete an existing memory.

        Args:
            memory_id: The memory ID
            user: The user object

        Returns:
            True if successful, False otherwise
        """
        try:
            memory = Memories.get_memory_by_id(memory_id)
            if not memory or str(memory.user_id) != str(user.id):
                return False

            deleted = Memories.delete_memory_by_id(memory_id)
            # Essential logging (always shown)
            logger.info("Deleted memory with ID: %s", memory_id)
            # Verbose logging (only when verbose logging is enabled)
            if self.valves.verbose_logging and hasattr(memory, "content"):
                logger.info("Deleted content: %s", memory.content)
            return True
        except Exception as e:
            logger.error("Error deleting memory %s: %s", memory_id, e)
            return False

    # No helper methods needed - using direct attribute access

    async def _process_memories_background(
        self,
        combined_message: str,
        user: Any,
        user_id: str,
        db_memories: Optional[List[Any]] = None,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> None:
        """
        Process memories in the background.

        Args:
            combined_message: The combined user message
            user: The user object
            user_id: The user ID
            event_emitter: Optional event emitter for notifications
        """
        try:
            # Log background task start with timestamp
            import time

            start_time = time.time()
            logger.info("Background memory processing started")

            # Essential logging (always shown)
            logger.info(
                "Starting memory processing for message: %s",
                (
                    combined_message[:50] + "..."
                    if len(combined_message) > 50
                    else combined_message
                ),
            )

            # First, identify potential memories without retrieving existing memories
            # Add timing for memory processing
            stage_start_time = time.time()

            # Stage 1: Identify potential memories based on importance
            logger.info("Processing memories for user %s", user_id)
            potential_memories = await self._identify_potential_memories(
                combined_message
            )

            # Log the results of Stage 1 and return early if no memories found
            if not potential_memories:
                logger.info("Stage 1: Did not identify any potential memories")
                stage_duration = time.time() - stage_start_time
                logger.info(
                    "Memory processing completed in %.2f seconds", stage_duration
                )
                logger.info("Memory operations returned: []")

                # Emit completion status if enabled - use non-blocking version
                if self.valves.show_status and event_emitter:
                    self._non_blocking_emit(
                        event_emitter,
                        {
                            "type": "status",
                            "data": {
                                "description": "🧠 0 Memories changed",
                                "done": True,
                            },
                        },
                    )

                    # Create a background task to clear the status after delay
                    task = asyncio.create_task(
                        self._delayed_clear_status(event_emitter)
                    )
                    # Add task to set and set up callback to remove it when done
                    self.background_tasks.add(task)
                    task.add_done_callback(self.background_tasks.discard)
                return

            # Check if any memories are above the importance threshold
            important_memories = [
                mem
                for mem in potential_memories
                if mem.get("importance", 0.0) >= self.valves.memory_importance_threshold
            ]

            # Log results with information about importance
            logger.info(
                "Stage 1: Identified %d potential memories (%d above threshold)",
                len(potential_memories),
                len(important_memories),
            )

            # Early return if no important memories
            if not important_memories:
                logger.info(
                    "No memories above threshold %.2f, skipping Stage 2",
                    self.valves.memory_importance_threshold,
                )
                stage_duration = time.time() - stage_start_time
                logger.info(
                    "Memory processing completed in %.2f seconds", stage_duration
                )
                logger.info("Memory operations returned: []")

                # Emit completion status if enabled - use non-blocking version
                if self.valves.show_status and event_emitter:
                    self._non_blocking_emit(
                        event_emitter,
                        {
                            "type": "status",
                            "data": {
                                "description": "🧠 0 Memories changed",
                                "done": True,
                            },
                        },
                    )

                    # Create a background task to clear the status after delay
                    task = asyncio.create_task(
                        self._delayed_clear_status(event_emitter)
                    )
                    # Add task to set and set up callback to remove it when done
                    self.background_tasks.add(task)
                    task.add_done_callback(self.background_tasks.discard)
                return

            # Check if Stage 2 (memory integration) is enabled
            if not self.valves.memory_integration:
                # If Stage 2 is disabled, convert potential memories directly to NEW operations
                logger.info(
                    "Memory integration (Stage 2) is disabled, converting potential memories directly to NEW operations"
                )
                memory_operations = [
                    {
                        "operation": "NEW",
                        "content": mem.get("content", ""),
                        "importance": mem.get("importance", 0.0),
                    }
                    for mem in potential_memories
                    if mem.get("importance", 0.0)
                    >= self.valves.memory_importance_threshold
                ]
                logger.info(
                    "Created %d NEW operations from potential memories",
                    len(memory_operations),
                )
            else:
                # Only retrieve existing memories if we have potential memories to compare with
                # and Stage 2 is enabled
                if db_memories is None:
                    db_memories = Memories.get_memories_by_user_id(user_id)

                memory_count = len(db_memories) if db_memories else 0
                logger.info(
                    "Processing %d existing memories for user %s", memory_count, user_id
                )

                # Continue with Stage 2 processing
                memory_operations = await self._integrate_potential_memories(
                    combined_message, potential_memories, db_memories or []
                )
            # If we get here, we have memory operations from Stage 2
            stage_duration = time.time() - stage_start_time
            logger.info("Memory processing completed in %.2f seconds", stage_duration)

            # Verbose logging (only when verbose logging is enabled)
            if self.valves.verbose_logging:
                truncated_operations = self._truncate_log_lines(str(memory_operations))
                logger.info("Memory operations returned: %s", truncated_operations)

            # If no memory operations after Stage 2, emit completion status and return
            if not memory_operations:
                # Emit completion status if enabled - use non-blocking version
                if self.valves.show_status and event_emitter:
                    self._non_blocking_emit(
                        event_emitter,
                        {
                            "type": "status",
                            "data": {
                                "description": "🧠 0 Memories changed",
                                "done": True,
                            },
                        },
                    )

                    # Create a background task to clear the status after delay
                    task = asyncio.create_task(
                        self._delayed_clear_status(event_emitter)
                    )
                    # Add task to set and set up callback to remove it when done
                    self.background_tasks.add(task)
                    task.add_done_callback(self.background_tasks.discard)
                return

            # Emit status update if enabled - use non-blocking version
            if self.valves.show_status and event_emitter:
                self._non_blocking_emit(
                    event_emitter,
                    {
                        "type": "status",
                        "data": {
                            "description": "💭 Processing memories (async)",
                            "done": False,
                        },
                    },
                )

            # Process the memory operations
            logger.info("Processing %d memory operations", len(memory_operations))
            operation_start_time = time.time()
            self.memory_operations = await self._process_memory_operations(
                memory_operations, user
            )
            operation_duration = time.time() - operation_start_time
            logger.info(
                "Memory operations processing completed in %.2f seconds",
                operation_duration,
            )

            # Count operations by type
            new_count = 0
            update_count = 0
            delete_count = 0

            for op in self.memory_operations:
                if op.get("success", False):
                    if op.get("operation") == "NEW":
                        new_count += 1
                    elif op.get("operation") == "UPDATE":
                        update_count += 1
                    elif op.get("operation") == "DELETE":
                        delete_count += 1

            # Create citation content for memory operations
            if self.memory_operations and event_emitter:
                citation_content = "Memory operations:\n"
                for op in self.memory_operations:
                    status = "✅" if op.get("success", False) else "❌"
                    operation = op.get("operation", "UNKNOWN")
                    content = op.get("content", "")
                    importance = op.get("importance", 0)
                    citation_content += f"{status} {operation}: {content} (importance: {importance:.2f})\n"

                # Send citation
                await self._send_citation(
                    event_emitter,
                    url="module://mis/memories",
                    title="Memories Saved",
                    content=citation_content,
                )

            # Emit consolidated status update based on operation counts - use non-blocking version
            if self.valves.show_status and event_emitter:
                # Build a consolidated status message
                status_parts = []
                if new_count > 0:
                    status_parts.append(f"{new_count} created")
                if update_count > 0:
                    status_parts.append(f"{update_count} updated")
                if delete_count > 0:
                    status_parts.append(f"{delete_count} deleted")

                if status_parts:
                    status_message = f"🧠 Memories: {', '.join(status_parts)}"
                else:
                    status_message = "🧠 0 Memories changed"

                # Send a single status update - use non-blocking version
                self._non_blocking_emit(
                    event_emitter,
                    {
                        "type": "status",
                        "data": {
                            "description": status_message,
                            "done": True,
                        },
                    },
                )

                # Create only one background task to clear the status after delay
                task = asyncio.create_task(self._delayed_clear_status(event_emitter))
                # Add task to set and set up callback to remove it when done
                self.background_tasks.add(task)
                task.add_done_callback(self.background_tasks.discard)
            # Clear the memory operations
            self.memory_operations = []

            # Log overall task completion with timing
            total_duration = time.time() - start_time
            logger.info(
                "Background memory processing completed in %.2f seconds", total_duration
            )

            # Add summary of operations
            total_ops = new_count + update_count + delete_count
            if total_ops > 0:
                logger.info(
                    "Memory operations summary: %d created, %d updated, %d deleted",
                    new_count,
                    update_count,
                    delete_count,
                )
            else:
                logger.info("No memory operations were performed")

        except Exception as e:
            # Calculate how far we got in the process
            elapsed_time = time.time() - start_time

            # Enhanced error logging
            logger.error(
                "Error in background memory processing after %.2f seconds: %s",
                elapsed_time,
                e,
            )
            import traceback

            logger.error("Traceback: %s", traceback.format_exc())

            # Log the stage where the error occurred
            if "memory_operations" not in locals():
                logger.error("Error occurred during memory identification stage")
            elif not hasattr(self, "memory_operations") or not self.memory_operations:
                logger.error("Error occurred during memory operations processing stage")
            else:
                logger.error("Error occurred during final processing stage")

            # Emit error status - use non-blocking version
            if self.valves.show_status and event_emitter:
                self._non_blocking_emit(
                    event_emitter,
                    {
                        "type": "status",
                        "data": {
                            "description": "❌ Error processing memories",
                            "done": True,
                        },
                    },
                )
        finally:
            # Ensure the status is cleared if it's still showing "Processing memories (async)"
            if self.valves.show_status and event_emitter:
                # Create a background task to clear the status after a short delay
                # This ensures any stuck "Processing memories (async)" status is cleared
                task = asyncio.create_task(
                    self._delayed_clear_status(
                        event_emitter, delay_seconds=1.0
                    )  # Short delay to ensure other status messages are shown first
                )
                # Add task to set and set up callback to remove it when done
                self.background_tasks.add(task)
                task.add_done_callback(self.background_tasks.discard)

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

    def _non_blocking_emit(
        self,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
        data: Dict[str, Any],
    ) -> None:
        """
        Non-blocking version of _safe_emit that doesn't wait for the event emission to complete.

        Args:
            event_emitter: The event emitter function
            data: The data to emit
        """
        if not event_emitter:
            return

        # Create a task without awaiting it
        task = asyncio.create_task(self._safe_emit(event_emitter, data))

        # Add to background tasks set for proper tracking and cleanup
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def _send_citation(
        self,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
        url: str,
        title: str,
        content: str,
    ) -> None:
        """
        Send a citation event with memory operation information.

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

    async def _delayed_clear_status(
        self,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
        delay_seconds: float = 1.0,  # Reduced from 2.0 to 1.0 seconds
    ) -> None:
        """
        Clear status after a delay, without blocking the main execution flow.

        Args:
            event_emitter: The event emitter function
            delay_seconds: Delay in seconds before clearing (default: 1.0)
        """
        if not event_emitter:
            return

        try:
            # Wait for the specified delay
            await asyncio.sleep(delay_seconds)

            # Clear the status - use non-blocking version
            self._non_blocking_emit(
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

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process incoming messages (called by Open WebUI).
        Identifies and stores memories from user messages.

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
            # Check if there are any user messages
            user_messages = [m for m in body["messages"] if m.get("role") == "user"]
            if not user_messages:
                return body

            # Get the specified number of recent user messages
            recent_count = min(self.valves.recent_messages_count, len(user_messages))
            recent_user_messages = user_messages[-recent_count:]

            # Combine messages with a separator to maintain context
            combined_message = " [Next message] ".join(
                [m.get("content", "") for m in recent_user_messages]
            )
            if not combined_message:
                return body

            logger.info(
                "Processing combined message: %s",
                (
                    combined_message[:50] + "..."
                    if len(combined_message) > 50
                    else combined_message
                ),
            )

            # Get user object
            user = Users.get_user_by_id(__user__["id"])
            if not user:
                return body

            # If background processing is enabled, create a background task
            if self.valves.background_processing:
                # Initial status update - use non-blocking version
                if self.valves.show_status and __event_emitter__:
                    self._non_blocking_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": "💭 Analyzing memories (async)",
                                "done": False,
                            },
                        },
                    )
                # Create and register background task
                logger.info("Creating background memory processing task")

                # Add verbose logging for task tracking
                if self.valves.verbose_logging:
                    logger.info(
                        "Active background tasks before creation: %d",
                        len(self.background_tasks),
                    )

                # No need to pre-fetch memories since we'll check for potential memories first
                task = asyncio.create_task(
                    self._process_memories_background(
                        combined_message, user, __user__["id"], None, __event_emitter__
                    )
                )

                # Add task to set and set up callback for tracking and cleanup
                self.background_tasks.add(task)

                # Define a simplified callback to log task completion
                def task_done_callback(completed_task):
                    try:
                        logger.info("Background memory processing task completed")

                        # Add minimal but useful logging
                        if self.valves.verbose_logging:
                            if completed_task.cancelled():
                                logger.info("Task was cancelled")
                            elif completed_task.exception():
                                logger.info(
                                    "Task failed with exception: %s",
                                    completed_task.exception(),
                                )
                            else:
                                logger.info("Task completed successfully")
                    except Exception as e:
                        logger.error("Error in task completion callback: %s", e)
                    finally:
                        # Remove task from set
                        self.background_tasks.discard(completed_task)

                # Add the simplified callback (only once)
                task.add_done_callback(task_done_callback)

                # Return immediately without waiting for memory processing
                return body
            else:
                # Original synchronous processing
                db_memories = Memories.get_memories_by_user_id(__user__["id"])

                # Emit status update if enabled - use non-blocking version
                if self.valves.show_status and __event_emitter__:
                    self._non_blocking_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": "💭 Analyzing memories (sync)",
                                "done": False,
                            },
                        },
                    )

                # Process memories through both stages
                memory_operations = await self._process_memories(
                    combined_message, __user__["id"], db_memories or []
                )

                # If no memory operations, emit completion status and return body unchanged
                if not memory_operations:
                    # Emit completion status if enabled - use non-blocking version
                    if self.valves.show_status and __event_emitter__:
                        self._non_blocking_emit(
                            __event_emitter__,
                            {
                                "type": "status",
                                "data": {
                                    "description": "🧠 0 Memories changed",
                                    "done": True,
                                },
                            },
                        )

                        # Create a background task to clear the status after delay
                        asyncio.create_task(
                            self._delayed_clear_status(__event_emitter__)
                        )
                    return body

                # Emit status update if enabled - use non-blocking version
                if self.valves.show_status and __event_emitter__:
                    self._non_blocking_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": "💭 Processing memories",
                                "done": False,
                            },
                        },
                    )

                # Process the memory operations
                self.memory_operations = await self._process_memory_operations(
                    memory_operations, user
                )

                # Create citation content for memory operations
                if self.memory_operations and __event_emitter__:
                    citation_content = "Memory operations:\n"
                    for op in self.memory_operations:
                        status = "✅" if op.get("success", False) else "❌"
                        operation = op.get("operation", "UNKNOWN")
                        content = op.get("content", "")
                        importance = op.get("importance", 0)
                        citation_content += f"{status} {operation}: {content} (importance: {importance:.2f})\n"

                    # Send citation
                    await self._send_citation(
                        __event_emitter__,
                        url="module://mis/memories",
                        title="Memories Saved",
                        content=citation_content,
                    )

                # No need to modify the message body - we're just storing memories

        except Exception as e:
            logger.error("Error in inlet: %s", e)
            import traceback

            logger.error("Traceback: %s", traceback.format_exc())

        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
    ) -> dict:
        """
        Process outgoing messages (called by Open WebUI).
        Optionally adds memory operation status to the response.

        Args:
            body: The message body
            __event_emitter__: Event emitter for notifications
            __user__: Optional user information

        Returns:
            The processed message body
        """
        # Skip if disabled or no memory operations and not in background mode
        if not self.valves.enabled or (
            not self.memory_operations and self.valves.background_processing
        ):
            return body

        try:
            # Count operations by type
            new_count = 0
            update_count = 0
            delete_count = 0

            for op in self.memory_operations:
                if op.get("success", False):
                    if op.get("operation") == "NEW":
                        new_count += 1
                    elif op.get("operation") == "UPDATE":
                        update_count += 1
                    elif op.get("operation") == "DELETE":
                        delete_count += 1

            # Emit completion status based on operation counts
            if self.valves.show_status and __event_emitter__:
                # Build a consolidated status message
                status_parts = []
                if new_count > 0:
                    status_parts.append(f"{new_count} created")
                if update_count > 0:
                    status_parts.append(f"{update_count} updated")
                if delete_count > 0:
                    status_parts.append(f"{delete_count} deleted")

                if status_parts:
                    status_message = f"🧠 Memories: {', '.join(status_parts)}"

                    # Send a single status update - use non-blocking version
                    self._non_blocking_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": status_message,
                                "done": True,
                            },
                        },
                    )

                    # Create only one background task to clear the status after delay
                    asyncio.create_task(self._delayed_clear_status(__event_emitter__))

            # Add memory operation status to the response
            if "messages" in body:
                # Format the memory operations status
                status_message = "Memory operations:\n"
                for op in self.memory_operations:
                    status = "✅" if op.get("success", False) else "❌"
                    operation = op.get("operation", "UNKNOWN")
                    content = op.get("content", "")
                    importance = op.get("importance", 0)
                    status_message += f"{status} {operation}: {content} (importance: {importance:.2f})\n"

                # Add as a system message
                body["messages"].append({"role": "system", "content": status_message})

            # Clear the memory operations
            self.memory_operations = []

        except Exception as e:
            logger.error("Error in outlet: %s", e)

        return body
