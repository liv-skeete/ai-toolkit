"""
title: AMM_Memory_Identification_Storage
description: Memory Identification & Storage Module for Open WebUI - Identifies and stores memories from user messages
author: Cody
version: 1.1.0
date: 2025-04-11
changes:
- Refactored to remove Stage 2 integration logic (moved to MMC module)
- Removed distinction between sync/async processing (now always sync)
- Removed background processing logic
- Simplified memory operations to only handle NEW memories
- Removed obsolete valves (enabled, memory_integration, background_processing, memory_relevance_threshold, integration_prompt)
- Updated logging messages to remove stage references
- prior changes (v0.9.x):
  - Added valve parameter set truncation length
  - Improved async function
  - Abstracted prompts
  - Removed hardcode threshold fallbacks
  - updated logging strategy to use isolated logger with standardized formatting
  - renamed debug_mode valve to verbose_logging for consistency with MRE module
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
import time  # Import time for synchronous processing timing
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

import aiohttp
from open_webui.models.memories import Memories
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
            default="qwen2.5:14b",
            description="Ollama model to use for memory processing",
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
        # Memory Identification settings
        memory_importance_threshold: float = Field(
            default=0.5,
            description="Minimum importance threshold for identifying potential memories (0.0-1.0)",
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

    # No static prompt constants - prompts are stored in valves

    def __init__(self) -> None:
        """Initialize the Memory Identification & Storage module."""
        logger.info("Initializing Memory Identification & Storage module")

        # Initialize with empty prompts - must be set via update_valves
        try:
            self.valves = self.Valves(
                identification_prompt="",  # Empty string to start - must be set via update_valves
                # integration_prompt removed
            )
            # This warning is expected during initial load before update_valves is called
            logger.warning(
                "Identification prompt is empty - module will not function until prompt is set"
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

        # self.background_tasks removed
        logger.info(
            "MIS module initialized with API provider: %s", self.valves.api_provider
        )

    async def close(self) -> None:
        """Close the aiohttp session."""
        logger.info("Closing MIS module session")

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
                if key == "identification_prompt":
                    if isinstance(value, str) and value.strip():
                        preview = (
                            value[:50].replace("\n", "\\n") + "..."
                            if len(value) > 50
                            else value.replace("\n", "\\n")
                        )
                        logger.info(f"Updating {key} with preview: {preview}")
                        setattr(self.valves, key, value)
                    else:
                        logger.warning(
                            f"Received empty or non-string value for identification_prompt: type={type(value)}"
                        )
                        # Ensure it's at least an empty string if invalid value received
                        setattr(self.valves, key, "")
                # Handle other valve types
                elif (
                    key != "identification_prompt"
                ):  # Avoid logging prompt again if handled above
                    logger.info(f"Updating {key} with: {value}")
                    setattr(self.valves, key, value)

        logger.info("Finished updating valves")

    def _format_memories(self, db_memories: List[Any]) -> str:
        """
        Format existing memories for inclusion in the prompt. (Currently unused after refactor)
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

    # --------------------------------------------------------------------------
    # Core Memory Processing Logic
    # --------------------------------------------------------------------------

    async def _identify_potential_memories(
        self, current_message: str
    ) -> List[Dict[str, Any]]:
        """
        Identify potential memories from the current message based on importance.

        Args:
            current_message: The current user message

        Returns:
            List of potential memories with content and importance scores
        """
        # Essential logging (always shown)
        logger.info("Identifying potential memories from message")

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
            logger.info(
                "System prompt for Identification (truncated):\n%s", truncated_prompt
            )

        # Query the appropriate API
        if self.valves.api_provider == "OpenAI API":
            response = await self.query_openai_api(system_prompt, current_message)
        else:  # Ollama API
            response = await self.query_ollama_api(system_prompt, current_message)

        # Verbose logging (only when verbose logging is enabled)
        if self.valves.verbose_logging:
            truncated_response = self._truncate_log_lines(response)
            logger.info("Identification: Raw API response: %s", truncated_response)

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
        return potential_memories

    async def _identify_and_prepare_memories(
        self, current_message: str
    ) -> List[Dict[str, Any]]:  # Returns list of NEW memory operations
        """
        Identifies potential memories and prepares NEW memory operations.

        Args:
            current_message: The current user message

        Returns:
            List of NEW memory operations for memories meeting the importance threshold.
            Returns an empty list if no important memories are identified.
        """
        logger.info("Starting memory identification for message")

        # Identify potential memories
        potential_memories = await self._identify_potential_memories(current_message)

        # Filter potential memories by importance threshold
        important_memories = [
            mem
            for mem in potential_memories
            if mem.get("importance", 0) >= self.valves.memory_importance_threshold
        ]

        if not important_memories:
            logger.info(
                "No potential memories met the importance threshold (%.2f). No operations needed.",
                self.valves.memory_importance_threshold,
            )
            return []

        # Log the important memories identified
        logger.info(
            "Identified %d important potential memories meeting threshold (%.2f)",
            len(important_memories),
            self.valves.memory_importance_threshold,
        )

        # Directly create NEW operations for the important memories
        memory_operations = [
            {
                "operation": "NEW",
                "content": mem["content"],
                "importance": mem["importance"],
            }
            for mem in important_memories
        ]

        # Log the final set of operations
        op_summary = ", ".join(
            f"NEW({op.get('importance', 0):.2f})" for op in memory_operations
        )
        logger.info("Prepared NEW memory operations: %s", op_summary)

        return memory_operations

    # --------------------------------------------------------------------------
    # API Query Helpers
    # --------------------------------------------------------------------------

    async def _query_api(self, provider: str, messages: List[Dict[str, Any]]) -> str:
        """
        Generic function to query either OpenAI or Ollama API with retry logic.

        Args:
            provider: "OpenAI API" or "Ollama API"
            messages: List of messages for the API call

        Returns:
            API response content as a string
        """
        retries = 0
        while retries <= self.valves.max_retries:
            try:
                if provider == "OpenAI API":
                    url = f"{self.valves.openai_api_url}/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {self.valves.openai_api_key}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "model": self.valves.openai_model,
                        "messages": messages,
                        "temperature": self.valves.temperature,
                        "max_tokens": self.valves.max_tokens,
                        "response_format": {"type": "json_object"},
                    }
                else:  # Ollama API
                    url = f"{self.valves.ollama_api_url.rstrip('/')}/api/chat"
                    headers = {"Content-Type": "application/json"}
                    payload = {
                        "model": self.valves.ollama_model,
                        "messages": messages,
                        "stream": False,
                        "format": "json",
                        "options": {
                            "temperature": self.valves.temperature,
                            # Ollama doesn't have a direct max_tokens equivalent in the same way
                            # num_predict can be used, but it's not exactly the same
                        },
                    }

                async with self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.valves.request_timeout,
                ) as response:
                    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                    data = await response.json()

                    if provider == "OpenAI API":
                        content = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                    else:  # Ollama API
                        content = data.get("message", {}).get("content", "")

                    if not content:
                        raise ValueError("API returned empty content")
                    return content

            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                ValueError,
                json.JSONDecodeError,
                KeyError,
            ) as e:
                logger.warning(
                    f"API call failed (attempt {retries + 1}/{self.valves.max_retries + 1}): {e}"
                )
                retries += 1
                if retries > self.valves.max_retries:
                    logger.error("Max retries exceeded. Failing operation.")
                    raise  # Re-raise the last exception
                await asyncio.sleep(
                    self.valves.retry_delay * (2 ** (retries - 1))
                )  # Exponential backoff

        # Should not be reachable if max_retries >= 0
        raise RuntimeError("API query failed after maximum retries.")

    async def query_openai_api(self, system_prompt: str, prompt: str) -> str:
        """
        Query the OpenAI API.

        Args:
            system_prompt: The system prompt content
            prompt: The user prompt content

        Returns:
            API response content
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return await self._query_api("OpenAI API", messages)

    async def query_ollama_api(self, system_prompt: str, prompt: str) -> str:
        """
        Query the Ollama API.

        Args:
            system_prompt: The system prompt content
            prompt: The user prompt content

        Returns:
            API response content
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return await self._query_api("Ollama API", messages)

    # --------------------------------------------------------------------------
    # Response Parsing
    # --------------------------------------------------------------------------

    def _parse_json_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse the JSON response from the API, handling potential errors.

        Args:
            response_text: The raw response text from the API

        Returns:
            List of dictionaries representing memories or operations, or empty list on error.
        """
        try:
            # Attempt to find JSON within potential markdown code blocks
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
            if match:
                json_str = match.group(1).strip()
                logger.debug("Extracted JSON from markdown block")
            else:
                # Assume the whole response might be JSON, try cleaning it
                json_str = response_text.strip()
                # Remove potential leading/trailing non-JSON characters if necessary
                # This is a basic cleanup, might need refinement
                if not json_str.startswith(("[", "{")):
                    json_str = (
                        json_str[json_str.find("[") :]
                        if "[" in json_str
                        else json_str[json_str.find("{") :]
                    )
                if not json_str.endswith(("]", "}")):
                    json_str = (
                        json_str[: json_str.rfind("]") + 1]
                        if "]" in json_str
                        else json_str[: json_str.rfind("}") + 1]
                    )

                logger.debug("Attempting to parse response directly as JSON")

            # Ensure the extracted string is not empty
            if not json_str:
                logger.warning("Could not extract valid JSON content from response.")
                return []

            # Parse the JSON string
            parsed_data = json.loads(json_str)

            # Standardize the output format to always be a list of dictionaries
            if isinstance(parsed_data, dict):
                # If the API returns a single dictionary, check if it contains a list
                # under a common key like 'memories' or 'operations'
                for key in ["memories", "operations", "memory_operations"]:
                    if key in parsed_data and isinstance(parsed_data[key], list):
                        logger.debug(f"Found list under key '{key}'")
                        return parsed_data[key]
                # If it's a single dictionary without a list, wrap it in a list
                logger.debug("API returned a single dictionary, wrapping in list")
                return [parsed_data]
            elif isinstance(parsed_data, list):
                # Ensure all items in the list are dictionaries
                if all(isinstance(item, dict) for item in parsed_data):
                    logger.debug("API returned a list of dictionaries")
                    return parsed_data
                else:
                    logger.warning(
                        "API returned a list containing non-dictionary items. Discarding."
                    )
                    return []
            else:
                logger.warning(
                    f"Parsed JSON is not a dictionary or list, but {type(parsed_data)}. Discarding."
                )
                return []

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            if self.valves.verbose_logging:
                logger.error(
                    "Raw response causing error:\n%s",
                    self._truncate_log_lines(response_text),
                )
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}", exc_info=True)
            if self.valves.verbose_logging:
                logger.error(
                    "Raw response causing error:\n%s",
                    self._truncate_log_lines(response_text),
                )
            return []

    # --------------------------------------------------------------------------
    # Database Operations
    # --------------------------------------------------------------------------

    async def _execute_memory_creation(
        self,
        memory_operations: List[Dict[str, Any]],
        __user__: dict,
        __event_emitter__: Optional[Callable[..., Awaitable[None]]] = None,
    ) -> None:
        """
        Executes the NEW memory creation operations.

        Args:
            memory_operations: List of NEW operations from _identify_and_prepare_memories
            __user__: The user dictionary object
            __event_emitter__: Optional event emitter for sending status updates
        """
        logger.info(
            "Executing %d NEW memory creation operations", len(memory_operations)
        )
        if not memory_operations:
            return

        # Store status messages with importance for sorting (only NEW operations)
        status_updates_with_importance = []

        for op in memory_operations:
            operation_type = op.get("operation")

            # Only NEW operations are expected now
            if operation_type == "NEW":
                content = op.get("content")
                importance = op.get("importance", 0)  # Get importance for logging
                if content:
                    logger.info(
                        "Attempting to CREATE memory for user %s (Importance: %.2f): %s",
                        __user__.get("id", "UNKNOWN"),
                        importance,
                        content[:100] + "...",
                    )
                    memory_id = await self._create_memory(content, __user__)
                    if memory_id:
                        status = f"{content} (importance: {importance:.2f}) ✅"
                        logger.info(status)
                        # Store status with importance for sorting
                        status_updates_with_importance.append((status, importance))
                    else:
                        status = "⚠️ Failed to create new memory"
                        logger.warning(status)
                        # Failed operations get importance 0 for sorting
                        status_updates_with_importance.append((status, 0))
                else:
                    logger.warning("Skipping NEW operation: Content is missing")
                    # Skipped operations get importance 0 for sorting
                    status_updates_with_importance.append(
                        ("⚠️ Skipped creating memory: missing content", 0)
                    )
            else:
                # This case should not happen after refactoring
                logger.warning(
                    "Skipping unexpected operation type '%s': %s", operation_type, op
                )
                # Unknown operations get importance 0 for sorting
                status_updates_with_importance.append(
                    (f"⚠️ Skipped unknown operation: {operation_type}", 0)
                )

        # Sort status updates by importance (highest first)
        sorted_status_updates = [
            status
            for status, _ in sorted(
                status_updates_with_importance, key=lambda x: x[1], reverse=True
            )
        ]

        # --- Send Citation ---
        if self.valves.show_status and __event_emitter__ and sorted_status_updates:
            # Send citation only if there were operations attempted
            await self._send_citation(
                __event_emitter__, sorted_status_updates, __user__.get("id")
            )  # Fix NameError

    async def _create_memory(self, content: str, __user__: dict) -> Optional[str]:
        """
        Create a new memory in the database.

        Args:
            content: The content of the memory
            __user__: The user dictionary object

        Returns:
            The ID of the created memory, or None if creation failed
        """
        try:
            # Fix TypeError: remove unexpected timestamp argument
            memory = Memories.insert_new_memory(
                user_id=__user__.get("id"), content=content
            )
            if memory and hasattr(memory, "id"):
                logger.info("Successfully created memory %s", memory.id)
                return memory.id
            else:
                logger.warning("Memory creation returned None or object without ID")
                return None
        except Exception as e:
            logger.error(f"Error creating memory: {e}", exc_info=True)
            return None

    # --------------------------------------------------------------------------
    # Event Emitter Helpers
    # --------------------------------------------------------------------------

    async def _safe_emit(
        self,
        emitter: Callable[
            [Dict[str, Any]], Awaitable[None]
        ],  # Emitter expects one Dict arg
        data: Dict[
            str, Any
        ],  # Data dict includes the 'type' (e.g., "status", "citation")
    ) -> None:
        """Safely emit events using the provided emitter, which expects a single data dictionary."""
        try:
            await emitter(data)  # Pass the single data dictionary
        except Exception as e:
            event_type = data.get("type", "UNKNOWN")  # Get type from data for logging
            logger.error(
                f"Failed to emit event type '{event_type}': {e}", exc_info=True
            )

    def _non_blocking_emit(
        self,
        emitter: Callable[
            [Dict[str, Any]], Awaitable[None]
        ],  # Emitter expects one Dict arg
        data: Dict[str, Any],  # Data dict includes the 'type'
    ) -> None:
        """Emit events in a non-blocking way using the corrected _safe_emit."""
        # Ensure emitter is valid before creating task
        if emitter:
            asyncio.create_task(
                self._safe_emit(emitter, data)
            )  # Pass only emitter and data
        else:
            event_type = data.get("type", "UNKNOWN")
            logger.debug(
                f"Emitter not available, cannot emit event type '{event_type}'"
            )

    async def _send_citation(
        self,
        emitter: Callable[..., Awaitable[None]],
        status_updates: List[str],
        user_id: Optional[str],  # Can be None if ID not found in __user__ dict
    ) -> None:
        """Send a citation message with the summary of memory operations."""
        if not status_updates or not emitter:
            return

        # Format the citation message
        citation_message = "Memories Saved (sorted by importance):\n" + "\n".join(
            f"- {update}" for update in status_updates
        )
        logger.info(
            "Sending citation for user %s:\n%s", user_id or "UNKNOWN", citation_message
        )

        # Use the EXACT same data structure as the working MMC module
        await self._safe_emit(
            emitter,
            {
                "type": "citation",
                "data": {
                    "document": [citation_message],  # Content goes in document array
                    "metadata": [
                        {"source": "module://mis/memories", "html": False}
                    ],  # Standard metadata format
                    "source": {
                        "name": "Memories Saved"
                    },  # Title for the citation (matching backup file)
                },
            },
        )

    # --------------------------------------------------------------------------
    # Inlet (Main Entry Point)
    # --------------------------------------------------------------------------

    async def inlet(
        self,
        body: Dict[str, Any],
        __user__: dict = {},  # Changed parameter name and added default
        __event_emitter__: Optional[Callable[..., Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for processing incoming messages.
        Identifies and stores memories from user messages.

        Args:
            body: The request body containing messages and other data
            __user__: The user dictionary object
            __event_emitter__: Optional event emitter for sending status updates

        Returns:
            The original body (unmodified).
        """

        # Check if identification prompt is set
        if not self.valves.identification_prompt:
            logger.error("Identification prompt is not set. Cannot process memories.")
            if __event_emitter__ and self.valves.show_status:
                self._non_blocking_emit(
                    __event_emitter__,
                    {
                        "type": "status",
                        "data": {
                            "description": "⚠️ MIS Error: Identification prompt not configured",
                            "done": True,
                        },
                    },
                )
            return body

        logger.info("Received message for memory identification and storage")

        # Extract the relevant user message(s)
        messages = body.get("messages", [])
        user_messages = [
            msg["content"] for msg in messages if msg.get("role") == "user"
        ]

        if not user_messages:
            logger.info("No user messages found in the body.")
            return body

        # Determine the message content to analyze based on settings
        current_message_content: str
        if self.valves.recent_messages_count > 1:
            start_index = max(0, len(user_messages) - self.valves.recent_messages_count)
            context_messages = user_messages[start_index:]
            current_message_content = "\n".join(context_messages)
            logger.info("Using last %d messages for context.", len(context_messages))
        else:
            current_message_content = user_messages[-1]
            logger.info("Using the last message for context.")

        # Process memories
        logger.info("Processing memories")
        start_sync_time = time.time()

        try:
            # Send initial "Processing..." status if enabled
            if self.valves.show_status and __event_emitter__:
                # Pass single data dict to _non_blocking_emit with correct format
                self._non_blocking_emit(
                    __event_emitter__,
                    {
                        "type": "status",
                        "data": {
                            "description": "⏳ Identifying memories...",  # Use description instead of message
                            "done": False,  # Not done yet
                        },
                    },
                )

            # First: Identify potential memories and prepare NEW operations
            # Note: Removed db_memories fetch as it's no longer needed for integration
            memory_operations = await self._identify_and_prepare_memories(
                current_message_content
            )

            # Then: Execute the creation operations
            if memory_operations:
                await self._execute_memory_creation(
                    memory_operations, __user__, __event_emitter__
                )
            else:
                logger.info("No memory operations to execute.")
                # Clear the "Processing..." status if nothing was done and status is shown
                if self.valves.show_status and __event_emitter__:
                    self._non_blocking_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": "",  # Empty description to clear
                                "done": True,  # Mark as done
                            },
                        },
                    )

        except Exception as e:
            logger.error(f"Error during memory processing: {e}", exc_info=True)
            # Send error status if enabled
            if __event_emitter__ and self.valves.show_status:
                error_message = f"⚠️ Error identifying/storing memories: {e}"
                self._non_blocking_emit(
                    __event_emitter__,
                    {
                        "type": "status",
                        "data": {
                            "description": error_message,  # Error message
                            "done": True,  # Mark as done
                        },
                    },
                )

        finally:
            end_sync_time = time.time()
            duration = end_sync_time - start_sync_time
            logger.info("Memory processing finished in %.2f seconds", duration)

            # Always clear the status when processing is complete
            if self.valves.show_status and __event_emitter__:
                self._non_blocking_emit(
                    __event_emitter__,
                    {
                        "type": "status",
                        "data": {
                            "description": "",  # Empty description to clear
                            "done": True,  # Mark as done
                        },
                    },
                )

        # Return the original body - MIS does not modify the chat flow directly
        logger.info("Memory identification/storage complete. Returning original body.")
        return body

    async def outlet(
        self, body: Dict[str, Any], __user__: dict = {}
    ) -> Dict[str, Any]:  # Changed parameter name and added default
        """
        Outlet function - currently does nothing as MIS doesn't modify outgoing messages.

        Args:
            body: The request body
            __user__: The user dictionary object

        Returns:
            The original body
        """
        # logger.debug("MIS outlet called, returning body unmodified.")
        return body
