"""
title: AMM_Memory_Management_Consolidation
description: Memory Management & Consolidation Module for Open WebUI - Consolidates and organizes user memories (async-only)
author: Cody
version: 0.3.3
date: 2025-04-21
changes:
- Standardized citation format
- Implemented smart content matching for complex memories (v0.3.2)
- Added force_delete_by_id valve parameter for ID-based fallback (v0.3.2)
- Enhanced memory content parsing to handle different formats (v0.3.2)
- Added key phrase extraction for long text comparison (v0.3.2)
- Implemented fuzzy matching for memory content comparison (v0.3.1)
- Added configurable similarity threshold valve parameter (v0.3.1)
- Added text normalization for more reliable memory matching (v0.3.1)
- Enhanced logging for memory comparison diagnostics (v0.3.1)
- Updated JSON parser to handle new LLM output format (v0.3.0)
- Replaced delete-then-create pattern with direct database updates (v0.3.0)
- Added intelligent memory processing with category-based grouping (v0.3.0)
- Enhanced citation reporting with category organization (v0.3.0)
- Aligned with MIS module's database handling approach (v0.3.0)
- Moved trigger phrases to a valve (v0.2.2)
- Removed various legacy code and general clean-up (v0.2.2)
- Initial implementation of Memory Management & Consolidation module (v0.2.0)
- Implemented memory consolidation logic using LLM (v0.2.0)
- Added support for user-triggered memory organization (v0.2.0)
- Removed sync processing path and valve toggle (v0.2.0)
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple

import aiohttp
from open_webui.models.memories import Memories
from open_webui.models.users import Users
from pydantic import BaseModel, Field

# Logger configuration
logger = logging.getLogger("amm_mmc")
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
    """Memory Management & Consolidation module for consolidating and organizing user memories."""

    class Valves(BaseModel):
        # Set processing priority
        priority: int = Field(
            default=4,  # Lower priority than MIS (3) and MRE (2)
            description="Priority level for the filter operations.",
        )
        # UI settings
        show_status: bool = Field(
            default=True,
            description="Show memory consolidation status in chat",
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
        # Trigger phrases configuration
        consolidation_trigger_phrases: List[str] = Field(
            default=[
                "organize memories",
                "organize my memories"
            ],
            description="Phrases that trigger memory consolidation when found in user messages",
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
            default=1000,
            description="Maximum tokens for API calls",
        )
        # Consolidation settings
        similarity_threshold: float = Field(
            default=0.85,  # 85% similarity by default
            description="Threshold for fuzzy matching memory content (0.0-1.0)",
        )
        force_delete_by_id: bool = Field(
            default=False,
            description="Force delete memories by ID when content matching fails",
        )
        # Prompt storage with no default value (required field)
        consolidation_prompt: str = Field(
            ...,  # This makes the field required with no default
            description="Memory consolidation prompt with preserved formatting",
        )

    def __init__(self) -> None:
        """Initialize the Memory Management & Consolidation module for async processing."""
        logger.info("Initializing Memory Management & Consolidation module (async-only)")

        # Initialize with empty prompt - must be set via update_valves
        try:
            self.valves = self.Valves(
                consolidation_prompt=""  # Empty string to start - must be set via update_valves
            )
            logger.warning(
                "Consolidation prompt is empty - module will not function until prompt is set"
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

        self.background_tasks = set()  # Track background tasks
        logger.info(
            "MMC module initialized with API provider: %s", self.valves.api_provider
        )

    async def close(self) -> None:
        """Close the aiohttp session and cancel any running background tasks."""
        logger.info("Closing MMC module session")

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
            formatted.append(f"[ID: {mem_id}] {content}")

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
            
        lines = text.split('\n')
        if len(lines) <= max_lines:
            return text
            
        truncated = lines[:max_lines]
        omitted = len(lines) - max_lines
        truncated.append(f"... [truncated, {omitted} more lines omitted]")
        return '\n'.join(truncated)

    def _is_consolidation_request(self, message: str) -> bool:
        """
        Determine if a message is requesting memory consolidation.
        
        Args:
            message: The user message
            
        Returns:
            True if the message is requesting consolidation, False otherwise
        """
        # Use trigger phrases from valve parameter
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in self.valves.consolidation_trigger_phrases)

    async def _consolidate_memories(
        self, 
        user_message: str,
        db_memories: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Consolidate memories using LLM.
        
        Args:
            user_message: The user message requesting consolidation
            db_memories: List of memories from the database
            
        Returns:
            List of memory operations (NEW/UPDATE/DELETE)
        """
        # Essential logging (always shown)
        logger.info("Consolidating memories based on user request")
        
        # Format existing memories
        formatted_memories = self._format_memories(db_memories)
        
        # Verbose logging (only when verbose logging is enabled)
        if self.valves.verbose_logging:
            truncated_memories = self._truncate_log_lines(formatted_memories)
            logger.info("Formatted existing memories for LLM:\n%s", truncated_memories)
        
        # Check if prompt is empty and fail fast
        if not self.valves.consolidation_prompt:
            logger.error("Consolidation prompt is empty - cannot process memories")
            raise ValueError("Consolidation prompt is empty - module cannot function")
        
        # Format the system prompt using the valve-stored prompt
        system_prompt = self.valves.consolidation_prompt.format(
            user_message=user_message,
            existing_memories=formatted_memories
        )
        
        # Verbose logging (only when verbose logging is enabled)
        if self.valves.verbose_logging:
            # Log a truncated version of the system prompt
            truncated_prompt = self._truncate_log_lines(system_prompt)
            logger.info("System prompt for consolidation (truncated):\n%s", truncated_prompt)
        
        # Query the appropriate API
        if self.valves.api_provider == "OpenAI API":
            response = await self.query_openai_api(system_prompt, user_message)
        else:  # Ollama API
            response = await self.query_ollama_api(system_prompt, user_message)
        
        # Verbose logging (only when verbose logging is enabled)
        if self.valves.verbose_logging:
            truncated_response = self._truncate_log_lines(response)
            logger.info("Consolidation: Raw API response: %s", truncated_response)
        
        # Parse the response
        memory_operations = self._parse_json_response(response)
        
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
                # Log details of each operation with reasoning
                for op in memory_operations:
                    operation_type = op.get("operation", "UNKNOWN")
                    content = op.get("content", "")
                    mem_id = op.get("id", "N/A") if operation_type != "NEW" else "new"
                    reasoning = op.get("reasoning", "No reasoning provided")
                    
                    logger.info(
                        "%s memory [id: %s]: %s",
                        operation_type,
                        mem_id,
                        content,
                    )
                    logger.info("Reasoning: %s", reasoning)
        else:
            # Essential logging (always shown)
            logger.info("No memory operations determined")
        
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

    # --- Helper functions for memory content handling ---
    
    def _get_user_id(self, user: Any) -> str:
        """Extract user ID from either a dictionary or object representation."""
        # Ensure user is not None and has an 'id' attribute or key
        if user:
            if isinstance(user, dict):
                return user.get("id")
            elif hasattr(user, "id"):
                return str(user.id)
        logger.error("Could not extract user ID from provided user object/dict.")
        raise ValueError("Invalid user object provided")

    def _clean_memory_id(self, memory_id: str) -> Optional[str]:
        """Clean memory ID consistently from various formats."""
        if not memory_id or not isinstance(memory_id, str):
            logger.warning(f"Received invalid memory_id for cleaning: {memory_id}")
            return None

        # Use a simple regex to extract UUID pattern if embedded in text
        uuid_pattern = re.compile(
            r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
            re.IGNORECASE,
        )
        match = uuid_pattern.search(memory_id)
        if match:
            cleaned_id = match.group(1)
            if self.valves.verbose_logging and cleaned_id != memory_id:
                 logger.debug(f"Cleaned memory ID from '{memory_id}' to '{cleaned_id}'")
            return cleaned_id

        # If no UUID pattern found, return the original string if it looks like a UUID
        # Basic check to avoid returning arbitrary strings
        if len(memory_id) == 36 and memory_id.count('-') == 4:
             return memory_id

        logger.warning(f"Could not clean or validate memory_id: {memory_id}")
        return None # Return None if cleaning fails or doesn't look like UUID

    def _validate_memory(self, memory_id: str, user_id: str) -> Optional[Any]:
        """Validate memory existence and ownership. Returns the memory object if valid."""
        if not memory_id or not user_id:
             logger.warning("Validation skipped: Missing memory_id or user_id.")
             return None

        cleaned_memory_id = self._clean_memory_id(memory_id)
        if not cleaned_memory_id:
             logger.warning(f"Validation failed: Could not clean memory_id '{memory_id}'.")
             return None

        if self.valves.verbose_logging:
            logger.debug(f"Validating memory {cleaned_memory_id} for user {user_id}")

        try:
            memory = Memories.get_memory_by_id(cleaned_memory_id)
            if not memory:
                logger.warning(f"Validation failed: Memory {cleaned_memory_id} not found.")
                return None

            if not hasattr(memory, "user_id"):
                 logger.error(f"Validation failed: Memory {cleaned_memory_id} object lacks 'user_id' attribute.")
                 return None

            if str(memory.user_id) != str(user_id):
                logger.warning(
                    f"Validation failed: Memory {cleaned_memory_id} belongs to user {memory.user_id}, not {user_id}."
                )
                return None

            if self.valves.verbose_logging:
                logger.debug(f"Memory {cleaned_memory_id} validation successful.")
            return memory # Return the validated memory object

        except Exception as e:
            logger.error(f"Exception during memory validation for {cleaned_memory_id}: {e}", exc_info=True)
            return None

    def _parse_memory_content(self, content: str) -> Tuple[str, List[str]]:
        """
        Parse memory content string into category and a list of bullet items.
        Enhanced to handle different memory formats including:
        - Standard category with bullet points
        - Standalone memories without category headers
        - Single-line memories
        - Multi-line memories without bullet points

        Returns:
            tuple: (category_name, list_of_items)
        """
        if not content:
            return "Miscellaneous", []

        content = content.strip()
        
        # Handle standalone memories without category headers and no bullet points
        if '\n' not in content and not content.endswith(':'):
            # This is a single-line memory without a category
            return "Miscellaneous", [content]
            
        lines = content.split('\n')
        if not lines:
            return "Miscellaneous", []

        # Extract category (first line, remove trailing colon if present)
        category = lines[0].strip().removesuffix(':').strip()
        if not category:
            category = "Miscellaneous" # Default if first line was just ":" or empty
            
        # Check if this is a category header followed by non-bullet content
        if len(lines) > 1 and not any(line.strip().startswith('- ') for line in lines[1:]):
            # This is a category with non-bullet content
            # Treat each line after the category as a separate item
            items = [line.strip() for line in lines[1:] if line.strip()]
            return category, items

        # Standard case: Extract items (lines starting with '- ')
        items = []
        for line in lines[1:]:
            stripped_line = line.strip()
            if stripped_line.startswith('- '):
                item_text = stripped_line[2:].strip()
                if item_text: # Avoid adding empty items
                    items.append(item_text)
            elif stripped_line and self.valves.verbose_logging:
                # Log non-bullet lines only when verbose logging is enabled
                logger.debug(f"Non-bullet line in memory content: '{stripped_line}'")
                
        # If we found a category but no items, and there are non-empty lines after the category,
        # treat the entire content after the category as one item
        if not items and len(lines) > 1:
            combined_content = '\n'.join(lines[1:]).strip()
            if combined_content:
                items = [combined_content]

        return category, items

    def _format_memory(self, category: str, items: List[str]) -> str:
        """
        Format a memory block string with consistent structure.

        Args:
            category: The memory category name.
            items: List of memory items (bullet points text).

        Returns:
            str: Formatted memory content string.
        """
        if not category:
            category = "Miscellaneous"

        # Ensure category ends with a colon for the header
        header = f"{category.strip().removesuffix(':')}:"

        # Format items, ensuring they start with '- '
        formatted_items = []
        for item in items:
            clean_item = item.strip()
            if clean_item: # Skip empty items
                 # Ensure item doesn't already start with '- '
                 if clean_item.startswith('- '):
                      formatted_items.append(clean_item)
                 else:
                      formatted_items.append(f"- {clean_item}")

        # Combine header and items
        if not formatted_items:
             return header # Return just the header if no items

        return f"{header}\n" + "\n".join(formatted_items)
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for more reliable comparison.
        
        Args:
            text: The text to normalize
            
        Returns:
            Normalized text string
        """
        if not text:
            return ""
            
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove common prefixes like "User"
        normalized = re.sub(r'^user\s+', '', normalized)
        
        # Standardize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Trim whitespace
        normalized = normalized.strip()
        
        return normalized
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Normalize both texts
        norm_text1 = self._normalize_text(text1)
        norm_text2 = self._normalize_text(text2)
        
        # If either string is empty after normalization, return 0
        if not norm_text1 or not norm_text2:
            return 0.0
            
        # Calculate similarity using SequenceMatcher
        return SequenceMatcher(None, norm_text1, norm_text2).ratio()
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """
        Extract key phrases from a long text for more effective matching.
        
        Args:
            text: The text to extract key phrases from
            
        Returns:
            List of key phrases
        """
        if not text:
            return []
            
        # Normalize the text
        normalized = self._normalize_text(text)
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', normalized)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # For very short text, return the whole text as a single phrase
        if len(normalized) < 50:
            return [normalized]
            
        phrases = []
        
        # Extract key phrases from each sentence
        for sentence in sentences:
            # Skip very short sentences
            if len(sentence) < 5:
                continue
                
            words = sentence.split()
            
            # For short sentences, use the whole sentence
            if len(words) <= 7:
                phrases.append(sentence)
            else:
                # For longer sentences, extract the first 7 words as a key phrase
                phrases.append(' '.join(words[:7]))
                
                # If the sentence is very long, also extract the last 7 words
                if len(words) > 14:
                    phrases.append(' '.join(words[-7:]))
        
        return phrases
    
    def _smart_content_match(self, delete_item: str, existing_item: str) -> float:
        """
        Smart content matching that handles different text lengths and formats.
        
        Args:
            delete_item: The item to delete
            existing_item: The existing item to compare against
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # For short texts, use direct similarity
        if len(delete_item) < 100 and len(existing_item) < 100:
            return self._calculate_similarity(delete_item, existing_item)
            
        # For longer texts, use key phrase matching
        delete_phrases = self._extract_key_phrases(delete_item)
        existing_phrases = self._extract_key_phrases(existing_item)
        
        if not delete_phrases or not existing_phrases:
            return 0.0
            
        # Calculate the best match score for each delete phrase
        phrase_scores = []
        for dp in delete_phrases:
            best_score = 0.0
            for ep in existing_phrases:
                score = self._calculate_similarity(dp, ep)
                best_score = max(best_score, score)
            phrase_scores.append(best_score)
            
        # Average the phrase scores
        if phrase_scores:
            avg_score = sum(phrase_scores) / len(phrase_scores)
            
            # Verbose logging
            if self.valves.verbose_logging:
                logger.debug(f"Smart content match score: {avg_score:.2f} for phrases: {delete_phrases} vs {existing_phrases}")
                
            return avg_score
        
        return 0.0
    
    # --- End helper functions ---

    def _parse_json_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse the JSON response from the LLM for consolidation operations.
        Maps LLM intents ('add', 'delete', 'negate') to internal operations ('NEW', 'UPDATE', 'DELETE').

        Args:
            response_text: The response text from the LLM

        Returns:
            List of parsed objects (memory operations)
        """
        if not response_text or not response_text.strip():
            logger.debug("Received empty response text for parsing.")
            return []

        try:
            # Clean potential markdown code blocks
            cleaned = re.sub(r"```json|```", "", response_text.strip())
            if not cleaned:
                 logger.warning("Response text became empty after cleaning markdown.")
                 return []

            data = json.loads(cleaned)

            if not isinstance(data, list):
                logger.error(f"LLM response is not a JSON array, but {type(data)}")
                if self.valves.verbose_logging:
                    logger.error(f"Invalid JSON content: {cleaned}")
                return []

            valid_items = []
            for item in data:
                if not isinstance(item, dict):
                    logger.warning(f"Skipping non-dictionary item in response array: {item}")
                    continue

                intent = item.get("intent")
                sub_memory = item.get("sub_memory")
                category = item.get("category")
                memory_id = item.get("memory_id") # LLM provides memory_id

                if not intent or not sub_memory or not category:
                    logger.warning(f"Skipping item with missing fields (intent, sub_memory, category): {item}")
                    continue

                operation = None
                if intent == "add":
                    operation = "NEW" # Code logic will decide if this becomes CREATE or UPDATE
                elif intent in ["delete", "negate"]:
                    operation = "DELETE" # Code logic will decide if this becomes UPDATE or DELETE block
                    if not memory_id:
                        logger.warning(f"Skipping '{intent}' intent missing 'memory_id': {item}")
                        continue
                else:
                    logger.warning(f"Skipping item with unknown intent '{intent}': {item}")
                    continue

                # Standardize internal representation
                parsed_op = {
                    "operation": operation,
                    "content": sub_memory, # Use 'content' internally for the item text
                    "category": category,
                }
                if memory_id:
                    # Clean and store as 'id' internally
                    parsed_op["id"] = self._clean_memory_id(memory_id)

                valid_items.append(parsed_op)

            if self.valves.verbose_logging:
                 logger.info(f"Parsed {len(valid_items)} valid operations from LLM response.")
                 logger.debug(f"Parsed operations: {json.dumps(valid_items, indent=2)}")

            return valid_items

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            logger.error(f"Failed JSON content: {response_text}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during JSON parsing: {e}", exc_info=True)
            return []

    async def _process_memory_operations(
        self, operations: List[Dict[str, Any]], user: Any
    ) -> List[Dict[str, Any]]:
        """
        Process memory operations intelligently, translating LLM intents into appropriate database actions.
        
        This function:
        1. Groups operations by category
        2. For "add" intents: Updates existing memories or creates new ones
        3. For "delete" intents: Removes items from memories or deletes entire memories
        
        Args:
            operations: List of memory operations from LLM
            user: The user object

        Returns:
            List of processed operations with status
        """
        # Add logging for operations being processed
        logger.info(f"Processing {len(operations)} memory operations: {[op.get('operation') for op in operations]}")
        results = []
        
        # Get user ID for database operations
        try:
            user_id = self._get_user_id(user)
        except ValueError as e:
            logger.error(f"Failed to get user ID: {e}")
            return []
            
        # Fetch all existing memories for this user once for efficient lookup
        db_memories = Memories.get_memories_by_user_id(user_id)
        if not db_memories:
            logger.info("No existing memories found for user")
            db_memories = []
            
        # Create a mapping of categories to memory objects for quick lookup
        category_to_memory_map = {}
        for mem in db_memories:
            if hasattr(mem, "id") and hasattr(mem, "content") and mem.content:
                category, _ = self._parse_memory_content(mem.content)
                if category:
                    # Store lowercase category for case-insensitive matching
                    category_to_memory_map[category.lower()] = mem
        
        # Group operations by category and operation type
        add_operations_by_category = {}  # {category: [content1, content2, ...]}
        delete_operations_by_memory = {}  # {memory_id: [content1, content2, ...]}
        
        # First pass: group operations
        for op in operations:
            operation = op.get("operation", "UNKNOWN")
            content = op.get("content", "")
            category = op.get("category", "Miscellaneous")
            memory_id = op.get("id")
            
            if not content or not category:
                logger.warning(f"Skipping operation with missing content or category: {op}")
                continue
                
            if operation == "NEW":  # Add intent
                # Initialize category list if needed
                if category.lower() not in add_operations_by_category:
                    add_operations_by_category[category.lower()] = []
                # Add content to the category list
                add_operations_by_category[category.lower()].append({
                    "content": content,
                    "category": category  # Preserve original case
                })
                
            elif operation == "DELETE":  # Delete/negate intent
                if not memory_id:
                    logger.warning(f"Skipping delete operation with missing memory_id: {op}")
                    continue
                    
                # Initialize memory list if needed
                if memory_id not in delete_operations_by_memory:
                    delete_operations_by_memory[memory_id] = []
                # Add content to the memory list
                delete_operations_by_memory[memory_id].append(content)
        
        # Second pass: process grouped operations
        
        # Process add operations by category
        for category_lower, items in add_operations_by_category.items():
            if not items:
                continue
                
            # Check if we have an existing memory with this category
            if category_lower in category_to_memory_map:
                # UPDATE existing memory by adding new items
                existing_memory = category_to_memory_map[category_lower]
                memory_id = existing_memory.id
                
                # Parse existing content
                category_name, existing_items = self._parse_memory_content(existing_memory.content)
                
                # Add new items, avoiding duplicates
                new_items = []
                for item_data in items:
                    item_content = item_data["content"]
                    if item_content not in existing_items:
                        new_items.append(item_content)
                    else:
                        logger.info(f"Skipping duplicate item: {item_content}")
                
                if not new_items:
                    logger.info(f"No new items to add to category '{category_name}'")
                    continue
                    
                # Combine existing and new items
                updated_items = existing_items + new_items
                
                # Format the updated memory content
                updated_content = self._format_memory(category_name, updated_items)
                
                # Update the memory
                success = await self._update_memory(memory_id, updated_content, user)
                
                # Record the result
                results.append({
                    "operation": "UPDATE",
                    "content": updated_content,
                    "id": memory_id,
                    "success": success,
                    "status": f"Added {len(new_items)} items" if success else "Failed to update",
                    "category_name": category_name,
                    "affected_items": new_items
                })
                
            else:
                # CREATE new memory for this category
                # Use the original case of the first item's category
                category_name = items[0]["category"]
                
                # Extract all item contents
                item_contents = [item_data["content"] for item_data in items]
                
                # Format the new memory content
                new_content = self._format_memory(category_name, item_contents)
                
                # Create the memory
                memory_id = await self._create_memory(new_content, user)
                
                # Record the result
                results.append({
                    "operation": "NEW",
                    "content": new_content,
                    "id": memory_id,
                    "success": bool(memory_id),
                    "status": f"Created with {len(item_contents)} items" if memory_id else "Failed to create",
                    "category_name": category_name,
                    "affected_items": item_contents
                })
        
        # Process delete operations by memory
        for memory_id, items_to_delete in delete_operations_by_memory.items():
            if not items_to_delete:
                continue
                
            # Validate memory exists and belongs to user
            memory = self._validate_memory(memory_id, user_id)
            if not memory:
                logger.warning(f"Cannot delete from invalid memory {memory_id}")
                results.append({
                    "operation": "DELETE",
                    "id": memory_id,
                    "success": False,
                    "status": "Failed: Memory not found or not owned by user",
                    "affected_items": items_to_delete
                })
                continue
                
            # Parse memory content
            category_name, existing_items = self._parse_memory_content(memory.content)
            
            # Find items to remove (smart content matching)
            items_to_keep = []
            removed_items = []
            
            # Track best matches for verbose logging
            best_matches = {}
            
            for existing_item in existing_items:
                should_keep = True
                for delete_item in items_to_delete:
                    # Use smart content matching
                    similarity_score = self._smart_content_match(delete_item, existing_item)
                    
                    # Track best match for this delete_item for potential logging
                    if delete_item not in best_matches or similarity_score > best_matches[delete_item]["score"]:
                        best_matches[delete_item] = {
                            "item": existing_item,
                            "score": similarity_score
                        }
                    
                    # Verbose logging of similarity scores
                    if self.valves.verbose_logging:
                        logger.debug(f"Smart similarity between '{delete_item}' and '{existing_item}': {similarity_score:.2f}")
                    
                    # Check if similarity meets threshold
                    if similarity_score >= self.valves.similarity_threshold:
                        should_keep = False
                        removed_items.append(existing_item)
                        
                        if self.valves.verbose_logging:
                            logger.debug(f"Match found: '{delete_item}' matches '{existing_item}' with score {similarity_score:.2f}")
                        break
                
                if should_keep:
                    items_to_keep.append(existing_item)
            
            if len(removed_items) == 0:
                # Log near-misses for debugging when verbose logging is enabled
                if self.valves.verbose_logging:
                    for delete_item, match_info in best_matches.items():
                        logger.debug(f"Best match for '{delete_item}': '{match_info['item']}' with score {match_info['score']:.2f}")
                
                # Check if we should force delete by ID
                if self.valves.force_delete_by_id:
                    logger.info(f"Forcing deletion of memory {memory_id} by ID")
                    success = await self._delete_memory(memory_id, user)
                    results.append({
                        "operation": "DELETE",
                        "id": memory_id,
                        "success": success,
                        "status": "Forced deletion by ID" if success else "Failed to force delete",
                        "category_name": category_name,
                        "affected_items": items_to_delete
                    })
                    continue
                else:
                    logger.info(f"No matching items found to delete in memory {memory_id}")
                    results.append({
                        "operation": "DELETE",
                        "id": memory_id,
                        "success": False,
                        "status": "Failed: No matching items found",
                        "category_name": category_name,
                        "affected_items": items_to_delete
                    })
                    continue
                
            if len(items_to_keep) == 0:
                # DELETE the entire memory if no items left
                logger.info(f"Deleting entire memory {memory_id} as all items were removed")
                success = await self._delete_memory(memory_id, user)
                results.append({
                    "operation": "DELETE",
                    "id": memory_id,
                    "success": success,
                    "status": "Deleted entire memory" if success else "Failed to delete memory",
                    "category_name": category_name,
                    "affected_items": removed_items
                })
            else:
                # UPDATE the memory with remaining items
                updated_content = self._format_memory(category_name, items_to_keep)
                success = await self._update_memory(memory_id, updated_content, user)
                results.append({
                    "operation": "UPDATE",
                    "content": updated_content,
                    "id": memory_id,
                    "success": success,
                    "status": f"Removed {len(removed_items)} items" if success else "Failed to update",
                    "category_name": category_name,
                    "affected_items": removed_items
                })
        
        # Add summary logging for operation results
        successful_ops = sum(1 for op in results if op.get('success', False))
        logger.info(f"Completed {successful_ops}/{len(results)} memory operations successfully")
        
        # Log details of failed operations
        for op in results:
            if not op.get('success', False):
                logger.error(f"Failed {op.get('operation')} operation: {op.get('status')}")
                
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
                # Log ID only when verbose
                if self.valves.verbose_logging:
                    logger.info("Created memory with ID: %s", memory_id)
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
        Update an existing memory directly using Memories.update_memory_by_id_and_user_id.

        Args:
            memory_id: The ID of the memory to update.
            content: The new, complete content for the memory block.
            user: The user object or dictionary.

        Returns:
            True if successful, False otherwise.
        """
        try:
            user_id = self._get_user_id(user)
            cleaned_memory_id = self._clean_memory_id(memory_id)

            if not cleaned_memory_id:
                 logger.error(f"Update failed: Invalid memory_id '{memory_id}'.")
                 return False

            # Validate memory exists and belongs to user before attempting update
            if not self._validate_memory(cleaned_memory_id, user_id):
                 # Validation logs the specific reason (not found or wrong user)
                 return False

            # Assume 'content' is the complete, properly formatted new content
            if self.valves.verbose_logging:
                 logger.debug(f"Attempting to update memory {cleaned_memory_id} with content:\n{content}")

            # Use the direct update method from Memories model
            updated_memory = Memories.update_memory_by_id_and_user_id(
                id=cleaned_memory_id, user_id=user_id, content=content
            )

            if updated_memory and hasattr(updated_memory, "id"):
                logger.info(f"Successfully updated memory {cleaned_memory_id}")
                return True
            else:
                # This case might indicate the update method returned None or an unexpected object
                logger.error(f"Failed to update memory {cleaned_memory_id}. DB method returned: {updated_memory}")
                return False

        except Exception as e:
            logger.error(f"Error during memory update for {memory_id}: {e}", exc_info=True)
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
            # Log ID only when verbose
            if self.valves.verbose_logging:
                logger.info("Deleted memory with ID: %s", memory_id)
                if hasattr(memory, "content"):
                    logger.info("Deleted content: %s", memory.content)
            return True
        except Exception as e:
            logger.error("Error deleting memory %s: %s", memory_id, e)
            return False

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
        delay_seconds: float = 1.0,
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

    async def _process_consolidation_background(
        self,
        user_message: str,
        user: Any,
        user_id: str,
        event_emitter: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> None:
        """
        Process memory consolidation in the background.
        
        Args:
            user_message: The user message requesting consolidation
            user: The user object
            user_id: The user ID
            event_emitter: Optional event emitter for notifications
        """
        try:
            # Log background task start with timestamp
            start_time = time.time()
            logger.info("Background memory consolidation started")

            # Short delay and re-emit status to potentially override faster module clears
            if self.valves.show_status and event_emitter:
                await asyncio.sleep(2) # Small delay (0.5 seconds)
                self._non_blocking_emit(
                    event_emitter,
                    {
                        "type": "status",
                        "data": {
                            "description": "🧠 Consolidating memories...",
                            "done": False,
                        },
                    },
                )

            # Fetch all memories
            db_memories = Memories.get_memories_by_user_id(user_id)
            if not db_memories:
                # No memories to consolidate
                if self.valves.show_status and event_emitter:
                    self._non_blocking_emit(
                        event_emitter,
                        {
                            "type": "status",
                            "data": {
                                "description": "☑ No memories to consolidate",
                                "done": True,
                            },
                        },
                    )
                    
                    # Status will be cleared after the final message or completion.
                
                return
            
            # Process consolidation
            memory_operations = await self._consolidate_memories(
                user_message, db_memories
            )
            
            # If no memory operations, emit completion status and return
            if not memory_operations:
                if self.valves.show_status and event_emitter:
                    self._non_blocking_emit(
                        event_emitter,
                        {
                            "type": "status",
                            "data": {
                                "description": "☑ No memory consolidation needed",
                                "done": True,
                            },
                        },
                    )
                    
                    # Status will be cleared after the final message or completion.
                
                return
            
            # Process the memory operations
            processed_operations = await self._process_memory_operations(
                memory_operations, user
            )
            
            # Count operations by type
            new_count = 0
            update_count = 0
            delete_count = 0
            
            for op in processed_operations:
                if op.get("success", False):
                    if op.get("operation") == "NEW":
                        new_count += 1
                    elif op.get("operation") == "UPDATE":
                        update_count += 1
                    elif op.get("operation") == "DELETE":
                        delete_count += 1
            
            # Create citation content for memory operations
            if processed_operations and event_emitter:
                # Group operations by category for clearer reporting
                operations_by_category = {}
                
                for op in processed_operations:
                    if not op.get("success", False):
                        continue  # Skip failed operations in citation
                        
                    category = op.get("category_name", "Miscellaneous")
                    if category not in operations_by_category:
                        operations_by_category[category] = []
                    operations_by_category[category].append(op)
                
                # Format citation content by category
                citation_content = ""
                
                # Process each category once, combining operations of the same type
                for category, ops in operations_by_category.items():
                    # Group operations by type within this category
                    new_ops = [op for op in ops if op.get("operation") == "NEW"]
                    update_add_ops = [op for op in ops if op.get("operation") == "UPDATE" and "Added" in op.get("status", "")]
                    update_remove_ops = [op for op in ops if op.get("operation") == "UPDATE" and "Removed" in op.get("status", "")]
                    delete_ops = [op for op in ops if op.get("operation") == "DELETE"]
                    
                    # Add category with appropriate action description
                    if new_ops:
                        citation_content += f"{category}: (Added info to category)\n"
                        # Show the items that were added
                        for op in new_ops:
                            for item in op.get("affected_items", []):
                                citation_content += f"✅ - {item}\n"
                        citation_content += "\n"
                    
                    if update_add_ops:
                        citation_content += f"{category}: (Added info to category)\n"
                        # Show the items that were added
                        for op in update_add_ops:
                            for item in op.get("affected_items", []):
                                citation_content += f"✅ - {item}\n"
                        citation_content += "\n"
                    
                    if update_remove_ops:
                        citation_content += f"{category}: (Removed info from category)\n"
                        # Show the items that were removed
                        for op in update_remove_ops:
                            for item in op.get("affected_items", []):
                                citation_content += f"❎ - {item}\n"
                        citation_content += "\n"
                    
                    if delete_ops:
                        # Check if this is a standalone memory (not in a category)
                        if category == "Miscellaneous" or category == "Unknown":
                            citation_content += f"No category: (Deleted info)\n"
                        else:
                            citation_content += f"{category}: (Deleted info)\n"
                        # Show the items that were deleted
                        for op in delete_ops:
                            for item in op.get("affected_items", []):
                                citation_content += f"❎ - {item}\n"
                        citation_content += "\n"
                
                # Send citation
                await self._send_citation(
                    event_emitter,
                    url="module://mmc/memories",
                    title="Memories Consolidated",
                    content=citation_content,
                )
            
            # Emit consolidated status update
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
                    status_message = f"🧠 Memory consolidation: {', '.join(status_parts)}"
                else:
                    status_message = "🧠 Memory consolidation complete"
                
                # Send a single status update
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
                
                # Status will be cleared after the final message or completion.
            
            # Log overall task completion with timing
            total_duration = time.time() - start_time
            logger.info(
                "Background memory consolidation completed in %.2f seconds", total_duration
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
            
            # Send a message to the user with the consolidation results
            if event_emitter:
                await self._safe_emit(
                    event_emitter,
                    {
                        "type": "message",
                        "data": {
                            "role": "assistant",
                            # Add leading newline for better separation in UI
                            "content": f"\nI've organized your memories. {total_ops} operations performed ({new_count} created, {update_count} updated, {delete_count} deleted)."
                        },
                    },
                )
                
                # Clear status immediately without affecting citation persistence
                if event_emitter:
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
            # Enhanced error logging
            logger.error(
                "Error in background memory consolidation: %s",
                e,
            )
            import traceback
            logger.error("Traceback: %s", traceback.format_exc())
            
            # Emit error status
            if self.valves.show_status and event_emitter:
                self._non_blocking_emit(
                    event_emitter,
                    {
                        "type": "status",
                        "data": {
                            "description": "❌ Error consolidating memories",
                            "done": True,
                        },
                    },
                )

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process incoming messages (called by Open WebUI).
        Detects consolidation requests and triggers async memory consolidation.

        Args:
            body: The message body
            __event_emitter__: Optional event emitter for notifications
            __user__: Optional user information

        Returns:
            The processed message body
        """
        # Basic validation
        if not body or not isinstance(body, dict) or not __user__:
            return body
        
        # Process only if we have messages
        if not body.get("messages"):
            return body
        
        try:
            # Check if there are any user messages
            user_messages = [m for m in body["messages"] if m.get("role") == "user"]
            if not user_messages:
                return body
            
            # Get the last user message
            last_user_message = user_messages[-1].get("content", "")
            if not last_user_message:
                return body
            
            # Check if this is a consolidation request
            if not self._is_consolidation_request(last_user_message):
                return body
            
            logger.info(
                "Detected memory consolidation request: %s",
                (
                    last_user_message[:50] + "..."
                    if len(last_user_message) > 50
                    else last_user_message
                ),
            )
            
            # Get user object
            user = Users.get_user_by_id(__user__["id"])
            if not user:
                return body
            
            # Always use background processing for consolidation
            # Initial status update
            if self.valves.show_status and __event_emitter__:
                self._non_blocking_emit(
                    __event_emitter__,
                    {
                        "type": "status",
                        "data": {
                            "description": "🧠 Consolidating memories...",
                            "done": False,
                        },
                    },
                )

            # Create and register background task
            logger.info("Creating background memory consolidation task")

            # Create the task
            task = asyncio.create_task(
                self._process_consolidation_background(
                    last_user_message, user, __user__["id"], __event_emitter__
                )
            )

            # Add task to set and set up callback for tracking and cleanup
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

            # Add a system message to acknowledge the consolidation request
            if "messages" in body:
                body["messages"].append({
                    "role": "assistant",
                    "content": "I'll organize your memories. This will run in the background and I'll let you know when it's complete."
                })

            return body

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
        """Pass-through method for outgoing messages."""
        # No processing needed on outlet for MMC
        return body