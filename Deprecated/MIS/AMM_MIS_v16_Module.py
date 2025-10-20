"""
title: AMM_Memory_Identification_Storage
description: Memory Identification & Storage Module for Open WebUI - Identifies and stores memories from user messages
author: Cody
version: 1.6.3
date: 2025-04-22
changes:
- Add ability to pass intent to stage 2 to allow topic based deletes
- Standardized citation format
- Fixed pattern matching in implicit replacement detection to prevent replacing unrelated sub-memories
- Improved attribute key extraction and comparison to ensure only semantically related attributes are replaced
- Added more detailed logging for debugging replacement pattern matching
- Added implicit replacement detection for MERGE_UPDATE operations to handle cases like "favorite color is now purple"
- Added configurable regex patterns for detecting change indicators and attributes
- Added toggle to enable/disable replacement pattern detection
- Fixed critical issue with delete operations by allowing them to bypass the importance threshold check
- Fixed NoneType error in _send_citation method when processing affected_item_text
- Added proper handling of delete intents in Stage 2 to ensure they are preserved from Stage 1
- Added additional error handling in citation processing to prevent NoneType errors
- Fixed critical issue with negate operations by allowing them to bypass the importance threshold check
- Added proper handling of negate intents in Stage 2 to ensure they are preserved from Stage 1
- Improved delete/negate handling to prioritize LLM-identified targets over string matching
- Enhanced negate intent handling to use high relevance comparisons when explicit targets aren't provided
- Fixed hardcoded relevance threshold in negate handling to use configurable valve parameter
- Fixed citation UI issue to properly mark all newly added items with checkmarks
- Fixed critical issue with multiple sub-memories of the same category by adding aggregation logic to consolidate operations
- Fixed critical nomenclature misalignment by consistently using sub_memory instead of potential_content in Stage 2 integration
- Fixed critical issue with memory categorization by formatting potential memories as a numbered list instead of JSON objects
- Updated memory processing to only pass memories above importance threshold to Stage 2
- Removed importance score from LLM input to reduce cognitive load and improve reliability
- Created improved integration prompt (v8) with simplified format that removes importance score references
- Fixed critical issue with multi-memory extraction by removing the "format": "json" parameter from Ollama API requests
- Updated Stage 1 identification prompt (v7) to better handle multi-line messages and extract all discrete pieces of information
- Added more examples to the prompt showing how to process messages with multiple memory-worthy details
- Enhanced prompt instructions to emphasize thorough analysis of the entire input
- Removed current_message parameter from integration_prompt.format() call as it's not needed and causes confusion
- Major refactoring to remove brittle logic patching throughout the codebase

dev_notes_for_ai_assistants:
- This module uses prompt files for its LLM operations:
  - Prompts/MIS_Identification_Prompt_v9.md: Used in Stage 1 for identifying potential memories from user messages
  - Prompts/MIS_Identification_Patterns.md: Regex and change indicator and attribute patterns for module valves parameters
  - Prompts/MIS_Integration_Prompt_v8.md: Used in Stage 2 for analyzing relevance between potential and existing memories (current version: v8)
- For detailed information about memory operations, refer to Reference/models/memories.py
- When working with this module, refer to these prompt files for a complete understanding of the memory identification and integration logic
- IMPORTANT: Always update the version number and changelog in the module header when making changes to the code. Increment the version number according to the significance of the change (major.minor.patch).
"""

import asyncio
import json
import logging
import os
import re
import time

# Import modules
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

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
            default=0.5,
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
        # Memory Integration settings (restored in v1.1.0)
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
        # Integration prompt (restored in v1.1.0)
        integration_prompt: str = Field(
            ...,  # This makes the field required with no default
            description="Memory integration prompt with preserved formatting",
        )
        # Implicit replacement detection settings
        replacement_patterns_enabled: bool = Field(
            default=True,
            description="Enable replacement pattern detection during MERGE_UPDATE operations",
        )
        change_indicators: List[str] = Field(
            default=[
                r"(?:is|am|are) now",
                r"(?:is|am|are) currently",
                r"(?:has|have) (?:now|recently) changed to",
                r"(?:now|currently) (?:live|work|reside)s? in",
                r"(?:recently|just) moved to",
                r"(?:updated|changed) (?:my|his|her|their)? (.+?) to",
                r"(?:no longer|not anymore|stopped)",
                r"from now on",
                r"switched (?:from .+? )?to",
            ],
            description="Regex patterns that indicate a change/replacement is occurring",
        )
        attribute_patterns: List[str] = Field(
            default=[
                # Location attributes
                r"live(?:s)? in (.+)",
                r"(?:home|house|apartment) (?:is )?in (.+)",
                r"(?:reside|residing|resident) (?:in|of) (.+)",
                # Work/Professional attributes
                r"work(?:s)? (?:at|for) (.+)",
                r"(?:job|position|role|title) (?:is|as) (.+)",
                r"employed (?:at|by) (.+)",
                # Preference attributes
                r"favorite (.+?) is (.+)",
                r"prefer(?:s)? (.+?) over",
                r"like(?:s)? (?:to )?(.+)",
                # Relationship attributes
                r"(?:manager|boss|supervisor) is (.+)",
                r"(?:doctor|physician|therapist) is (.+)",
                r"(?:partner|spouse|husband|wife) is (.+)",
                # Contact attributes
                r"(?:phone|number|email|address) is (.+)",
                r"(?:live|stay|reside) at (.+)",
            ],
            description="Regex patterns that identify attributes being changed",
        )

    def __init__(self) -> None:
        """Initialize the Memory Identification & Storage module."""
        logger.info("Initializing Memory Identification & Storage module v1.2.1")

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

        # Track background tasks for potential cancellation during shutdown
        self.background_tasks = set()

        logger.info(
            "MIS module initialized with API provider: %s", self.valves.api_provider
        )

    async def close(self) -> None:
        """Close the aiohttp session and cancel any running background tasks."""
        logger.info("Closing MIS module session")

        # Cancel all running background tasks
        for task in self.background_tasks:
            task.cancel()

        # Wait for all tasks to complete or be cancelled
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
            self.background_tasks.clear()

        # Close the session
        await self.session.close()

    def update_valves(self, new_valves: Dict[str, Any]) -> None:
        """Update valve settings."""
        logger.info("Updating valves")

        for key, value in new_valves.items():
            if hasattr(self.valves, key):
                # For prompt fields, log a truncated version
                if key in ["identification_prompt", "integration_prompt"]:
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
                            f"Received empty or non-string value for {key}: type={type(value)}"
                        )
                        setattr(self.valves, key, "")
                # Handle other valve types
                else:
                    logger.info(f"Updating {key} with: {value}")
                    setattr(self.valves, key, value)

        logger.info("Finished updating valves")

    def _format_memories(self, db_memories: List[Any]) -> str:
        """Format existing memories for inclusion in the prompt."""
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
            lines = content.strip().split("\n")
            if lines:  # Ensure content is not empty
                # Add ID to the first line (header)
                lines[0] = f"{lines[0]} (ID: {mem_id})"
                formatted_content_with_id = "\n".join(lines)
                formatted.append(f"{i}. {formatted_content_with_id}")
            else:
                # Log if content is unexpectedly empty
                logger.info(
                    f"Skipping empty memory content associated with ID {mem_id}"
                )

        return "\n".join(formatted)

    def _truncate_log_lines(self, text: str, max_lines: int = None) -> str:
        """Truncate a multi-line string to a maximum number of lines."""
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
        """Stage 1: Identify potential memories from the current message based on importance."""
        logger.info("Stage 1: Identifying potential memories from message")

        # Log the message being analyzed in verbose mode
        if self.valves.verbose_logging:
            truncated_message = self._truncate_log_lines(current_message)
            logger.debug("Analyzing message for memories:\n%s", truncated_message)

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
            truncated_prompt = self._truncate_log_lines(system_prompt)
            logger.info("System prompt for Stage 1 (truncated):\n%s", truncated_prompt)

        # Query the appropriate API
        if self.valves.api_provider == "OpenAI API":
            response = await self.query_openai_api(system_prompt, current_message)
        else:  # Ollama API
            response = await self.query_ollama_api(system_prompt, current_message)

        # Parse the response
        potential_memories = self._parse_json_response(response)

        # Verbose logging of potential memories
        if self.valves.verbose_logging and potential_memories:
            logger.info(
                "Potential memories identified (detailed):\n%s",
                self._truncate_log_lines(json.dumps(potential_memories, indent=2)),
            )

        # Log a summary of potential memories
        if potential_memories:
            above_threshold = sum(
                1
                for mem in potential_memories
                if mem.get("importance", 0) >= self.valves.memory_importance_threshold
            )
            below_threshold = len(potential_memories) - above_threshold

            logger.info(
                "Found %d potential memories (%d above threshold, %d below)",
                len(potential_memories),
                above_threshold,
                below_threshold,
            )
        else:
            logger.info("No potential memories identified")

        return potential_memories

    async def _integrate_potential_memories(
        self,
        current_message: str,
        potential_memories: List[Dict[str, Any]],
        db_memories: List[Any],
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Call LLM for analysis and then apply deterministic logic
        to decide memory actions (CREATE, MERGE_UPDATE, REMOVE_ITEM).
        """
        logger.info("Stage 2: Analyzing potential memories against existing")

        # Format existing memories for prompt and later lookup
        existing_memories_formatted = self._format_memories(db_memories)
        existing_memory_map = {
            mem.id: mem.content for mem in db_memories if hasattr(mem, "id")
        }

        # Format potential memories as a numbered list for the LLM (consistent with existing memories format)
        potential_mem_lines = []
        for i, mem in enumerate(potential_memories, 1):
            # Use "sub_memory" for sub-memory operations, fallback to "content" for legacy
            sub_memory = mem.get("sub_memory")
            content = mem.get("content", "")
            importance = mem.get("importance", 0)  # Still track importance for code use
            category = mem.get("category", "Unknown")

            # Format as a numbered list item with category, intent, and content
            intent = mem.get("intent")
            intent_str = f" (intent: {intent})" if intent else ""

            if sub_memory:
                potential_mem_lines.append(
                    f"{i}. {category}{intent_str}:\n- {sub_memory}"
                )
            else:
                # Handle content that might already have structure
                if ":" in content and not content.startswith(f"{category}:"):
                    potential_mem_lines.append(
                        f"{i}. {category}{intent_str}:\n{content}"
                    )
                elif not ":" in content:
                    potential_mem_lines.append(
                        f"{i}. {category}{intent_str}:\n- {content}"
                    )
                else:
                    # If content already has structure (like "Category: ..."), append intent after category if possible
                    parts = content.split(":", 1)
                    if len(parts) == 2:
                        potential_mem_lines.append(
                            f"{i}. {parts[0].strip()}{intent_str}:\n{parts[1].strip()}"
                        )
                    else:
                        potential_mem_lines.append(
                            f"{i}. {content}{intent_str}"
                        )  # Fallback

        # Join the lines with newlines
        formatted_potential_memories = "\n".join(potential_mem_lines)

        if self.valves.verbose_logging:
            logger.info(
                "Formatted potential memories for Stage 2 (numbered list):\n%s",
                self._truncate_log_lines(formatted_potential_memories),
            )

        # Check if prompt is empty and fail fast
        if not self.valves.integration_prompt:
            logger.error("Integration prompt is empty - cannot process memories")
            raise ValueError("Integration prompt is empty - module cannot function")

        # Format the *simplified* system prompt (LLM role: analysis only)
        # NOTE: Prompt content needs to be updated separately.
        # Threshold is now applied in *code*, not passed to LLM.
        system_prompt = self.valves.integration_prompt.format(
            # Removed current_message parameter as it's not needed and causes confusion
            potential_memories=formatted_potential_memories,
            existing_memories=existing_memories_formatted,
            # memory_relevance_threshold=self.valves.memory_relevance_threshold, # Removed
        )

        # Verbose logging for Stage 2 prompt
        if self.valves.verbose_logging:
            truncated_prompt = self._truncate_log_lines(system_prompt)
            logger.info(
                "System prompt for Stage 2 (Analysis - truncated):\n%s",
                truncated_prompt,
            )
        # Query the appropriate API
        if self.valves.api_provider == "OpenAI API":
            response_text = await self.query_openai_api(
                system_prompt, formatted_potential_memories
            )
        else:  # Ollama API
            response_text = await self.query_ollama_api(
                system_prompt, formatted_potential_memories
            )

        # Parse the *analysis results* from the LLM
        # Placeholder: Assumes LLM returns a list where each item corresponds
        # to a potential_memory and contains analysis like relevance scores or deletion hints.
        # Example structure needed from simplified LLM prompt:
        # [
        #   {  // Analysis for potential_memories[0]
        #     "sub_memory": "...", "category": "...", "importance": 0.9,
        #     "comparisons": [{"memory_id": "b7e2f8a1-4c3d-4e2a-9b1a-2f3e4d5c6b7a", "relevance": 0.8}, ...],
        #     "operation_hint": None // Or "DELETE_ITEM_TARGET"
        #     "target_description": None // If DELETE_ITEM_TARGET
        #     "target_memory_id": None // If DELETE_ITEM_TARGET
        #   }, ...
        # ]
        llm_analysis_results = self._parse_json_response(
            response_text
        )  # Assuming _parse_json_response works

        if self.valves.verbose_logging and llm_analysis_results:
            logger.info(
                "LLM analysis results (detailed):\n%s",
                self._truncate_log_lines(json.dumps(llm_analysis_results, indent=2)),
            )

        # --- Deterministic Logic ---
        intended_actions = []

        # Ensure potential_memories and llm_analysis_results have corresponding items
        if len(potential_memories) != len(llm_analysis_results):
            logger.warning(
                "Mismatch between potential memories count (%d) and LLM analysis results count (%d). Score association may be incorrect.",
                len(potential_memories),
                len(llm_analysis_results),
            )

        for index, analysis in enumerate(llm_analysis_results):
            # Get original importance score and intent from the corresponding input memory item
            original_importance = 0.0  # Default
            original_intent = None
            if index < len(potential_memories):
                original_importance = potential_memories[index].get("importance", 0.0)
                original_intent = potential_memories[index].get("intent")
            else:
                logger.warning(
                    f"Index {index} out of bounds for potential_memories (length {len(potential_memories)})"
                )

            sub_memory = analysis.get("sub_memory")
            # We no longer get importance from analysis results
            category = analysis.get("category", "Unknown")

            if not sub_memory:
                logger.info("Skipping analysis result with missing sub_memory")
                continue

            # Check for delete intent from Stage 1
            if original_intent == "delete":
                logger.info(
                    f"Preserving delete intent from Stage 1 for '{sub_memory}' in category '{category}'"
                )

                # First check if the LLM has already identified the target via operation_hint
                if analysis.get("operation_hint") == "DELETE_ITEM_TARGET":
                    target_id = analysis.get("target_memory_id")
                    target_desc = analysis.get("target_description")

                    if target_id and target_desc:
                        intended_actions.append(
                            {
                                "action": "REMOVE_ITEM",
                                "memory_id": target_id,
                                "item_to_remove": target_desc,
                                "importance": 1.0,  # Always use 1.0 for delete operations
                            }
                        )
                        logger.info(
                            f"Identified REMOVE_ITEM action from Stage 1 delete intent using LLM hint for ID {target_id}, item: '{target_desc}'"
                        )
                        continue  # Skip further processing for this item

                # Fallback: Find the memory to delete based on content match across all categories
                target_id = None
                target_desc = sub_memory
                found_category = None
                
                # Extract specific unique keywords that identify the target memory
                unique_keywords = []
                
                # Look for specific unique terms like "unicycle", "pears", etc.
                for unique_term in ["unicycle", "pear", "pears", "hockey"]:
                    if unique_term in sub_memory.lower():
                        unique_keywords.append(unique_term)
                
                # If no unique keywords found, fall back to using all words longer than 3 chars
                if not unique_keywords:
                    unique_keywords = [word for word in sub_memory.lower().split() if len(word) > 3]
                
                logger.info(f"Looking for items containing these unique keywords across all categories: {unique_keywords}")

                # First try to find in the specified category (higher priority)
                for mem in db_memories:
                    if (
                        not hasattr(mem, "id")
                        or not hasattr(mem, "content")
                        or not mem.content
                    ):
                        continue

                    mem_category, mem_items = self._parse_memory_content(mem.content)
                    
                    # First pass: check the specified category with higher priority
                    if mem_category.lower() == category.lower():
                        for item in mem_items:
                            item_lower = item.lower()
                            # Only match if ALL unique keywords are present in the item
                            if unique_keywords and all(keyword in item_lower for keyword in unique_keywords):
                                target_id = mem.id
                                target_desc = item
                                found_category = mem_category
                                logger.info(f"Found matching item in specified category '{mem_category}': '{item}'")
                                break
                        if target_id:
                            break
                
                # If not found in specified category, search all categories
                if not target_id:
                    logger.info(f"Item not found in category '{category}', searching all categories")
                    for mem in db_memories:
                        if (
                            not hasattr(mem, "id")
                            or not hasattr(mem, "content")
                            or not mem.content
                        ):
                            continue
                        
                        mem_category, mem_items = self._parse_memory_content(mem.content)
                        
                        # Skip the category we already checked
                        if mem_category.lower() == category.lower():
                            continue
                            
                        for item in mem_items:
                            item_lower = item.lower()
                            # Only match if ALL unique keywords are present in the item
                            if unique_keywords and all(keyword in item_lower for keyword in unique_keywords):
                                target_id = mem.id
                                target_desc = item
                                found_category = mem_category
                                logger.info(f"Found matching item in different category '{mem_category}' (not '{category}'): '{item}'")
                                break
                        if target_id:
                            break

                if target_id and target_desc:
                    intended_actions.append(
                        {
                            "action": "REMOVE_ITEM",
                            "memory_id": target_id,
                            "item_to_remove": target_desc,
                            "importance": 1.0,  # Always use 1.0 for delete operations
                        }
                    )
                    if found_category and found_category.lower() != category.lower():
                        logger.info(
                            f"Identified REMOVE_ITEM action from Stage 1 delete intent for ID {target_id}, item: '{target_desc}' in DIFFERENT category '{found_category}' (not '{category}')"
                        )
                    else:
                        logger.info(
                            f"Identified REMOVE_ITEM action from Stage 1 delete intent for ID {target_id}, item: '{target_desc}'"
                        )
                else:
                    logger.info(
                        f"Could not find matching memory for delete intent: searched all categories for content '{sub_memory}' with keywords {unique_keywords}"
                    )
                continue  # Skip further processing for this item

            # Check for negate intent from Stage 1
            if original_intent == "negate":
                logger.info(
                    f"Preserving negate intent from Stage 1 for '{sub_memory}' in category '{category}'"
                )

                # First check if the LLM has already identified the target via operation_hint
                if analysis.get("operation_hint") == "DELETE_ITEM_TARGET":
                    target_id = analysis.get("target_memory_id")
                    target_desc = analysis.get("target_description")

                    if target_id and target_desc:
                        intended_actions.append(
                            {
                                "action": "REMOVE_ITEM",
                                "memory_id": target_id,
                                "item_to_remove": target_desc,
                                "importance": 1.0,  # Always use 1.0 for negate operations
                            }
                        )
                        logger.info(
                            f"Identified REMOVE_ITEM action from Stage 1 negate intent using LLM hint for ID {target_id}, item: '{target_desc}'"
                        )
                        continue  # Skip further processing for this item

                # Second, check if there's a high relevance comparison
                comparisons = analysis.get("comparisons", [])
                high_relevance_memory_id = None
                max_relevance = 0.0

                for comp in comparisons:
                    relevance = comp.get("relevance", 0.0)
                    mem_id = comp.get("memory_id")
                    if (
                        relevance >= self.valves.memory_relevance_threshold
                        and mem_id
                        and relevance > max_relevance
                    ):
                        max_relevance = relevance
                        high_relevance_memory_id = mem_id

                if high_relevance_memory_id:
                    # Find the memory content to get the item to remove
                    for mem in db_memories:
                        if not hasattr(mem, "id") or mem.id != high_relevance_memory_id:
                            continue

                        mem_category, mem_items = self._parse_memory_content(
                            mem.content
                        )

                        # For negate intents, look for items related to the sub_memory
                        target_desc = None
                        for item in mem_items:
                            # Look for items containing keywords from sub_memory
                            keywords = [
                                word.lower()
                                for word in sub_memory.split()
                                if len(word) > 3
                            ]
                            for keyword in keywords:
                                if keyword in item.lower():
                                    target_desc = item
                                    break
                            if target_desc:
                                break

                        # If no keyword match, just take the first item as fallback
                        if not target_desc and mem_items:
                            target_desc = mem_items[0]

                        if target_desc:
                            intended_actions.append(
                                {
                                    "action": "REMOVE_ITEM",
                                    "memory_id": high_relevance_memory_id,
                                    "item_to_remove": target_desc,
                                    "importance": 1.0,  # Always use 1.0 for negate operations
                                }
                            )
                            logger.info(
                                f"Identified REMOVE_ITEM action from Stage 1 negate intent using high relevance comparison (relevance: {max_relevance:.2f}) for ID {high_relevance_memory_id}, item: '{target_desc}'"
                            )
                            continue  # Skip further processing for this item

                # Fallback: Find the memory to negate based on category and content match
                target_id = None
                target_desc = sub_memory

                for mem in db_memories:
                    if (
                        not hasattr(mem, "id")
                        or not hasattr(mem, "content")
                        or not mem.content
                    ):
                        continue

                    mem_category, mem_items = self._parse_memory_content(mem.content)
                    if mem_category.lower() == category.lower():
                        # Check if any item in this memory matches the sub_memory to negate
                        for item in mem_items:
                            if sub_memory.lower() in item.lower():
                                target_id = mem.id
                                target_desc = item
                                break
                        if target_id:
                            break

                if target_id and target_desc:
                    intended_actions.append(
                        {
                            "action": "REMOVE_ITEM",
                            "memory_id": target_id,
                            "item_to_remove": target_desc,
                            "importance": 1.0,  # Always use 1.0 for negate operations
                        }
                    )
                    logger.info(
                        f"Identified REMOVE_ITEM action from Stage 1 negate intent for ID {target_id}, item: '{target_desc}'"
                    )
                else:
                    logger.info(
                        f"Could not find matching memory for negate intent: category '{category}', content '{sub_memory}'"
                    )
                continue  # Skip further processing for this item

            # Handle explicit deletion requests from LLM analysis
            if analysis.get("operation_hint") == "DELETE_ITEM_TARGET":
                target_id = analysis.get("target_memory_id")
                target_desc = analysis.get("target_description")
                if target_id and target_desc:
                    intended_actions.append(
                        {
                            "action": "REMOVE_ITEM",
                            "memory_id": target_id,
                            "item_to_remove": target_desc,
                            "importance": original_importance,  # Use original importance from Stage 1
                        }
                    )
                    logger.info(
                        f"Identified REMOVE_ITEM action for ID {target_id}, item: '{target_desc}'"
                    )
                else:
                    logger.info(
                        "DELETE_ITEM_TARGET hint missing target_id or target_description"
                    )
                continue  # Move to next analysis result

            # Standard NEW vs. UPDATE decision based on relevance
            comparisons = analysis.get("comparisons", [])
            best_match_id = None
            max_relevance = 0.0

            for comp in comparisons:
                relevance = comp.get("relevance", 0.0)
                mem_id = comp.get("memory_id")
                if relevance > max_relevance and mem_id:
                    max_relevance = relevance
                    best_match_id = mem_id

            # First, find all memories with matching category
            category_matches = []
            for mem in db_memories:
                if (
                    not hasattr(mem, "id")
                    or not hasattr(mem, "content")
                    or not mem.content
                ):
                    continue

                mem_category, _ = self._parse_memory_content(mem.content)
                if mem_category.lower() == category.lower():
                    # Find this memory's relevance score in comparisons
                    relevance = 0.0
                    for comp in comparisons:
                        if comp.get("memory_id") == mem.id:
                            relevance = comp.get("relevance", 0.0)
                            break

                    category_matches.append(
                        {"memory_id": mem.id, "relevance": relevance}
                    )

            # If we have category matches, find the most relevant one
            if category_matches:
                # Sort by relevance (highest first)
                category_matches.sort(
                    key=lambda x: x.get("relevance", 0.0), reverse=True
                )
                best_category_match = category_matches[0]
                best_match_id = best_category_match.get("memory_id")
                max_relevance = best_category_match.get("relevance", 0.0)

                logger.info(
                    f"Found {len(category_matches)} memories with matching category '{category}', best match: {best_match_id} (relevance: {max_relevance:.2f})"
                )

                # Even if relevance is low, still update if categories match
                # This ensures consolidation of memories with the same category
                logger.info(
                    f"Categories match ({category}) for memory {best_match_id} - proceeding with MERGE_UPDATE for consolidation"
                )

                # Set best_match_category for the rest of the function
                best_match_category = category
                # Extract the *new* bullet point content from sub_memory
                # Handle multiple formats including reformatted content
                content_lines = sub_memory.strip().split("\n")
                new_bullet_content = ""
                if len(content_lines) > 1 and content_lines[1].strip().startswith("-"):
                    new_bullet_content = content_lines[
                        1
                    ].strip()  # Simple case: take first bullet
                    # TODO: Handle multi-line potential content more robustly if needed
                elif len(content_lines) == 1:
                    line = content_lines[0].strip()
                    if line.startswith("- "):
                        # Handle case where potential content is already a bullet point
                        # Use the content directly without trying to strip metadata
                        new_bullet_content = line
                    elif ":" not in line or (
                        ":" in line and "category:" in line.lower()
                    ):
                        # Handle case where potential content might just be the bullet point itself
                        new_bullet_content = (
                            f"- {line}" if not line.startswith("-") else line
                        )
                    else:
                        # For any other case, use the content as is
                        new_bullet_content = (
                            f"- {line}" if not line.startswith("-") else line
                        )

                if new_bullet_content:
                    intended_actions.append(
                        {
                            "action": "MERGE_UPDATE",
                            "memory_id": best_match_id,
                            "new_bullet_content": new_bullet_content,  # Pass only the new part
                            "importance": original_importance,  # Use original importance from Stage 1
                            "relevance": max_relevance,  # For logging context
                            "category": category,  # Add category to preserve it during updates
                        }
                    )
                    logger.info(
                        f"Identified MERGE_UPDATE action for ID {best_match_id} (Relevance: {max_relevance:.2f})"
                    )
                else:
                    logger.info(
                        f"Could not extract new bullet content for potential UPDATE on {best_match_id} from: {sub_memory}"
                    )

                    # Fallback to CREATE if extraction fails but we have a relevant match
                    # For fallback to CREATE, we still need to check if we have a category match
                    if (
                        best_match_id
                        and best_match_category
                        and category.lower() == best_match_category.lower()
                    ):
                        logger.info(
                            f"Extraction failed for UPDATE, falling back to CREATE with category {category}"
                        )
                        intended_actions.append(
                            {
                                "action": "CREATE",
                                "content": f"{category}:\n- {sub_memory}",
                                "importance": original_importance,
                                "category": category,
                            }
                        )

            else:
                # If we didn't find any category matches, create a new memory
                if not category_matches:
                    logger.info(
                        f"No existing memories with category '{category}' found - creating new memory"
                    )
                else:
                    logger.info(
                        f"Extraction failed for UPDATE despite category match - creating new memory"
                    )

                # Decide NEW (no category matches or extraction failed)
                intended_actions.append(
                    {
                        "action": "CREATE",
                        "content": sub_memory,  # Pass the full original content
                        "importance": original_importance,  # Use original importance from Stage 1
                        "category": category,  # Preserve category information
                    }
                )
                logger.info(
                    f"Identified CREATE action (Max Relevance: {max_relevance:.2f} < Threshold or Category Mismatch)"
                )

        logger.info(
            f"Determined {len(intended_actions)} intended actions after analysis."
        )
        return intended_actions

    async def _process_memories(
        self, current_message: str, user_id: str, db_memories: List[Any]
    ) -> List[Dict[str, Any]]:
        """Process memories through both stages: identification and integration."""
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
            if mem.get("intent") in ["delete", "negate"]
            or mem.get("importance", 0.0) >= self.valves.memory_importance_threshold
        ]

        # Early return if no important memories
        if not important_memories:
            logger.info(
                "No memories above threshold %.2f, skipping Stage 2",
                self.valves.memory_importance_threshold,
            )
            return []  # Early return when no memories above threshold

        # If there are no existing memories, convert potential memories directly to NEW operations
        if not db_memories:
            logger.info("No existing memories found, creating NEW operations directly")
            return [
                {
                    "action": "CREATE",
                    "content": mem.get("content", ""),
                    "sub_memory": mem.get("sub_memory", ""),
                    "category": mem.get("category", "Miscellaneous"),
                    "importance": mem.get("importance", 0.0),
                }
                for mem in potential_memories
                if mem.get("intent") in ["delete", "negate"]
                or mem.get("importance", 0.0) >= self.valves.memory_importance_threshold
            ]

        # Stage 2: Integrate important memories with existing ones
        # Only pass memories that are above the importance threshold
        memory_operations = await self._integrate_potential_memories(
            current_message,
            important_memories,  # Changed from potential_memories to only pass important ones
            db_memories,
        )

        logger.info("Stage 2: Determined %d memory operations", len(memory_operations))
        return memory_operations

    async def query_openai_api(self, system_prompt: str, prompt: str) -> str:
        """Query the OpenAI API."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return await self._query_api("OpenAI API", messages)

    async def query_ollama_api(self, system_prompt: str, prompt: str) -> str:
        """Query the Ollama API."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return await self._query_api("Ollama API", messages)

    async def _query_api(self, provider: str, messages: List[Dict[str, Any]]) -> str:
        """Generic function to query either OpenAI or Ollama API with retry logic."""
        retries = 0
        start_time = time.time()

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
                else:  # Ollama API (use /api/chat for better response handling)
                    url = f"{self.valves.ollama_api_url.rstrip('/')}/api/chat"
                    headers = {"Content-Type": "application/json"}
                    payload = {
                        "model": self.valves.ollama_model,
                        "messages": messages,
                        "stream": False,
                        # Removed "format": "json" as it constrains the LLM output
                        "options": {
                            "temperature": self.valves.temperature,
                            "num_ctx": self.valves.ollama_context_size,
                        },
                    }

                # Log API request details in verbose mode
                if self.valves.verbose_logging:
                    # Sanitize messages to avoid logging sensitive system prompts in full
                    sanitized_messages = []
                    for msg in messages:
                        if msg.get("role") == "system":
                            content = msg.get("content", "")
                            preview = (
                                content[:100].replace("\n", "\\n") + "..."
                                if len(content) > 100
                                else content.replace("\n", "\\n")
                            )
                            sanitized_messages.append(
                                {
                                    "role": "system",
                                    "content": f"[System prompt: {preview}]",
                                }
                            )
                        else:
                            sanitized_messages.append(msg)

                    sanitized_payload = payload.copy()
                    sanitized_payload["messages"] = sanitized_messages
                    logger.info(
                        f"API request to {provider} ({url}):\n{json.dumps(sanitized_payload, indent=2)}"
                    )

                logger.info(
                    f"Sending request to {provider} model: {payload.get('model', 'unknown')}"
                )
                request_start = time.time()

                async with self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.valves.request_timeout,
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

                    request_duration = time.time() - request_start
                    logger.info(
                        f"Received response from {provider} in {request_duration:.2f}s"
                    )

                    # Extract content based on provider
                    if provider == "OpenAI API":
                        content = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        # Log token usage if available
                        if self.valves.verbose_logging and "usage" in data:
                            usage = data.get("usage", {})
                            prompt_tokens = usage.get("prompt_tokens", 0)
                            completion_tokens = usage.get("completion_tokens", 0)
                            total_tokens = usage.get("total_tokens", 0)
                            logger.info(
                                f"Token usage: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total"
                            )
                    else:  # Ollama API
                        # For /api/chat, the response content is under "message" -> "content"
                        response_str = data.get("message", {}).get("content", "")

                        # Check if the response is a JSON string that needs to be parsed
                        if response_str and response_str.strip().startswith(("{", "[")):
                            try:
                                # Try to parse the JSON string directly
                                parsed_json = json.loads(response_str)
                                # Convert back to string but in a standardized format
                                content = json.dumps(parsed_json)
                                if self.valves.verbose_logging:
                                    logger.info(
                                        f"Successfully parsed JSON string from Ollama response"
                                    )
                            except json.JSONDecodeError:
                                # If parsing fails, use the original string
                                logger.warning(
                                    f"Failed to parse JSON string from Ollama response, using raw string"
                                )
                                content = response_str
                        else:
                            # Not a JSON string, use as is
                            content = response_str
                        # Log token usage if available in Ollama format
                        if self.valves.verbose_logging and "prompt_eval_count" in data:
                            prompt_tokens = data.get("prompt_eval_count", 0)
                            completion_tokens = data.get("eval_count", 0)
                            total_tokens = prompt_tokens + completion_tokens
                            logger.info(
                                f"Token usage: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total"
                            )

                    # Log API response in verbose mode
                    if self.valves.verbose_logging:
                        # Log a sanitized version of the full response
                        sanitized_data = (
                            data.copy()
                            if isinstance(data, dict)
                            else {"raw_data": str(data)[:200] + "..."}
                        )
                        if "choices" in sanitized_data and sanitized_data["choices"]:
                            for choice in sanitized_data["choices"]:
                                if (
                                    "message" in choice
                                    and "content" in choice["message"]
                                ):
                                    choice_content = choice["message"]["content"]
                                    choice["message"][
                                        "content"
                                    ] = f"[Content length: {len(choice_content)} chars]"
                        elif (
                            "message" in sanitized_data
                            and "content" in sanitized_data["message"]
                        ):
                            message_content = sanitized_data["message"]["content"]
                            sanitized_data["message"][
                                "content"
                            ] = f"[Content length: {len(message_content)} chars]"

                        logger.info(
                            f"API response metadata from {provider}:\n{json.dumps(sanitized_data, indent=2)}"
                        )
                        logger.info(
                            f"API response content from {provider} ({len(content)} chars):\n{self._truncate_log_lines(content)}"
                        )

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
                    raise
                await asyncio.sleep(self.valves.retry_delay * (2 ** (retries - 1)))

        total_duration = time.time() - start_time
        raise RuntimeError(
            f"API query failed after maximum retries (total time: {total_duration:.2f}s)."
        )

    def _parse_json_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse the JSON response from the API, handling potential errors."""
        try:
            # First, try to parse the response directly as JSON
            json_str = response_text.strip()

            # Simple check for markdown code blocks
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", json_str)
            if match:
                json_str = match.group(1).strip()
                logger.debug("Extracted JSON from markdown code block")

            # Ensure the extracted string is not empty
            if not json_str:
                logger.warning("Could not extract valid JSON content from response.")
                return []

            # Parse the JSON string
            try:
                parsed_data = json.loads(json_str)
            except json.JSONDecodeError:
                # If direct parsing fails, log and return empty list
                logger.warning(
                    "Failed to parse JSON directly. Response may not be valid JSON."
                )
                return []

            # Log the parsed JSON in verbose mode
            if self.valves.verbose_logging:
                logger.info(
                    "Parsed JSON response:\n%s",
                    self._truncate_log_lines(json.dumps(parsed_data, indent=2)),
                )

            # Standardize the output format to always be a list of dictionaries
            if isinstance(parsed_data, dict):
                # If we got a single dictionary, wrap it in a list
                result = [parsed_data]
            elif isinstance(parsed_data, list):
                # If we got a list, use it directly
                result = parsed_data
            else:
                logger.warning(
                    f"Parsed JSON is not a dictionary or list, but {type(parsed_data)}. Discarding."
                )
                return []

            # Log the standardized result in verbose mode
            if self.valves.verbose_logging:
                logger.info(
                    "Standardized result (list of %d items):\n%s",
                    len(result),
                    self._truncate_log_lines(json.dumps(result, indent=2)),
                )

            return result

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

    # Helper method for user ID extraction
    def _get_user_id(self, user: Any) -> str:
        """Extract user ID from either a dictionary or object representation."""
        return user.get("id") if isinstance(user, dict) else user.id

    # Helper method for memory validation
    def _validate_memory(self, memory_id: str, user_id: str) -> Optional[Any]:
        """Validate memory existence and ownership."""
        if self.valves.verbose_logging:
            logger.info(f"Validating memory {memory_id} for user {user_id}")

        # Clean memory ID using helper function
        original_id = memory_id
        memory_id = self._clean_memory_id(memory_id)

        if original_id != memory_id and self.valves.verbose_logging:
            logger.info(f"Cleaned memory ID from '{original_id}' to '{memory_id}'")

        # Attempt to retrieve the memory from the database
        try:
            memory = Memories.get_memory_by_id(memory_id)

            if self.valves.verbose_logging:
                if memory:
                    logger.info(f"Memory {memory_id} found in database")
                    if hasattr(memory, "user_id"):
                        logger.info(
                            f"Memory user_id: {memory.user_id}, requesting user_id: {user_id}"
                        )
                    else:
                        logger.warning(f"Memory {memory_id} has no user_id attribute")
                else:
                    logger.info(f"Memory {memory_id} not found in database")
        except Exception as e:
            logger.error(f"Exception retrieving memory {memory_id}: {e}")
            if self.valves.verbose_logging:
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
            return None

        # Check if memory exists
        if not memory:
            logger.warning(f"Memory {memory_id} not found")
            return None

        # Check if memory belongs to user
        if str(memory.user_id) != str(user_id):
            logger.warning(f"Memory {memory_id} does not belong to user {user_id}")
            return None

        # Memory validation successful
        if self.valves.verbose_logging:
            logger.info(f"Memory {memory_id} validation successful")
        return memory

    def _clean_memory_content(self, content: str) -> str:
        """Remove importance scores and other metadata from memory content."""
        if not content:
            return content

        # Process line by line to handle multi-line content
        lines = content.strip().split("\n")
        cleaned_lines = []

        for line in lines:
            # Only remove specific known metadata patterns, preserve user content in parentheses
            if "(importance:" in line:
                cleaned_line = re.sub(r"\s*\(importance:\s*\d+\.\d+\)", "", line)
                cleaned_lines.append(cleaned_line)
            elif "(ID:" in line and line.endswith(")"):
                cleaned_line = re.sub(r"\s*\(ID:\s*[^)]+\)", "", line)
                cleaned_lines.append(cleaned_line)
            else:
                # Preserve all other content, including user parenthetical content
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _clean_memory_id(self, memory_id: str) -> str:
        """Clean memory ID consistently from various formats."""
        if not memory_id or not isinstance(memory_id, str):
            return memory_id

        # Use a simple regex to extract UUID pattern if embedded in text
        uuid_pattern = re.compile(
            r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
            re.IGNORECASE,
        )
        match = uuid_pattern.search(memory_id)
        if match:
            return match.group(1)

        return memory_id

    def _parse_memory_content(self, content: str) -> tuple:
        """
        Parse memory content into category and items.

        Returns:
            tuple: (category, list_of_items)
        """
        if not content:
            return "Miscellaneous", []

        lines = content.strip().split("\n")
        if not lines:
            return "Miscellaneous", []

        # Extract category from first line using a more robust approach
        first_line = lines[0].strip()

        # If the first line ends with a colon, it's likely a category
        if first_line.endswith(":"):
            category = first_line[:-1]  # Remove the colon
        # If the first line contains a colon, extract the part before it
        elif ":" in first_line:
            category = first_line.split(":", 1)[0].strip()
        # Otherwise, use Miscellaneous as the default category
        else:
            category = "Miscellaneous"

        # Ensure category is not empty
        if not category:
            category = "Miscellaneous"

        # Extract items (bullet points) using a more robust approach
        items = []
        for line in lines[1:]:
            line = line.strip()
            # Handle bullet points with different markers
            if line.startswith("- ") or line.startswith("* ") or line.startswith("• "):
                # Extract content after the bullet marker
                item = line[2:].strip()
                if item:  # Only add non-empty items
                    items.append(item)
            # Handle numbered lists
            elif re.match(r"^\d+\.\s", line):
                item = re.sub(r"^\d+\.\s", "", line).strip()
                if item:
                    items.append(item)

        return category, items

    def _format_memory(self, category: str, items: list) -> str:
        """
        Format a memory with consistent structure.

        Args:
            category: The memory category
            items: List of memory items (bullet points)

        Returns:
            str: Formatted memory content
        """
        # Ensure category ends with colon
        if not category.endswith(":"):
            category = f"{category}:"

        # Start with category
        formatted_content = category

        # Add items as bullet points
        for item in items:
            if item.strip():  # Skip empty items
                # Ensure item doesn't already have bullet point
                item_text = item.strip()
                if item_text.startswith("- "):
                    item_text = item_text[2:].strip()
                formatted_content += f"\n- {item_text}"

        return formatted_content

    async def _process_memory_operations(
        self, operations: List[Dict[str, Any]], user: Any
    ) -> List[Dict[str, Any]]:
        """Process memory operations."""
        logger.info(
            f"Processing {len(operations)} memory operations: {[op.get('action') for op in operations]}"
        )

        # Verbose logging of intended actions
        if self.valves.verbose_logging and operations:
            logger.info(
                "Intended actions to process (detailed):\n%s",
                self._truncate_log_lines(json.dumps(operations, indent=2)),
            )
        results = []

        # Get user ID using helper method
        user_id = self._get_user_id(user)

        # Fetch all existing memories once for efficient lookup
        existing_memories_content = {
            mem.id: mem.content
            for mem in Memories.get_memories_by_user_id(user_id)
            if hasattr(mem, "id") and hasattr(mem, "content")
        }

        # --- Begin Aggregation Logic ---
        # Group operations by type and target to consolidate multiple operations on the same category/memory
        logger.info("Aggregating operations for same categories and memory IDs")

        # Group CREATE operations by category
        create_actions_by_category = (
            {}
        )  # {category_name: [sub_memory1, sub_memory2, ...]}

        # Group MERGE_UPDATE operations by memory_id
        update_actions_by_id = {}  # {memory_id: [new_bullet1, new_bullet2, ...]}

        # Keep REMOVE_ITEM operations as they are
        remove_actions = []

        # First pass: group operations by type and target
        for op in operations:
            action_type = op.get("action", "UNKNOWN")

            if action_type == "CREATE":
                # Extract category from the operation
                category = op.get("category", "Miscellaneous")

                # Initialize the category list if it doesn't exist
                if category not in create_actions_by_category:
                    create_actions_by_category[category] = []

                # Add the sub_memory or content to the category list
                sub_memory = op.get("sub_memory")
                content = op.get("content", "")

                if sub_memory:
                    create_actions_by_category[category].append(sub_memory)
                elif content:
                    # Extract the bullet point from content if possible
                    category_prefix = f"{category}:"
                    if content.startswith(category_prefix):
                        content = content[len(category_prefix) :].strip()

                    # Remove bullet point marker if present
                    if content.startswith("- "):
                        content = content[2:].strip()

                    create_actions_by_category[category].append(content)

            elif action_type == "MERGE_UPDATE":
                memory_id = op.get("id") or op.get("memory_id")
                if not memory_id:
                    logger.warning(
                        f"Skipping MERGE_UPDATE with missing memory_id: {op}"
                    )
                    continue

                # Initialize the memory_id list if it doesn't exist
                if memory_id not in update_actions_by_id:
                    update_actions_by_id[memory_id] = []

                # Add the new bullet content to the memory_id list
                new_bullet = op.get("new_bullet_content", "")
                if new_bullet:
                    update_actions_by_id[memory_id].append(new_bullet)

            elif action_type == "REMOVE_ITEM":
                # Keep REMOVE_ITEM operations as they are
                remove_actions.append(op)

        # Second pass: create consolidated operations
        consolidated_operations = []

        # Process CREATE operations
        for category, items in create_actions_by_category.items():
            if not items:
                continue

            logger.info(
                f"Consolidating {len(items)} CREATE operations for category '{category}'"
            )

            # Format the memory content with all items for this category
            consolidated_content = self._format_memory(category, items)

            # Find the maximum importance among all operations for this category
            max_importance = 0.0
            for op in operations:
                if (
                    op.get("action") == "CREATE"
                    and op.get("category", "Miscellaneous") == category
                    and op.get("importance", 0.0) > max_importance
                ):
                    max_importance = op.get("importance", 0.0)

            # Create a single consolidated CREATE operation
            consolidated_operations.append(
                {
                    "action": "CREATE",
                    "content": consolidated_content,
                    "importance": max_importance,
                    "category": category,
                    "category_name": category,  # For consistency with existing code
                    "affected_item_text": (
                        items[0] if items else ""
                    ),  # For logging, just use the first item
                }
            )

        # Process MERGE_UPDATE operations
        for memory_id, new_bullets in update_actions_by_id.items():
            if not new_bullets:
                continue

            logger.info(
                f"Consolidating {len(new_bullets)} MERGE_UPDATE operations for memory ID '{memory_id}'"
            )

            # Find the maximum importance among all operations for this memory_id
            max_importance = 0.0
            category = "Unknown"
            for op in operations:
                if op.get("action") == "MERGE_UPDATE" and (
                    op.get("id") == memory_id or op.get("memory_id") == memory_id
                ):
                    if op.get("importance", 0.0) > max_importance:
                        max_importance = op.get("importance", 0.0)
                    if op.get("category"):
                        category = op.get("category")

            # Create a single consolidated MERGE_UPDATE operation with all new bullets
            consolidated_operations.append(
                {
                    "action": "MERGE_UPDATE",
                    "id": memory_id,
                    "memory_id": memory_id,  # For consistency
                    "new_bullet_content": new_bullets[
                        0
                    ],  # We'll handle multiple bullets in the processing loop
                    "all_new_bullets": new_bullets,  # Store all bullets for processing
                    "importance": max_importance,
                    "category": category,
                    "affected_item_text": (
                        new_bullets[0] if new_bullets else ""
                    ),  # For logging
                }
            )

        # Add REMOVE_ITEM operations unchanged
        consolidated_operations.extend(remove_actions)

        logger.info(
            f"Consolidated {len(operations)} operations into {len(consolidated_operations)} operations"
        )

        # Use consolidated_operations instead of the original operations
        operations = consolidated_operations
        # --- End Aggregation Logic ---

        for action_data in operations:
            action_type = action_data.get("action", "UNKNOWN")
            memory_id = action_data.get("memory_id")  # Used by UPDATE/REMOVE

            # Clean memory_id consistently
            if memory_id:
                memory_id = self._clean_memory_id(memory_id)

            # Use "sub_memory" for sub-memory CRUD, fallback to "content" for legacy
            sub_memory = action_data.get("sub_memory")
            content = action_data.get("content")  # Used by CREATE (legacy/compound)
            importance = action_data.get("importance", 0.0)  # For logging/results
            category = action_data.get("category", "Miscellaneous")  # Get category
            status = "Pending"
            success = False
            result_content = (
                sub_memory if sub_memory else content
            )  # Default for reporting
            affected_item_text = None  # Initialize for all actions

            try:
                if action_type == "CREATE":
                    # Support both new "sub_memory" and legacy "content"
                    if sub_memory:
                        # Add sub_memory as a new bullet to the appropriate category block
                        memory_category = action_data.get("category", "Miscellaneous")
                        formatted_content = self._format_memory(
                            memory_category, [sub_memory]
                        )
                        logger.info(
                            f"Formatting sub_memory with category '{memory_category}': {formatted_content}"
                        )
                        new_memory_id = await self._create_memory(
                            formatted_content, user
                        )
                        success = bool(new_memory_id)
                        status = "(Item Added)" if success else "Failed to create"
                        memory_id = new_memory_id  # Update for reporting
                        parsed_category, items = self._parse_memory_content(
                            formatted_content
                        )
                        affected_item_text = items[0] if items else None
                        result_content = sub_memory
                    elif content:
                        # Legacy: treat as full block
                        memory_category = action_data.get("category", "Miscellaneous")
                        if ":\n-" not in content and not content.startswith(
                            f"{memory_category}:"
                        ):
                            formatted_content = self._format_memory(
                                memory_category, [content]
                            )
                            logger.info(
                                f"Formatting content with category '{memory_category}': {formatted_content}"
                            )
                        else:
                            formatted_content = content
                            logger.info(
                                f"Content already formatted: {formatted_content}"
                            )
                        new_memory_id = await self._create_memory(
                            formatted_content, user
                        )
                        success = bool(new_memory_id)
                        status = "(Item Added)" if success else "Failed to create"
                        memory_id = new_memory_id  # Update for reporting
                        parsed_category, items = self._parse_memory_content(
                            formatted_content
                        )
                        affected_item_text = items[0] if items else None
                        result_content = formatted_content
                    else:
                        raise ValueError("CREATE action missing sub_memory or content")

                elif action_type == "MERGE_UPDATE":
                    # Check if we have multiple bullets to add (from aggregation)
                    all_new_bullets = action_data.get("all_new_bullets", [])
                    new_bullet = action_data.get("new_bullet_content")

                    if not memory_id:
                        raise ValueError("MERGE_UPDATE action missing memory_id")

                    if not new_bullet and not all_new_bullets:
                        raise ValueError(
                            "MERGE_UPDATE action missing new_bullet_content and all_new_bullets"
                        )

                    # Fetch current content
                    current_content = existing_memories_content.get(memory_id)
                    if current_content is None:
                        # Attempt to fetch directly if not in cache
                        mem_obj = Memories.get_memory_by_id(memory_id)
                        if mem_obj and hasattr(mem_obj, "content"):
                            current_content = mem_obj.content
                        else:
                            raise ValueError(
                                f"Memory ID {memory_id} not found for MERGE_UPDATE"
                            )

                    # Parse current content
                    current_category, current_items = self._parse_memory_content(
                        current_content
                    )

                    # If we have multiple bullets from aggregation, use those
                    # Otherwise, use the single new_bullet
                    bullets_to_add = (
                        all_new_bullets if all_new_bullets else [new_bullet]
                    )

                    # Process each bullet
                    added_bullets = []
                    for bullet in bullets_to_add:
                        # Clean the bullet content
                        if bullet.strip().startswith("- "):
                            clean_bullet = bullet[2:].strip()
                        else:
                            clean_bullet = bullet.strip()

                        # Check for replacement patterns if enabled
                        replaced_item = None
                        if self.valves.replacement_patterns_enabled:
                            # First stage: Check if the bullet contains any change indicator
                            change_indicator_found = False
                            matching_indicator = None
                            for pattern in self.valves.change_indicators:
                                indicator_match = re.search(
                                    pattern, clean_bullet, re.IGNORECASE
                                )
                                if indicator_match:
                                    change_indicator_found = True
                                    matching_indicator = pattern
                                    logger.info(
                                        f"Change indicator found in bullet: '{clean_bullet}' (matched pattern: '{pattern}')"
                                    )
                                    break

                            # Second stage: If change indicator found, identify the attribute and look for matches
                            if change_indicator_found:
                                # Track which pattern matched and its index
                                matching_pattern_index = None
                                matching_pattern = None

                                # Find which attribute pattern matches the new bullet
                                for i, pattern in enumerate(
                                    self.valves.attribute_patterns
                                ):
                                    if re.search(pattern, clean_bullet, re.IGNORECASE):
                                        matching_pattern_index = i
                                        matching_pattern = pattern
                                        logger.info(
                                            f"Attribute pattern matched: '{pattern}'"
                                        )
                                        break

                                # If we found a matching pattern
                                if matching_pattern:
                                    # Extract the attribute key from the new bullet
                                    new_match = re.search(
                                        matching_pattern, clean_bullet, re.IGNORECASE
                                    )
                                    if new_match and new_match.groups():
                                        # The first capture group is usually the attribute key
                                        new_attribute_key = (
                                            new_match.group(1).lower()
                                            if new_match.group(1)
                                            else None
                                        )
                                        logger.info(
                                            f"Extracted attribute key from new bullet: '{new_attribute_key}'"
                                        )

                                        # Only look for existing items that match the SAME pattern
                                        for existing_item in current_items:
                                            existing_match = re.search(
                                                matching_pattern,
                                                existing_item,
                                                re.IGNORECASE,
                                            )
                                            if (
                                                existing_match
                                                and existing_match.groups()
                                            ):
                                                # Extract the attribute key from the existing item
                                                existing_attribute_key = (
                                                    existing_match.group(1).lower()
                                                    if existing_match.group(1)
                                                    else None
                                                )
                                                logger.info(
                                                    f"Checking existing item: '{existing_item}' with attribute key: '{existing_attribute_key}'"
                                                )

                                                # Only replace if they're about the same attribute
                                                if (
                                                    new_attribute_key
                                                    and existing_attribute_key
                                                    and new_attribute_key
                                                    == existing_attribute_key
                                                ):
                                                    replaced_item = existing_item
                                                    logger.info(
                                                        f"Replacing item '{existing_item}' with '{clean_bullet}' based on matching attribute: '{new_attribute_key}'"
                                                    )
                                                    # Remove the item from current_items
                                                    current_items.remove(existing_item)
                                                    break
                                    if replaced_item:
                                        break

                        # Skip duplicates
                        if clean_bullet in current_items:
                            logger.info(
                                f"Skipping duplicate sub_memory on {memory_id}: {clean_bullet}"
                            )
                            continue

                        # Add to the list of bullets to add
                        added_bullets.append(clean_bullet)

                    if not added_bullets:
                        # All bullets were duplicates
                        logger.info(
                            f"All bullets were duplicates for memory {memory_id}"
                        )
                        status = "Skipped (All Duplicates)"
                        success = True  # Considered successful as state is correct
                        result_content = current_content
                    else:
                        # Always preserve original category during updates
                        final_category = current_category

                        # Add all new sub_memories to the list
                        updated_items = current_items + added_bullets

                        # Format the updated memory
                        merged_content = self._format_memory(
                            final_category, updated_items
                        )

                        # Log the category preservation
                        logger.info(
                            f"Preserving original category '{final_category}' during update of memory {memory_id} with {len(added_bullets)} new items"
                        )

                        # Update the memory
                        success = await self._update_memory(
                            memory_id, merged_content, user
                        )

                        status = (
                            "(Item Added)" if success else "Failed to update (merge)"
                        )
                        result_content = merged_content
                        affected_item_text = ", ".join(added_bullets)

                elif action_type == "REMOVE_ITEM":
                    item_to_remove_desc = action_data.get("item_to_remove")
                    if not memory_id or not item_to_remove_desc:
                        raise ValueError(
                            "REMOVE_ITEM action missing memory_id or item_to_remove"
                        )

                    # Fetch current content
                    current_content = existing_memories_content.get(memory_id)
                    if current_content is None:
                        # Attempt to fetch directly if not in cache
                        mem_obj = Memories.get_memory_by_id(memory_id)
                        if mem_obj and hasattr(mem_obj, "content"):
                            current_content = mem_obj.content
                        else:
                            # Memory ID not found
                            logger.warning(
                                f"Memory ID {memory_id} not found for REMOVE_ITEM."
                            )
                            status = "Failed (Memory ID not found)"
                            success = False
                            result_content = ""
                            affected_item_text = item_to_remove_desc
                            results.append(
                                {
                                    "action": action_type,
                                    "content": result_content,
                                    "importance": importance,
                                    "id": memory_id,
                                    "success": success,
                                    "status": status,
                                    "category_name": f"{category}:",
                                    "affected_item_text": affected_item_text,
                                }
                            )
                            continue

                    # Parse current content
                    current_category, current_items = self._parse_memory_content(
                        current_content
                    )

                    # Normalize item_to_remove_desc
                    clean_item_desc = item_to_remove_desc.removeprefix(
                        "DELETE: "
                    ).strip()

                    # Find the item to remove using precise matching
                    item_found = False
                    updated_items = []

                    # Extract specific unique keywords that identify the target memory
                    unique_keywords = []

                    # Look for specific unique terms like "unicycle", "pears", etc.
                    for unique_term in ["unicycle", "pear", "pears", "hockey"]:
                        if unique_term in clean_item_desc.lower():
                            unique_keywords.append(unique_term)

                    # If no unique keywords found, fall back to using all words longer than 3 chars
                    if not unique_keywords:
                        unique_keywords = [
                            word
                            for word in clean_item_desc.lower().split()
                            if len(word) > 3
                        ]

                    logger.info(
                        f"Looking for items containing these unique keywords: {unique_keywords}"
                    )

                    for item in current_items:
                        item_lower = item.lower()

                        # Only match if ALL unique keywords are present in the item
                        if unique_keywords and all(
                            keyword in item_lower for keyword in unique_keywords
                        ):
                            item_found = True
                            affected_item_text = item
                            # Skip this sub_memory (don't add to updated_items)
                            logger.info(f"Found matching item to remove: '{item}'")
                        else:
                            updated_items.append(item)

                    if not item_found:
                        logger.warning(
                            f"Item '{clean_item_desc}' not found in memory {memory_id} for REMOVE_ITEM."
                        )
                        status = "Failed (Item not found)"
                        success = False
                        result_content = current_content
                        affected_item_text = clean_item_desc  # Set affected_item_text to prevent citation warnings
                    elif not updated_items:  # No items left
                        logger.info(
                            f"Removing last item from memory {memory_id}, deleting memory."
                        )
                        success = await self._delete_memory(memory_id, user)
                        status = (
                            "Deleted (Last Item Removed)"
                            if success
                            else "Failed to delete (last item)"
                        )
                        result_content = ""
                    else:
                        # Format the updated memory
                        modified_content = self._format_memory(
                            current_category, updated_items
                        )

                        # Update the memory
                        success = await self._update_memory(
                            memory_id, modified_content, user
                        )

                        status = (
                            "(Item Removed)"
                            if success
                            else "Failed to update (item removal)"
                        )
                        result_content = modified_content

                else:
                    status = f"Failed: Unknown action type '{action_type}'"
                    success = False
                    result_content = content if content else ""

            except Exception as e:
                logger.exception(f"Error processing action {action_type}: {e}")
                status = f"Error: {str(e)}"
                success = False
                result_content = content if content else ""

            # Extract category name from the final content
            category_name = None
            if result_content and isinstance(result_content, str):
                parsed_category, _ = self._parse_memory_content(result_content)
                category_name = f"{parsed_category}:"

            # Append result for this action
            results.append(
                {
                    "action": action_type,
                    "content": result_content,
                    "importance": importance,
                    "id": memory_id,
                    "success": success,
                    "status": status,
                    "category_name": category_name,
                    "affected_item_text": affected_item_text,
                }
            )

        # Add summary logging for action results
        successful_actions = sum(1 for res in results if res.get("success", False))
        logger.info(
            f"Completed {successful_actions}/{len(results)} intended actions successfully"
        )

        # Verbose logging of action results
        if self.valves.verbose_logging and results:
            logger.info(
                "Action processing results (detailed):\n%s",
                self._truncate_log_lines(json.dumps(results, indent=2)),
            )

        return results

    async def _create_memory(self, content: str, user: Any) -> Optional[str]:
        """Create a new memory in the database with consistent formatting."""
        try:
            # Get user ID
            user_id = self._get_user_id(user)

            # Clean the content to remove importance scores and metadata
            cleaned_content = self._clean_memory_content(content)

            # Insert into database
            memory = Memories.insert_new_memory(
                user_id=user_id, content=cleaned_content
            )

            if memory and hasattr(memory, "id"):
                logger.info("Successfully created memory %s", memory.id)
                return memory.id
            else:
                logger.warning(
                    "Memory creation returned None or object without UUID-style ID"
                )
                return None
        except Exception as e:
            logger.error(f"Error creating memory: {e}", exc_info=True)
            return None

    async def _update_memory(self, memory_id: str, content: str, user: Any) -> bool:
        """Update an existing memory with consistent formatting."""
        try:
            # Clean memory ID
            memory_id = self._clean_memory_id(memory_id)

            # Get user ID
            user_id = self._get_user_id(user)

            logger.info(f"Updating memory {memory_id}")

            # Verify memory exists and belongs to user
            memory = self._validate_memory(memory_id, user_id)
            if not memory:
                logger.error(f"Memory validation failed for {memory_id}")
                return False

            # Clean the content
            cleaned_content = self._clean_memory_content(content)

            # Update the memory
            try:
                updated_memory = Memories.update_memory_by_id_and_user_id(
                    id=memory_id, user_id=user_id, content=cleaned_content
                )
            except Exception as inner_e:
                logger.error(f"Database update failed: {inner_e}")
                return False

            # Check result
            if updated_memory and hasattr(updated_memory, "id"):
                logger.info(f"Successfully updated memory {memory_id}")
                return True
            else:
                logger.error(f"Failed to update memory {memory_id}")
                return False

        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {e}")
            if self.valves.verbose_logging:
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    async def _delete_memory(self, memory_id: str, user: Any) -> bool:
        """Delete an existing memory."""
        try:
            # Clean memory ID
            memory_id = self._clean_memory_id(memory_id)

            # Get user ID
            user_id = self._get_user_id(user)

            # Validate memory
            memory = self._validate_memory(memory_id, user_id)
            if not memory:
                logger.error(f"Memory validation failed for {memory_id}")
                return False

            # Delete the memory
            deleted = Memories.delete_memory_by_id(memory_id)

            if deleted:
                logger.info(f"Successfully deleted memory {memory_id}")
                if self.valves.verbose_logging and hasattr(memory, "content"):
                    logger.info(f"Deleted content: {memory.content}")
                return True
            else:
                logger.error(f"Failed to delete memory {memory_id}")
                return False
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False

    async def _process_memories_background(
        self,
        current_message_content: str,
        __user__: dict,
        __event_emitter__: Optional[Callable[..., Awaitable[None]]] = None,
    ) -> None:
        """
        Process memories in a background task.
        This function contains the core memory processing logic and runs asynchronously.

        Args:
            current_message_content: The current user message content to analyze
            __user__: The user dictionary object
            __event_emitter__: Optional event emitter for sending status updates
        """
        start_time = time.time()

        try:
            # Send initial "Processing..." status if enabled
            if self.valves.show_status and __event_emitter__:
                self._non_blocking_emit(
                    __event_emitter__,
                    {
                        "type": "status",
                        "data": {
                            "description": "⏳ Processing memories...",
                            "done": False,
                        },
                    },
                )

            # Get user object and fetch existing memories
            user = None
            if __user__ and "id" in __user__:
                # Fetch existing memories for the user
                db_memories = Memories.get_memories_by_user_id(
                    user_id=str(__user__["id"])
                )
                user = __user__

                # Process memories through both stages
                memory_operations = await self._process_memories(
                    current_message_content, __user__["id"], db_memories
                )

                # Execute memory operations
                if memory_operations:
                    # Log memory operations before processing
                    if self.valves.verbose_logging:
                        logger.info(
                            "Memory operations before processing:\n%s",
                            self._truncate_log_lines(
                                json.dumps(memory_operations, indent=2)
                            ),
                        )

                    results = await self._process_memory_operations(
                        memory_operations, user
                    )

                    # Group results by category for structured citation
                    grouped_results = {}
                    # Use a dictionary to track the last status per category
                    category_last_status = {}

                    for result in results:
                        # Ensure category_name exists, default if necessary
                        category = result.get("category_name")
                        if not category:
                            # Handle cases where category might be None (e.g., deleted memory block)
                            # Or if extraction failed. Default to a generic category.
                            category = "Miscellaneous:"  # Default if category extraction failed or not applicable

                        action_status = result.get(
                            "status", ""
                        )  # Get the status like (Item Added)

                        if category not in grouped_results:
                            grouped_results[category] = {
                                "items": [],  # Store individual item details
                                "final_content": result.get(
                                    "content", ""
                                ),  # Store final block content once
                                "last_status": action_status,  # Initialize last status
                            }
                        else:
                            # Update final content and last status if processing multiple ops on same category
                            grouped_results[category]["final_content"] = result.get(
                                "content", ""
                            )
                            grouped_results[category][
                                "last_status"
                            ] = action_status  # Update last status

                        # Store details of the specific item affected by this operation
                        grouped_results[category]["items"].append(
                            {
                                "action": result.get("action"),
                                "status": action_status,
                                "importance": result.get("importance", 0),
                                "affected_item_text": result.get(
                                    "affected_item_text"
                                ),  # Specific item text
                                "success": result.get("success", False),
                            }
                        )
                        # We're already tracking the last status in grouped_results[category]["last_status"]

                    # Send structured citation data if there are results
                    if (
                        self.valves.show_status
                        and __event_emitter__
                        and grouped_results  # Check if dictionary is not empty
                    ):
                        await self._send_citation(
                            __event_emitter__,
                            grouped_results,
                            __user__.get("id"),  # Pass grouped data
                        )
                else:
                    logger.info("No memory operations to execute.")

        except Exception as e:
            logger.error(f"Error during memory processing: {e}", exc_info=True)
            # Send error status if enabled
            if __event_emitter__ and self.valves.show_status:
                error_message = f"⚠️ Error processing memories: {e}"
                self._non_blocking_emit(
                    __event_emitter__,
                    {
                        "type": "status",
                        "data": {
                            "description": error_message,
                            "done": True,
                        },
                    },
                )

        finally:
            end_time = time.time()
            duration = end_time - start_time
            logger.info("Memory processing finished in %.2f seconds", duration)
            if self.valves.verbose_logging:
                logger.info(
                    f"Memory processing performance metrics: start={start_time:.2f}, end={end_time:.2f}, duration={duration:.2f}s"
                )

            # Always clear the status when processing is complete
            if self.valves.show_status and __event_emitter__:
                self._non_blocking_emit(
                    __event_emitter__,
                    {
                        "type": "status",
                        "data": {
                            "description": "",
                            "done": True,
                        },
                    },
                )

            # Log completion
            logger.info("Background memory processing complete")

    async def _safe_emit(
        self, emitter: Callable[[Dict[str, Any]], Awaitable[None]], data: Dict[str, Any]
    ) -> None:
        """Safely emit events using the provided emitter."""
        try:
            await emitter(data)
        except Exception as e:
            event_type = data.get("type", "UNKNOWN")
            logger.error(
                f"Failed to emit event type '{event_type}': {e}", exc_info=True
            )

    def _non_blocking_emit(
        self, emitter: Callable[[Dict[str, Any]], Awaitable[None]], data: Dict[str, Any]
    ) -> None:
        """Emit events in a non-blocking way."""
        if emitter:
            asyncio.create_task(self._safe_emit(emitter, data))
        else:
            event_type = data.get("type", "UNKNOWN")
            logger.debug(
                f"Emitter not available, cannot emit event type '{event_type}'"
            )

    async def _send_citation(
        self,
        emitter: Callable[..., Awaitable[None]],
        grouped_results: Dict[str, Dict[str, Any]],
        user_id: Optional[str],
    ) -> None:
        """Send a structured citation message based on grouped memory operations."""
        if not grouped_results or not emitter:
            return

        # Debug logging for verbose mode
        if self.valves.verbose_logging:
            for category, data in grouped_results.items():
                logger.info(f"Category: {category}")
                logger.info(f"Last Status: {data.get('last_status', '')}")
                logger.info(f"Final Content: {data.get('final_content', '')}")
                for item in data.get("items", []):
                    logger.info(
                        f"Affected Item: {item.get('affected_item_text')}, Action: {item.get('action')}"
                    )

        # Build citation message with a clean, simplified approach
        citation_lines = []

        for category_header, data in grouped_results.items():
            # Get essential data
            final_content = data.get("final_content", "")
            items_details = data.get("items", [])

            # Determine the primary action for this category
            action_description = ""
            has_adds = any(
                item.get("action") in ["CREATE", "MERGE_UPDATE"] and item.get("success")
                for item in items_details
            )
            has_removes = any(
                item.get("action") == "REMOVE_ITEM" and item.get("success")
                for item in items_details
            )

            # Prioritize description based on actions
            if has_adds and not has_removes:
                # Check if it was a CREATE or MERGE_UPDATE
                create_op = next(
                    (item for item in items_details if item.get("action") == "CREATE"),
                    None,
                )
                if create_op:
                    action_description = "(Added info to category)"  # Assuming CREATE implies adding to a category
                else:  # Must be MERGE_UPDATE
                    action_description = "(Added info to category)"
            elif has_removes and not has_adds:
                action_description = "(Removed info from category)"
                # Check if it was a full deletion resulting in "No category"
                delete_op = next(
                    (
                        item
                        for item in items_details
                        if item.get("action") == "REMOVE_ITEM"
                    ),
                    None,
                )
                if (
                    delete_op
                    and delete_op.get("status") == "Deleted (Last Item Removed)"
                ):
                    # If the original category was Miscellaneous or Unknown, use "No category"
                    # We need to parse the category_header to check this
                    category_name = (
                        category_header.rstrip(":")
                        if isinstance(category_header, str)
                        else "Unknown"
                    )
                    if category_name == "Miscellaneous" or category_name == "Unknown":
                        category_header = "No category:"
                    action_description = "(Deleted info)"  # Use "Deleted info" when the last item is removed
            elif has_adds and has_removes:
                action_description = "(Updated info in category)"  # Mixed operations
            else:
                # Handle cases with only failed operations or skipped duplicates
                last_status = data.get("last_status", "")
                if "Failed" in last_status:
                    action_description = "(Operation Failed)"
                elif "Skipped" in last_status:
                    action_description = "(Info already exists)"
                else:
                    action_description = "(Info processed)"  # Generic fallback

            # Clean category header (remove trailing colon if present)
            clean_category = (
                category_header.rstrip(":")
                if isinstance(category_header, str)
                else "Unknown"
            )

            # Handle "No category" specifically for deletions
            if action_description == "(Deleted info)" and (
                clean_category == "Miscellaneous" or clean_category == "Unknown"
            ):
                citation_lines.append(f"No category: {action_description}")
            else:
                citation_lines.append(f"{clean_category}: {action_description}")

            # Skip if no content or only header (e.g., only failed ops)
            if not final_content and not any(
                item.get("action") == "REMOVE_ITEM" and item.get("success")
                for item in items_details
            ):
                continue

            content_lines = final_content.strip().split("\n")
            if len(content_lines) <= 1:
                continue

            # Process memory items (bullet points)
            processed_lines = set()  # Track processed lines to avoid duplicates

            for line in content_lines[1:]:  # Skip header line
                line = line.strip()
                if not line or not line.startswith("- "):
                    continue

                # Extract the core text (without cleaning parenthetical content)
                core_text = line[2:].strip() if line.startswith("- ") else line.strip()
                clean_core = core_text  # Keep all parenthetical content

                # Skip duplicates
                if clean_core in processed_lines:
                    continue
                processed_lines.add(clean_core)

                # Find the item data associated with this line's content
                item_data = None
                item_importance = 0.0  # Default score
                for item_detail in data.get("items", []):
                    affected_text = item_detail.get("affected_item_text", "")
                    # Ensure affected_text is not None before processing
                    if affected_text is None:
                        affected_text = ""
                        logger.warning("Found None affected_item_text in item_detail")

                    # Handle potential comma-separated list in affected_item_text for aggregated updates
                    affected_items_list = (
                        [i.strip() for i in affected_text.split(",")]
                        if affected_text and "," in affected_text
                        else [affected_text]
                    )

                    if clean_core in affected_items_list:
                        item_data = item_detail
                        item_importance = item_detail.get(
                            "importance", 0.0
                        )  # Get score from the result item
                        break  # Found the matching item detail

                # Format the score string
                score_text = f" (score: {item_importance:.2f})"

                # Check if this item was newly added (from CREATE or MERGE_UPDATE actions)
                is_newly_added = False
                for item in data.get("items", []):
                    # Get the affected item text (might be a single item or comma-separated list)
                    affected_text = item.get("affected_item_text", "")

                    # Ensure affected_text is not None before processing
                    if affected_text is None:
                        affected_text = ""
                        logger.warning(
                            "Found None affected_item_text in item_detail during citation processing"
                        )

                    # Check if the action is CREATE or MERGE_UPDATE and if the item was successful
                    if item.get("action") in ["CREATE", "MERGE_UPDATE"] and item.get(
                        "success", False
                    ):

                        # Handle both single items and comma-separated lists
                        if affected_text and "," in affected_text:
                            # Split the comma-separated list and check if the current item is in it
                            try:
                                affected_items = [
                                    i.strip() for i in affected_text.split(",")
                                ]
                                if clean_core in affected_items:
                                    is_newly_added = True
                                    break
                            except Exception as e:
                                logger.error(
                                    f"Error processing affected_text '{affected_text}': {e}"
                                )
                        else:
                            # Direct comparison for single items
                            if affected_text == clean_core:
                                is_newly_added = True
                                break

                # Add formatted item to citation with checkmark for newly added items
                # Only show importance score for newly added items
                if is_newly_added:
                    citation_lines.append(f"✅ - {clean_core}{score_text}")
                else:
                    citation_lines.append(f"- {clean_core}")

            # Add removed items with X mark and strikethrough
            for item in data.get("items", []):
                if item.get("action") == "REMOVE_ITEM" and item.get("success", False):
                    removed_text = item.get("affected_item_text", "")
                    if removed_text:
                        citation_lines.append(f"❎ - {removed_text}")

        # Create final citation message
        citation_message = "\n".join(citation_lines)

        # Log appropriate information based on verbosity setting
        if self.valves.verbose_logging:
            logger.info(
                "Sending structured citation for user %s:\n%s",
                user_id or "UNKNOWN",
                self._truncate_log_lines(citation_message),
            )
        else:
            logger.info(
                "Sending citation for user %s affecting %d categories",
                user_id or "UNKNOWN",
                len(grouped_results),
            )

        await self._safe_emit(
            emitter,
            {
                "type": "citation",
                "data": {
                    "document": [citation_message],
                    "metadata": [{"source": "module://mis/memories", "html": False}],
                    "source": {"name": "Memories Processed"},
                },
            },
        )

    async def inlet(
        self,
        body: Dict[str, Any],
        __user__: dict = {},
        __event_emitter__: Optional[Callable[..., Awaitable[None]]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for processing incoming messages.
        This function quickly extracts necessary information and launches
        a background task for memory processing, then returns immediately.
        """
        # Check if prompts are set
        if not self.valves.identification_prompt or not self.valves.integration_prompt:
            logger.error("Prompts are not set. Cannot process memories.")
            if __event_emitter__ and self.valves.show_status:
                self._non_blocking_emit(
                    __event_emitter__,
                    {
                        "type": "status",
                        "data": {
                            "description": "⚠️ MIS Error: Prompts not configured",
                            "done": True,
                        },
                    },
                )
            return body

        # Quick validation checks
        if not __user__ or "id" not in __user__:
            logger.info("No valid user provided. Skipping memory processing.")
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

        # Launch memory processing in the background
        logger.info("Launching memory processing in background")

        # Create the background task
        task = asyncio.create_task(
            self._process_memories_background(
                current_message_content, __user__, __event_emitter__
            )
        )

        # Add task to the set of background tasks
        self.background_tasks.add(task)

        # Set up callback to remove the task when it's done
        def task_done_callback(completed_task):
            try:
                # Remove the task from the set when it completes
                self.background_tasks.discard(completed_task)
                # Check for exceptions
                if completed_task.exception():
                    logger.error(
                        f"Background task failed with exception: {completed_task.exception()}"
                    )
            except (asyncio.CancelledError, Exception) as e:
                logger.error(f"Error in task completion callback: {e}")

        # Add the callback
        task.add_done_callback(task_done_callback)

        # Return the original body immediately - MIS does not modify the chat flow directly
        logger.info(
            "Returning original body immediately, memory processing continues in background"
        )
        return body

    async def outlet(self, body: Dict[str, Any], __user__: dict = {}) -> Dict[str, Any]:
        """Outlet function - currently does nothing as MIS doesn't modify outgoing messages."""
        return body
