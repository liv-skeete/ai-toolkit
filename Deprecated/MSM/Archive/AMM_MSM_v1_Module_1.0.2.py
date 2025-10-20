"""
title: AMM_Memory_Summarization_Management
description: Memory Summarization & Management Module for Open WebUI - On-demand memory management companion to MRS_v1
author: Cody
version: 1.0.2
date: 2025-05-21
changes:
- v1.0.2 (2025-05-21):
  - Made module completely tag-agnostic
  - Removed all tag parsing and manipulation logic
  - Simplified memory operations to treat content as opaque strings
  - Delegated all tag handling responsibility to the LLM through the prompt
  - Removed _extract_tags method and all tag-specific logic
- v1.0.1 (2025-05-21):
  - Updated memory format to align with MRS module (tags prepended to content field)
  - Added support for memory merging and splitting operations
  - Improved tag handling to support up to 3 tags (primary, secondary, tertiary)
  - Simplified memory formatting logic to let the LLM handle most formatting
  - Replaced category extraction with more flexible tag extraction
- v1.0.0 (2025-05-21):
  - Initial implementation of Memory Summarization & Management module
  - Implemented core memory management operations (delete, update, tag, deduplicate)
  - Added support for user-triggered memory management via natural language commands
  - Integrated with MRS_v1 module for consistent memory formatting and API patterns
  - Implemented comprehensive error handling and logging throughout
"""

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Set, Tuple

import aiohttp
from open_webui.models.memories import Memories
from open_webui.models.users import Users
from pydantic import BaseModel, Field

# Logger configuration
logger = logging.getLogger("amm_msm")
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
    """Memory Summarization & Management module for on-demand memory operations."""

    class Valves(BaseModel):
        # Enable/Disable Function
        enabled: bool = Field(
            default=True,
            description="Enable memory summarization & management",
        )
        # Set processing priority
        priority: int = Field(
            default=5,  # Lower priority than MRS (2), MIS (3), and MMC (4)
            description="Priority level for the filter operations.",
        )
        # UI settings
        show_status: bool = Field(
            default=True, description="Show memory operation status in chat"
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
        max_tokens: int = Field(
            default=750,
            description="Maximum tokens for API calls",
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
        
        # Memory management settings
        trigger_phrases: List[str] = Field(
            default=[
                "delete memories",
                "update memory",
                "tag memories",
                "deduplicate memories",
                "remove trivial memories",
                "summarize memories",
                "merge memories",
                "split memory",
                "organize my memories",
                "manage my memories",
            ],
            description="Phrases that trigger memory management operations",
        )
        similarity_threshold: float = Field(
            default=0.85,
            description="Threshold for memory similarity comparison (0.0-1.0)",
        )
        trivial_memory_threshold: float = Field(
            default=0.3,
            description="Threshold for identifying trivial memories (0.0-1.0)",
        )
        memory_management_prompt: str = Field(
            ...,  # This makes the field required with no default
            description="Memory management prompt with preserved formatting",
        )
    def __init__(self) -> None:
        """
        Initialize the Memory Summarization & Management module.
        """
        logger.info("Initializing Memory Summarization & Management module")

        # Initialize with empty prompt - must be set via update_valves
        try:
            self.valves = self.Valves(
                memory_management_prompt="",  # Empty string to start - must be set via update_valves
            )
            logger.warning(
                "Memory management prompt is empty - module will not function until prompt is set"
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

        # Track background tasks for memory operations
        self.background_tasks = set()

        logger.info(
            "MSM module initialized with API provider: %s", self.valves.api_provider
        )

    async def close(self) -> None:
        """Close the aiohttp session and wait for any pending background tasks."""
        logger.info("Closing MSM module session")

        # Wait for any pending background tasks to complete
        if self.background_tasks:
            logger.info(
                f"Waiting for {len(self.background_tasks)} background tasks to complete"
            )
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

    def _is_management_request(self, message: str) -> bool:
        """
        Determine if a message is requesting memory management.

        Args:
            message: The user message

        Returns:
            True if the message is requesting memory management, False otherwise
        """
        message_lower = message.lower()
        return any(
            keyword in message_lower for keyword in self.valves.trigger_phrases
        )
    async def _identify_target_memories(
        self, query: str, db_memories: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify memories matching a user query.

        Args:
            query: The user query
            db_memories: List of memories from the database

        Returns:
            List of dictionaries containing matching memories with their IDs
        """
        if not db_memories:
            return []

        # Simple implementation: use string matching for now
        # In a real implementation, this would use semantic search or LLM-based matching
        query_lower = query.lower()
        matching_memories = []

        for memory in db_memories:
            if hasattr(memory, "id") and hasattr(memory, "content") and memory.content:
                if query_lower in memory.content.lower():
                    matching_memories.append({
                        "id": memory.id,
                        "content": memory.content
                    })

        return matching_memories

    # _extract_tags method removed - tag handling is now the LLM's responsibility

    def _calculate_memory_similarity(self, memory1: str, memory2: str) -> float:
        """
        Calculate similarity between two memories.

        Args:
            memory1: First memory content
            memory2: Second memory content

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Simple implementation: use string similarity
        # In a real implementation, this would use semantic similarity
        
        # Normalize text
        mem1 = memory1.lower().strip()
        mem2 = memory2.lower().strip()
        
        # Calculate Jaccard similarity
        tokens1 = set(mem1.split())
        tokens2 = set(mem2.split())
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        if union == 0:
            return 0.0
            
        return intersection / union

    def _format_memories_for_display(self, memories: List[Dict[str, Any]]) -> str:
        """
        Format memories for display to the user.

        Args:
            memories: List of memory dictionaries

        Returns:
            Formatted string of memories
        """
        if not memories:
            return "No memories found."
            
        formatted = []
        for memory in memories:
            formatted.append(f"• {memory['content']} (ID: {memory['id']})")
            
        return "\n".join(formatted)

    def _format_memories_for_llm(self, memories: List[Any]) -> str:
        """
        Format memories for LLM processing.

        Args:
            memories: List of memory objects

        Returns:
            Formatted string of memories
        """
        if not memories:
            return "No existing memories."

        # Format memories as a numbered list with clear separation between content and ID
        formatted = ["Available memories:"]
        for i, mem in enumerate(memories, 1):
            if hasattr(mem, "id") and hasattr(mem, "content") and mem.content:
                formatted.append(f"{i}. [ID: {mem.id}] {mem.content}")

        return "\n".join(formatted)

    def _format_memory_item(self, content: str, score: float = None, score_label: str = "score") -> str:
        """
        Format a single memory item with consistent formatting.
        
        Args:
            content: The memory content
            score: Optional score value
            score_label: Label for the score
            
        Returns:
            Formatted memory string with consistent formatting
        """
        if score is not None:
            return f"• {content} ({score_label}: {score:.2f})"
        return f"• {content}"

    def _parse_llm_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse the JSON response from the LLM.

        Args:
            response_text: The raw response text from the LLM

        Returns:
            List of dictionaries containing memory operations
        """
        try:
            # Extract JSON from the response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response_text.strip()

            # Clean up the JSON string
            json_str = re.sub(r'```json\s*|\s*```', '', json_str)
            
            # Parse the JSON
            memory_operations = json.loads(json_str)
            
            # Validate the structure
            if not isinstance(memory_operations, list):
                logger.error("Invalid response format: not a list")
                return []
                
            # Validate each operation
            valid_operations = []
            for op in memory_operations:
                if not isinstance(op, dict):
                    logger.warning(f"Invalid operation format: {op}")
                    continue
                
                if "operation" not in op:
                    logger.warning(f"Missing operation field: {op}")
                    continue
                    
                # Add more validation as needed
                valid_operations.append(op)
                
            return valid_operations
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            if self.valves.verbose_logging:
                logger.error(f"Raw response: {response_text}")
            return []
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return []
    async def _delete_memories_by_query(
        self, query: str, user: Any, db_memories: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Delete memories based on a user query.

        Args:
            query: The user query
            user: The user object
            db_memories: List of memories from the database

        Returns:
            List of dictionaries containing deleted memory information
        """
        # Identify target memories
        target_memories = await self._identify_target_memories(query, db_memories)
        
        if not target_memories:
            logger.info(f"No memories found matching query: {query}")
            return []
            
        # Delete the memories
        deleted_memories = []
        for memory in target_memories:
            success = await self._delete_memory(memory["id"], user)
            if success:
                deleted_memories.append(memory)
                
        return deleted_memories

    async def _update_memory_by_query(
        self, query: str, new_content: str, user: Any, db_memories: List[Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Update a specific memory based on a query.

        Args:
            query: The user query
            new_content: The new memory content
            user: The user object
            db_memories: List of memories from the database

        Returns:
            Dictionary containing updated memory information, or None if update failed
        """
        # Identify target memories
        target_memories = await self._identify_target_memories(query, db_memories)
        
        if not target_memories:
            logger.info(f"No memories found matching query: {query}")
            return None
            
        # Update the first matching memory
        target = target_memories[0]
        success = await self._update_memory(target["id"], new_content, user)
        
        if success:
            return {
                "id": target["id"],
                "old_content": target["content"],
                "new_content": new_content
            }
            
        return None

    # _tag_memories method removed - tag operations are now handled exactly like UPDATE operations

    # Method removed as part of alignment with MRS module

    # Method removed as part of alignment with MRS module

    async def _deduplicate_memories(
        self, user: Any, db_memories: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate memories.

        Args:
            user: The user object
            db_memories: List of memories from the database

        Returns:
            List of dictionaries containing duplicate memory information
        """
        # Find duplicate memories
        content_map = {}
        duplicates = []
        
        for memory in db_memories:
            if hasattr(memory, "content") and memory.content:
                content = memory.content.strip()
                if content in content_map:
                    # This is a duplicate
                    duplicates.append({
                        "id": memory.id,
                        "content": content,
                        "duplicate_of": content_map[content]
                    })
                else:
                    content_map[content] = memory.id
                    
        # Delete duplicate memories
        deleted_duplicates = []
        for duplicate in duplicates:
            success = await self._delete_memory(duplicate["id"], user)
            if success:
                deleted_duplicates.append(duplicate)
                
        return deleted_duplicates

    async def _delete_trivial_memories(
        self, user: Any, db_memories: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Remove low-importance memories.

        Args:
            user: The user object
            db_memories: List of memories from the database

        Returns:
            List of dictionaries containing deleted trivial memory information
        """
        # In a real implementation, this would use an LLM to score memory importance
        # The LLM is now responsible for determining which memories are trivial
        # This method is kept for backward compatibility but doesn't implement
        # any tag-specific logic
        
        # This is now a placeholder - actual trivial memory identification
        # should be done by the LLM through the prompt
        logger.info("Trivial memory identification is now handled by the LLM")
        return []

    async def _summarize_and_delete_old_memories(
        self, query: str, user: Any, db_memories: List[Any]
    ) -> Dict[str, Any]:
        """
        Create a summary of old memories and remove the originals.

        Args:
            query: The user query to identify memories to summarize
            user: The user object
            db_memories: List of memories from the database

        Returns:
            Dictionary containing summary information
        """
        # Identify target memories
        target_memories = await self._identify_target_memories(query, db_memories)
        
        if not target_memories:
            logger.info(f"No memories found matching query: {query}")
            return {"summary": None, "deleted": []}
            
        # Create a summary
        # In a real implementation, this would use an LLM to generate a summary
        summary_content = f"Summary of {len(target_memories)} memories about {query}"
        
        # Create the summary memory
        # In a real implementation, this would create a new memory in the database
        summary = {
            "content": summary_content,
            "summarized_count": len(target_memories)
        }
        
        # Delete the original memories
        deleted_memories = []
        for memory in target_memories:
            success = await self._delete_memory(memory["id"], user)
            if success:
                deleted_memories.append(memory)
                
        return {
            "summary": summary,
            "deleted": deleted_memories
        }
    async def _execute_memory_operations(
        self, operations: List[Dict[str, Any]], user: Any, db_memories: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute memory operations.

        Args:
            operations: List of memory operations
            user: The user object
            db_memories: List of memories from the database

        Returns:
            List of dictionaries containing operation results
        """
        results = []
        
        for op in operations:
            operation_type = op.get("operation", "").upper()
            
            try:
                if operation_type == "ADD":
                    # Create a new memory
                    content = op.get("content", "")
                    memory_id = await self._create_memory(content, user)
                    if memory_id:
                        results.append({
                            "operation": "ADD",
                            "status": "success",
                            "memory_id": memory_id,
                            "content": content
                        })
                    else:
                        results.append({
                            "operation": "ADD",
                            "status": "failed",
                            "content": content
                        })
                        
                elif operation_type == "DELETE":
                    # Delete a memory
                    memory_id = op.get("memory_id", "")
                    content = op.get("content", "")
                    success = await self._delete_memory(memory_id, user)
                    results.append({
                        "operation": "DELETE",
                        "status": "success" if success else "failed",
                        "memory_id": memory_id,
                        "content": content
                    })
                    
                elif operation_type == "UPDATE":
                    # Update a memory
                    memory_id = op.get("memory_id", "")
                    old_content = op.get("old_content", "")
                    new_content = op.get("new_content", "")
                    success = await self._update_memory(memory_id, new_content, user)
                    results.append({
                        "operation": "UPDATE",
                        "status": "success" if success else "failed",
                        "memory_id": memory_id,
                        "old_content": old_content,
                        "new_content": new_content
                    })
                    
                # TAG operation handling removed - use UPDATE instead
                
                elif operation_type == "MERGE":
                    try:
                        # Merge multiple memories into one
                        source_memory_ids = [self._clean_memory_id(mid) for mid in op.get("source_memory_ids", [])]
                        new_content = op.get("new_content", "")
                        
                        # Get user ID consistently
                        user_id = self._get_user_id(user)
                        if not user_id:
                            logger.error("Invalid user object - cannot validate memories")
                            results.append({
                                "operation": "MERGE",
                                "status": "failed",
                                "reason": "Invalid user object - cannot validate memories"
                            })
                            continue
                        
                        # Validate all source memories
                        valid_memory_ids = []
                        for memory_id in source_memory_ids:
                            memory = await self._validate_memory(memory_id, user_id)
                            if memory:
                                valid_memory_ids.append(memory_id)
                            else:
                                logger.error(f"Memory validation failed for ID: {memory_id}")
                    except Exception as e:
                        logger.error(f"Error in MERGE operation setup: {e}")
                        results.append({
                            "operation": "MERGE",
                            "status": "failed",
                            "reason": f"Error in MERGE operation setup: {str(e)}"
                        })
                        continue
                    
                    try:
                        if len(valid_memory_ids) < 2:
                            results.append({
                                "operation": "MERGE",
                                "status": "failed",
                                "reason": "Need at least two valid memories to merge"
                            })
                            continue
                        
                        # Create new merged memory
                        merged_memory_id = await self._create_memory(new_content, user)
                        
                        if not merged_memory_id:
                            results.append({
                                "operation": "MERGE",
                                "status": "failed",
                                "reason": "Failed to create merged memory"
                            })
                            continue
                        
                        # Delete source memories
                        deleted_ids = []
                        for memory_id in valid_memory_ids:
                            try:
                                success = await self._delete_memory(memory_id, user)
                                if success:
                                    deleted_ids.append(memory_id)
                                else:
                                    logger.error(f"Failed to delete source memory: {memory_id}")
                            except Exception as e:
                                logger.error(f"Error deleting source memory {memory_id}: {e}")
                        
                        results.append({
                            "operation": "MERGE",
                            "status": "success",
                            "memory_id": merged_memory_id,
                            "content": new_content,
                            "source_memory_ids": valid_memory_ids,
                            "deleted_source_ids": deleted_ids
                        })
                    except Exception as e:
                        logger.error(f"Error in MERGE operation execution: {e}")
                        results.append({
                            "operation": "MERGE",
                            "status": "failed",
                            "reason": f"Error in MERGE operation execution: {str(e)}"
                        })
                
                elif operation_type == "SPLIT":
                    # Split a memory into multiple memories
                    source_memory_id = op.get("source_memory_id", "")
                    new_contents = op.get("new_contents", [])
                    
                    # Get user ID consistently
                    user_id = self._get_user_id(user)
                    if not user_id:
                        logger.error("Invalid user object - cannot validate memories")
                        results.append({
                            "operation": "SPLIT",
                            "status": "failed",
                            "reason": "Invalid user object - cannot validate memories"
                        })
                        continue
                    
                    # Clean memory ID
                    source_memory_id = self._clean_memory_id(source_memory_id)
                    
                    # Validate source memory
                    source_memory = await self._validate_memory(source_memory_id, user_id)
                    if not source_memory:
                        results.append({
                            "operation": "SPLIT",
                            "status": "failed",
                            "reason": "Source memory not found or not owned by user"
                        })
                        continue
                    
                    if not new_contents:
                        results.append({
                            "operation": "SPLIT",
                            "status": "failed",
                            "reason": "No new contents provided for split operation"
                        })
                        continue
                    
                    # Create new memories
                    new_memory_ids = []
                    for content in new_contents:
                        memory_id = await self._create_memory(content, user)
                        if memory_id:
                            new_memory_ids.append(memory_id)
                    
                    # Delete source memory if we created at least one new memory
                    if new_memory_ids:
                        deleted = await self._delete_memory(source_memory_id, user)
                    else:
                        deleted = False
                    
                    results.append({
                        "operation": "SPLIT",
                        "status": "success" if new_memory_ids and deleted else "partial" if new_memory_ids else "failed",
                        "source_memory_id": source_memory_id,
                        "source_deleted": deleted,
                        "new_memory_ids": new_memory_ids,
                        "new_contents": new_contents
                    })
                    
                else:
                    logger.warning(f"Unknown operation type: {operation_type}")
                    results.append({
                        "operation": operation_type,
                        "status": "failed",
                        "reason": "Unknown operation type"
                    })
                    
            except Exception as e:
                logger.error(f"Error executing operation {operation_type}: {e}")
                results.append({
                    "operation": operation_type,
                    "status": "failed",
                    "reason": str(e)
                })
                
        return results

    async def _create_memory(self, content: str, user: Any) -> Optional[str]:
        """
        Create a new memory in the database.

        Args:
            content: The memory content
            user: The user object

        Returns:
            Memory ID if successful, None otherwise
        """
        try:
            # Get user ID using helper method
            user_id = self._get_user_id(user)
            if not user_id:
                logger.error("Invalid user object - cannot create memory")
                return None

            # Create the memory - remove await as Memories methods are not async
            memory = Memories.insert_new_memory(
                user_id=user_id,
                content=content,
            )
            
            return memory.id
            
        except Exception as e:
            logger.error(f"Error creating memory: {e}")
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
            # Get user ID using helper method
            user_id = self._get_user_id(user)
            if not user_id:
                logger.error("Invalid user object - cannot update memory")
                return False
                
            # Clean memory ID
            memory_id = self._clean_memory_id(memory_id)

            # Validate memory ownership
            memory = await self._validate_memory(memory_id, user_id)
            if not memory:
                logger.error(f"Memory {memory_id} not found or not owned by user {user_id}")
                return False

            # Update the memory - remove await as Memories methods are not async
            updated_memory = Memories.update_memory_by_id_and_user_id(
                id=memory_id,
                user_id=user_id,
                content=content
            )
            if not updated_memory:
                logger.error(f"Failed to update memory {memory_id}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
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
            # Get user ID using helper method
            user_id = self._get_user_id(user)
            if not user_id:
                logger.error("Invalid user object - cannot delete memory")
                return False
                
            # Clean memory ID
            memory_id = self._clean_memory_id(memory_id)

            # Validate memory ownership
            memory = await self._validate_memory(memory_id, user_id)
            if not memory:
                logger.error(f"Memory {memory_id} not found or not owned by user {user_id}")
                return False

            # Delete the memory - remove await as Memories methods are not async
            success = Memories.delete_memory_by_id_and_user_id(
                id=memory_id,
                user_id=user_id
            )
            if not success:
                logger.error(f"Failed to delete memory {memory_id}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return False
    async def _process_memory_management(
        self, user_message: str, user: Any, db_memories: List[Any]
    ) -> Dict[str, Any]:
        """
        Process memory management operations based on user request.

        Args:
            user_message: The user message
            user: The user object
            db_memories: List of memories from the database

        Returns:
            Dictionary containing operation results
        """
        logger.info("Processing memory management request")
        
        # DIAGNOSTIC: Log current API provider setting
        logger.info(f"Current API provider setting: {self.valves.api_provider}")

        # Format memories for LLM
        formatted_memories = self._format_memories_for_llm(db_memories)

        # Check if prompt is empty and fail fast
        if not self.valves.memory_management_prompt:
            logger.error("Memory management prompt is empty - cannot process request")
            raise ValueError("Memory management prompt is empty - module cannot function")

        # Format the system prompt using the valve-stored prompt
        system_prompt = self.valves.memory_management_prompt.format(
            user_message=user_message,
            existing_memories=formatted_memories,
        )

        # Verbose logging for prompt
        if self.valves.verbose_logging:
            truncated_prompt = self._truncate_log_lines(system_prompt)
            logger.info(f"System prompt (truncated):\n{truncated_prompt}")

        # Query the appropriate API
        if self.valves.api_provider == "OpenAI API":
            response = await self.query_openai_api(system_prompt, user_message)
        else:  # Ollama API
            response = await self.query_ollama_api(system_prompt, user_message)

        # Verbose logging for response
        if self.valves.verbose_logging:
            truncated_response = self._truncate_log_lines(response)
            logger.info(f"Raw API response: {truncated_response}")

        # Parse the response
        memory_operations = self._parse_llm_response(response)

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
            logger.info(f"Memory operations determined: {summary}")
        else:
            logger.info("No memory operations determined")

        # Execute the operations
        results = await self._execute_memory_operations(memory_operations, user, db_memories)

        return {
            "operations": memory_operations,
            "results": results
        }

    def _get_user_id(self, user: Any) -> Optional[str]:
        """
        Extract user ID from either a dictionary or object representation.
        
        Args:
            user: The user object or dictionary
            
        Returns:
            User ID as string, or None if not found
        """
        if isinstance(user, dict) and "id" in user:
            return str(user["id"])
        elif hasattr(user, "id"):
            return str(user.id)
        return None
    
    def _clean_memory_id(self, memory_id: str) -> str:
        """
        Clean memory ID consistently from various formats.
        
        Args:
            memory_id: The memory ID to clean
            
        Returns:
            Cleaned memory ID
        """
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
        
    async def _validate_memory(self, memory_id: str, user_id: str) -> Optional[Any]:
        """
        Validate memory existence and ownership.

        Args:
            memory_id: The memory ID
            user_id: The user ID

        Returns:
            Memory object if valid, None otherwise
        """
        try:
            # Clean memory ID using helper function
            memory_id = self._clean_memory_id(memory_id)
            
            # Query the memory - remove await as Memories methods are not async
            memory = Memories.get_memory_by_id(memory_id)
            
            if not memory:
                logger.warning(f"Memory {memory_id} not found")
                return None
            
            # First, ensure consistent access to memory attributes regardless of type
            # This prevents "'dict' object has no attribute 'id'" errors
            if isinstance(memory, dict):
                # Create a wrapper class that provides attribute access to dictionary
                class MemoryWrapper:
                    def __init__(self, memory_dict):
                        self.__dict__ = memory_dict
                
                # Wrap the dictionary before further processing
                memory = MemoryWrapper(memory)
            
            # Now validate ownership using attribute access
            if str(memory.user_id) != str(user_id):
                logger.warning(f"Memory {memory_id} does not belong to user {user_id}")
                return None
            
            return memory
            
        except Exception as e:
            logger.error(f"Error validating memory {memory_id}: {e}")
            return None

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

    async def _safe_emit(
        self,
        emitter: Callable[[Any], Awaitable[None]],
        data: Dict[str, Any],
    ) -> None:
        """
        Safely emit an event, handling missing emitter.

        Args:
            emitter: The event emitter function
            data: The data to emit
        """
        if not emitter:
            logger.debug("Event emitter not available")
            return

        try:
            await emitter(data)
        except Exception as e:
            logger.error(f"Error in event emitter: {e}")

    async def _send_citation(
        self,
        emitter: Callable[..., Awaitable[None]],
        status_updates: List[str],
        user_id: Optional[str] = None,
        citation_type: str = "managed",
    ) -> None:
        """
        Send a citation message with memory information.
        
        Args:
            emitter: The event emitter function
            status_updates: List of status update strings
            user_id: The user ID
            citation_type: Type of citation ("managed" by default)
        """
        if not status_updates or not emitter:
            return

        # Determine citation title and source
        title = "Memories Managed"
        source_path = "module://msm/memories/managed"
        header = "Memory Management Operations:"

        # Format the citation message
        citation_message = f"{header}\n" + "\n".join(status_updates)
        
        logger.info(
            "Sending %s citation for user %s", citation_type, user_id or "UNKNOWN"
        )

        await self._safe_emit(
            emitter,
            {
                "type": "citation",
                "data": {
                    "document": [citation_message],
                    "metadata": [
                        {"source": source_path, "html": False}
                    ],
                    "source": {"name": title},
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
        Handles memory management requests based on trigger phrases.

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

            # Check if this is a memory management request
            if not self._is_management_request(last_user_message):
                return body

            # Emit status update if enabled
            if self.valves.show_status and __event_emitter__:
                if self.valves.verbose_logging:
                    logger.info("Emitting status: Processing memory management request...")
                await self._safe_emit(
                    __event_emitter__,
                    {
                        "type": "status",
                        "data": {
                            "description": "🧠 Processing memory management request...",
                            "done": False,
                        },
                    },
                )

            # Get user memories - add logging for memory object structure
            logger.info(f"Getting memories for user {__user__['id']} using Memories.get_memories_by_user_id")
            db_memories = Memories.get_memories_by_user_id(__user__["id"])
            
            # Log memory structure for debugging
            if db_memories and len(db_memories) > 0:
                sample_memory = db_memories[0]
                logger.info(f"Sample memory type: {type(sample_memory)}")
                logger.info(f"Sample memory attributes: {dir(sample_memory)}")
            if not db_memories:
                logger.info("No memories found for user %s - nothing to manage", __user__["id"])
                
                # Emit completion status if enabled
                if self.valves.show_status and __event_emitter__:
                    await self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": "☒ No memories found to manage",
                                "done": True,
                            },
                        },
                    )
                    
                    # Create a background task to clear the status after delay
                    asyncio.create_task(self._delayed_clear_status(__event_emitter__))
                    
                return body

            # Process memory management
            operation_results = await self._process_memory_management(
                last_user_message, __user__, db_memories
            )
            
            # Format operation results for citation
            status_updates = []
            for op in operation_results.get("operations", []):
                op_type = op.get("operation", "UNKNOWN").upper()
                
                # Special handling for SPLIT operations
                if op_type == "SPLIT":
                    # Format source content with ☒ symbol
                    source_content = op.get("source_content", "Unknown source")
                    split_details = [f"☒ [Misc] {source_content}"]
                    
                    # Format new contents with ☑ symbol
                    for content in op.get("new_contents", []):
                        split_details.append(f"☑ {content}")
                    
                    # Join all details with newlines and proper indentation
                    formatted_details = "\n".join(split_details)
                    status_updates.append(f"• {op_type}: \n{formatted_details}")
                
                # Special handling for MERGE operations
                elif op_type == "MERGE":
                    # Get source memory IDs and new content
                    source_ids = op.get("source_memory_ids", [])
                    new_content = op.get("new_content", "Unknown merged content")
                    
                    # Format the details
                    merge_details = [f"☑ {new_content}"]
                    for source_id in source_ids:
                        merge_details.append(f"☒ Source ID: {source_id}")
                    
                    # Join all details with newlines and proper indentation
                    formatted_details = "\n".join(merge_details)
                    status_updates.append(f"• {op_type}: \n{formatted_details}")
                
                # Special handling for UPDATE operations
                elif op_type == "UPDATE":
                    # Get memory ID, old content, and new content
                    memory_id = op.get("memory_id", "Unknown ID")
                    old_content = op.get("old_content", "Unknown old content")
                    new_content = op.get("new_content", "Unknown new content")
                    
                    # Format the details
                    update_details = [
                        f"☒ {old_content}",
                        f"☑ {new_content}"
                    ]
                    
                    # Join all details with newlines and proper indentation
                    formatted_details = "\n".join(update_details)
                    status_updates.append(f"• {op_type}: \n{formatted_details}")
                
                # Special handling for ADD operations
                elif op_type == "ADD":
                    # Get content
                    content = op.get("content", "Unknown content")
                    
                    # Format the details
                    formatted_details = f"☑ {content}"
                    status_updates.append(f"• {op_type}: \n{formatted_details}")
                
                # Special handling for DELETE operations
                elif op_type == "DELETE":
                    # Get content
                    content = op.get("content", "Unknown content")
                    
                    # Format the details
                    formatted_details = f"☒ {content}"
                    status_updates.append(f"• {op_type}: \n{formatted_details}")
                
                else:
                    # Default handling for other operation types
                    details = op.get("details", "No details")
                    status_updates.append(f"• {op_type}: {details}")
            
            # Send citation with memory operations
            if __event_emitter__ and status_updates:
                await self._send_citation(
                    __event_emitter__,
                    status_updates,
                    __user__.get("id"),
                    "managed"
                )
            
            # Emit completion status if enabled
            if self.valves.show_status and __event_emitter__:
                op_count = len(operation_results.get("operations", []))
                await self._safe_emit(
                    __event_emitter__,
                    {
                        "type": "status",
                        "data": {
                            "description": f"☑ Completed {op_count} memory management operations",
                            "done": True,
                        },
                    },
                )
                
                # Create a background task to clear the status after delay
                asyncio.create_task(self._delayed_clear_status(__event_emitter__))
            
            return body
            
        except Exception as e:
            logger.error(f"Error in inlet: {e}")
            if self.valves.verbose_logging:
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                
            return body

    async def outlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Outlet processing (called by Open WebUI after response is generated).
        This module only processes inlet requests, so outlet just returns the body unchanged.

        Args:
            body: The message body
            __event_emitter__: Optional event emitter for notifications
            __user__: Optional user information

        Returns:
            The message body unchanged
        """
        # This module only processes inlet requests
        return body

    async def _delayed_clear_status(
        self,
        emitter: Callable[[Any], Awaitable[None]],
        delay_seconds: int = 3
    ) -> None:
        """
        Clear a status message after a delay.
        
        Args:
            emitter: The event emitter function
            delay_seconds: Delay in seconds before clearing
        """
        if not emitter:
            return
            
        try:
            await asyncio.sleep(delay_seconds)
            await self._safe_emit(
                emitter,
                {
                    "type": "status",
                    "data": {
                        "description": "",
                        "done": True,
                    },
                },
            )
        except Exception as e:
            logger.error(f"Error in delayed clear status: {e}")