"""
title: AMM_Memory_Identification_Storage
description: Memory Identification & Storage Module for Open WebUI - Identifies and stores memories from user messages
author: Claude
version: 0.8.8
date: 2025-03-25
changes:
- fixed status updates for UPDATE operations by changing if/elif to separate if statements
- updated status message from "background" to "async" for clarity
- prior changes:
  - made status clearing non-blocking to prevent chat latency
  - added status clearing after operations complete
  - added citation functionality to document memory operations
  - added background non-blocking processing to prevent memory operations from slowing down chat experience
  - added new valve parameter 'background_processing' to toggle background processing on/off
  - added example to prompt showing how to handle related information across multiple messages
  - added support for analyzing multiple recent user messages instead of just the last one
  - added new valve parameter 'recent_messages_count' to control how many messages to analyze
  - fixed memory query detection to properly handle "What do you know about me?" queries
  - added explicit guidance to only update the most relevant memory, not all memories
  - expanded MEMORY QUERIES section with additional example phrases
  - added new FOR ALL UPDATE OPERATIONS section with clear guidance
  - enhanced prompt to prioritize memory consolidation
  - added dedicated MEMORY CONSOLIDATION section to prompt
  - updated CORE TASK section to emphasize consolidation
  - added example of good consolidation
  - modified importance threshold to only apply to NEW operations, allowing all UPDATE/DELETE operations to proceed
  - fixed critical indentation error in _process_memory_operations method
  - moved _process_memory_operations method back inside Filter class
  - removed unnecessary fallback logic for UPDATE/DELETE operations
  - added explicit error handling for missing IDs
  - removed _find_memory_by_content method as it's no longer needed
  - added show_status feature to display memory operation status in chat
  - added _safe_emit method for handling status updates
  - updated inlet and outlet methods to show status during memory operations
  - complete redesign with LLM-first philosophy
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
from pydantic import BaseModel, Field, model_validator


class MemoryOperation(BaseModel):
    """Model for memory operations"""

    operation: Literal["NEW", "UPDATE", "DELETE"]
    id: Optional[str] = None
    content: str
    importance: float = 0.0

    @model_validator(mode="after")
    def validate_fields(self) -> "MemoryOperation":
        if self.operation in ["UPDATE", "DELETE"] and not self.id:
            raise ValueError("id is required for UPDATE and DELETE operations")
        if not self.content:
            raise ValueError("content is required for all operations")
        return self


class Filter:
    """Memory Identification & Storage module for identifying and storing memories from user messages."""

    class Valves(BaseModel):
        # Enable/Disable Function
        enabled: bool = Field(
            default=True,
            description="Enable memory identification & storage",
        )   
        # UI settings
        show_status: bool = Field(
            default=True,
            description="Show memory operation status in chat",
        )  
        # Background processing settings
        background_processing: bool = Field(
            default=True,
            description="Process memories in the background",
        )
        # Processing priority
        priority: int = Field(
            default=20,
            description="Processing priority (higher numbers have higher priority)",
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
            default=0.2,
            description="Temperature for API calls",
        )
        max_tokens: int = Field(
            default=500,
            description="Maximum tokens for API calls",
        )
        # Minimal memory settings
        min_importance_threshold: float = Field(
            default=0.4,
            description="Minimum importance threshold for NEW memories (0.0-1.0)",
        )
        # Message history settings
        recent_messages_count: int = Field(
            default=3,
            description="Number of recent user messages to analyze for memory (1-10)",
        )

    # Enhanced prompt for memory identification with improved CRUD directives and consolidation
    MEMORY_IDENTIFICATION_PROMPT = """
You are a memory identification system for an AI assistant. Your task is to analyze the user's message and identify information worth remembering for future interactions.

## CORE TASK
1. Analyze the user's message: "{current_message}"
2. Review existing memories if provided
3. Identify related pieces of information and consolidate them into comprehensive memories
4. For each consolidated memory:
   - Determine if it's important enough to remember
   - Decide if it should be a new memory or update an existing one
   - Assign an importance score from 0.0 to 1.0
5. Return memory operations with their importance scores

## MEMORY CONSOLIDATION
When identifying memories, prioritize consolidating related information into comprehensive entries rather than creating separate atomic memories. Look for connections between different pieces of information in the user's message and combine them into rich, contextual memories.
For example, if a user discusses their career, health impacts, and future plans related to that career, create ONE comprehensive memory that captures all these aspects rather than separate memories for each point.

## IMPORTANCE CRITERIA (0.0-1.0 scale)
- Explicit memory commands ("remember that...", "don't forget...", etc.) (0.9-1.0)
- Reminder requests ("remind me to/that...") (0.8-1.0)
- Instructions for future actions ("next time we talk about X, do Y") (0.8-0.9)
- Personal details that provide context about the user (0.7-1.0)
- Strong preferences or dislikes (0.6-0.9)
- Life goals and aspirations (0.7-0.9)
- Specific instructions for how the assistant should behave (0.8-1.0)
- Explicit requests to remember information (0.9-1.0)
- Temporary or minor preferences (0.3-0.5)
- Routine conversational details (0.0-0.2)

## EXPLICIT MEMORY COMMANDS
When the user explicitly asks to remember information:
- Assign the highest importance scores (0.9-1.0)
- Extract the exact information they want remembered
- Format it clearly and concisely
- Examples of command phrases: "remember that", "please note", "don't forget", "keep in mind"

## REMINDER REQUESTS
When the user explicitly asks to be reminded of something:
- Format the memory as "Remind user to/that [specific reminder content]"
- Preserve the exact action or information they want to be reminded about
- Assign high importance scores (0.8-1.0) to ensure these are retained
- Examples of reminder phrases: "remind me to", "remind me that", "don't let me forget to"

## MEMORY QUERIES
When the user asks about existing memories or reminders:
- Do NOT create new memories for information that already exists
- Examples of query phrases: "what reminders do I have", "show my reminders", "list my memories", "tell me my reminders", "did I ask you to remind me", "what do you know about me", "what information do you have about me", "tell me what you remember about me", "what do you remember about me", "what have I told you about myself"
- These are information retrieval requests, not requests to create new memories
- Return an empty array [] for these queries as they don't require memory operations

## INTENT PRESERVATION
Always preserve the intent behind the user's statement rather than just the content. This helps maintain the context and purpose of the information for future interactions.
Different user statements carry different intents that should be preserved in the memory:
- Reminders → "Reminded user to/that..."
- Preferences → "User prefers/likes/dislikes..."
- Future plans → "User plans to/intends to..."
- Requests → "User requested that..."
- Facts about self → "User is/has/does..."

## MEMORY OPERATIONS
- NEW: Create a new memory when the information doesn't relate to existing memories
- UPDATE: Modify an existing memory when the information adds to or corrects it
- DELETE: Remove a memory when it's explicitly contradicted or no longer valid

## OPERATION GUIDELINES
### For ALL UPDATE Operations
- Only update the SINGLE most relevant memory that directly relates to the new information
- Do NOT update multiple memories with the same information
- If multiple memories seem related, choose only the most relevant one to update
- If no existing memory is directly relevant, create a NEW memory instead
- When deciding which memory to update, prioritize memories with the highest semantic similarity to the new information

### For NEW Operations
- PRIORITIZE consolidating related information into comprehensive memories
- Create a single memory that captures the full context when information is related
- Only create distinct memories for truly unrelated pieces of information (e.g., different topics)
- Assign higher importance scores to consolidated memories that capture rich context

### For UPDATE Operations (Implicit Corrections)
- When the user corrects a previous statement (e.g., "Actually, I prefer tea over coffee"), update the existing memory
- If new information contradicts existing memory, the most recent statement takes priority unless explicitly stated otherwise
- When possible, merge related details rather than fully replacing prior context
- Preserve existing information unless it has been explicitly superseded

### For UPDATE Operations (Implicit Expansions)
- When the user adds new but related information (e.g., "I also train in Muay Thai"), update the existing memory to include the new details
- If the user mentions a temporary preference (using phrases like "these days," "for now," "currently"), store it separately rather than replacing a lasting preference
- Maintain long-term interests unless explicitly overridden
- When updating, combine the memories: "User likes [Food X]" should become "User likes [Food X and Food Y]"

### For UPDATE Operations (Temporal Changes)
- When the user indicates a change over time (e.g., "I used to like X but now I prefer Y")
- Create an update that clearly preserves the temporal relationship
- Use phrasing like "User previously liked X but now prefers Y" rather than just "User prefers Y"
- Maintain higher importance scores for current preferences while preserving historical context
- Look for temporal keywords: "used to", "previously", "before", "now", "currently", "these days", "lately"

### For UPDATE/DELETE Operations (Implicit Deletions)
- When the user indicates they no longer engage in something (e.g., "I'm not into board games anymore"), modify the existing memory to remove that aspect but keep other relevant details
- Modify only the relevant portion of an existing memory while preserving all other related content
- Use UPDATE operation to remove specific details within a memory
- Use DELETE operation only when the entire memory should be removed

### For UPDATE/DELETE Operations (Explicit Deletions)
- When the user explicitly requests to delete specific information (e.g., "Please delete my preference for Italian food"), perform an UPDATE operation to modify the existing memory
- If the user intends to delete an entire memory and it's clear in their request, perform a DELETE operation

## RESPONSE FORMAT
Your response must be a JSON array of objects with 'operation', 'content', 'importance', and optionally 'id' properties:
[
  {{"operation": "NEW", "content": "User lives in Seattle", "importance": 0.9}},
  {{"operation": "UPDATE", "id": "mem123", "content": "User prefers green tea over coffee", "importance": 0.7}}
]

If no important information is found, return an empty array: []

## EXAMPLES
### Example 1: New Information
User message: "I recently moved to Portland and I'm enjoying the food scene here."
Response:
[
  {{"operation": "NEW", "content": "User lives in Portland and enjoys the food scene there", "importance": 0.8}}
]

### Example 2: Updating Information
User message: "Actually, I've been living in Portland for 5 years now, not just recently."
Existing memory: [ID: mem456] User lives in Portland
Response:
[
  {{"operation": "UPDATE", "id": "mem456", "content": "User has lived in Portland for 5 years", "importance": 0.8}}
]

### Example 3: Implicit Deletion
User message: "I don't really enjoy Portland's food scene anymore since I developed dietary restrictions."
Existing memories:
[ID: mem456] User has lived in Portland for 5 years
[ID: mem789] User enjoys Portland's food scene
Response:
[
  {{"operation": "UPDATE", "id": "mem789", "content": "User previously enjoyed Portland's food scene but developed dietary restrictions", "importance": 0.7}},
  {{"operation": "NEW", "content": "User has dietary restrictions", "importance": 0.8}}
]

### Example 4: Explicit Memory Command
User message: "Remember that I'm allergic to peanuts and shellfish."
Response:
[
  {{"operation": "NEW", "content": "User is allergic to peanuts and shellfish", "importance": 1.0}}
]

### Example 5: Temporal Changes
User message: "I used to live in Chicago, but I moved to Seattle last year."
Existing memory: [ID: mem123] User lives in Chicago
Response:
[
  {{"operation": "UPDATE", "id": "mem123", "content": "User previously lived in Chicago but moved to Seattle last year", "importance": 0.9}}
]

### Example 6: Reminder Request
User message: "Please remind me to call my mother on Sunday."
Response:
[
  {{"operation": "NEW", "content": "Remind user to call their mother on Sunday", "importance": 0.9}}
]

### Example 7: Memory Query
User message: "What reminders do I have?"
Existing memories:
[ID: mem123] Remind user to call their parents
[ID: mem456] Remind user to buy cat food
Response:
[]

### Example 8: Preference Negation
User message: "I don't like coffee anymore."
Existing memory: [ID: mem123] User likes coffee
Response:
[
  {{"operation": "DELETE", "id": "mem123", "content": "User likes coffee", "importance": 0.7}}
]

User message: "I don't like tea."
No existing memory about tea.
Response:
[
  {{"operation": "NEW", "content": "User doesn't like tea", "importance": 0.7}}
]

### Example 9: Consolidating Related Information
User message: "I've been a stuntman for 10 years. It's hard on my body but I love the work. I'm starting to think about what I'll do in the future though."
Response:
[
  {{"operation": "NEW", "content": "User has been a stuntman for 10 years, enjoys the work despite physical toll, and is considering future career options", "importance": 0.85}}
]
### Example 10: Memory Query
User message: "What do you know about me?"
Existing memories:
[ID: mem123] User lives in LA with their girlfriend Bella
[ID: mem456] User has an account and their partner Bella is a dancer
[ID: mem789] User is training to become a chef
[ID: mem101] User has two pets: a dog named Dingo and a cat named Tom
Response:
[]

### Example 11: Related Information Across Messages
User message: "I like cold plunges [Next message] I do them after I train"
Existing memories:
[ID: mem123] User likes taking cold plunges
Response:
[
  {{"operation": "UPDATE", "id": "mem123", "content": "User likes taking cold plunges after training", "importance": 0.8}}
]

Available memories:
Available memories:
{existing_memories}
"""

    def __init__(self) -> None:
        """Initialize the Memory Identification & Storage module."""
        logging.info("Initializing Memory Identification & Storage module v8.4.0")
        self.valves = self.Valves()
        self.session = aiohttp.ClientSession()
        self.memory_operations = []
        self.background_tasks = set()  # Track background tasks
        logging.info(
            "MIS module initialized with API provider: %s", self.valves.api_provider
        )

    async def close(self) -> None:
        """Close the aiohttp session and cancel any running background tasks."""
        logging.info("Closing MIS module session")
        
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
        logging.info("Updating valves with: %s", new_valves)
        for key, value in new_valves.items():
            if hasattr(self.valves, key):
                setattr(self.valves, key, value)

    def _format_memories(self, db_memories: List[Any]) -> str:
        """
        Format existing memories for inclusion in the prompt.
        Simple, straightforward formatting without complex categorization.

        Args:
            db_memories: List of memory objects from the database

        Returns:
            Formatted string of memories
        """
        if not db_memories:
            return "No existing memories."

        formatted = []

        for mem in db_memories:
            if hasattr(mem, "id") and hasattr(mem, "content"):
                formatted.append(f"[ID: {mem.id}] {mem.content}")

        if not formatted:
            return "No valid existing memories."

        return "\n".join(formatted)

    async def _identify_memories(
        self, current_message: str, user_id: str, db_memories: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify memories from the current message using LLM-based scoring.

        Args:
            current_message: The current user message
            user_id: The user ID
            db_memories: List of memories from the database

        Returns:
            List of memory operations with importance scores
        """
        logging.info("Identifying memories from message for user %s", user_id)

        # Format existing memories
        existing_memories = self._format_memories(db_memories)

        # Format the system prompt
        system_prompt = self.MEMORY_IDENTIFICATION_PROMPT.format(
            current_message=current_message, existing_memories=existing_memories
        )

        # Query the appropriate API
        if self.valves.api_provider == "OpenAI API":
            response = await self.query_openai_api(system_prompt, current_message)
        else:  # Ollama API
            response = await self.query_ollama_api(system_prompt, current_message)

        # Parse the response
        memory_operations = self._parse_json_response(response)

        # Log all memory operations with their scores and threshold status
        for op in memory_operations:
            importance = op.get("importance", 0)
            operation_type = op.get("operation", "UNKNOWN")
            
            # Only NEW operations are subject to the threshold
            if operation_type == "NEW":
                meets_threshold = importance >= self.valves.min_importance_threshold
                status = "meets threshold" if meets_threshold else "below threshold"
            else:
                # UPDATE and DELETE operations bypass the threshold
                status = "threshold not applied"
            
            logging.info(
                "Memory operation - type: %s, content: %s, score: %.2f (threshold: %.2f) - %s",
                operation_type,
                op.get("content", ""),
                importance,
                self.valves.min_importance_threshold,
                status
            )

        # Filter NEW operations by importance threshold, but keep all UPDATE/DELETE operations
        filtered_operations = [
            op for op in memory_operations if (
                op.get("operation") != "NEW" or
                op.get("importance", 0) >= self.valves.min_importance_threshold
            )
        ]

        # Sort by importance score
        filtered_operations.sort(key=lambda x: x.get("importance", 0), reverse=True)
            
        return filtered_operations

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
                    logging.error(
                        f"API error (attempt {attempt+1}/{self.valves.max_retries+1}): {str(e)}"
                    )
                    await asyncio.sleep(self.valves.retry_delay)
                else:
                    logging.error(f"Max retries reached: {str(e)}")
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
            List of memory operations
        """
        if not response_text or not response_text.strip():
            return []

        # Remove markdown code block markers
        cleaned = re.sub(r"```json|```", "", response_text.strip())

        try:
            # Parse the JSON
            data = json.loads(cleaned)

            # Validate the data
            if not isinstance(data, list):
                logging.error("Response is not a JSON array")
                return []

            # Validate each operation
            valid_operations = []
            for op in data:
                if not isinstance(op, dict):
                    continue

                if "operation" not in op or op["operation"] not in [
                    "NEW",
                    "UPDATE",
                    "DELETE",
                ]:
                    continue

                if "content" not in op:
                    continue

                if op["operation"] in ["UPDATE", "DELETE"] and "id" not in op:
                    # For UPDATE and DELETE, we need an ID
                    continue

                # Ensure importance is a float
                if "importance" in op:
                    try:
                        op["importance"] = float(op["importance"])
                    except (ValueError, TypeError):
                        op["importance"] = 0.5  # Default importance
                else:
                    op["importance"] = 0.5  # Default importance

                valid_operations.append(op)

            return valid_operations

        except json.JSONDecodeError as e:
            logging.error("Error parsing JSON response: %s", e)
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
        results = []

        for op in operations:
            try:
                operation = op["operation"]
                content = op["content"]
                importance = op.get("importance", 0.5)

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
                        logging.error("UPDATE operation missing required ID field")
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
                        logging.error("DELETE operation missing required ID field")
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

            except Exception as e:
                logging.error("Error processing memory operation: %s", e)
                results.append(
                    {
                        "operation": op.get("operation", "UNKNOWN"),
                        "content": op.get("content", ""),
                        "importance": op.get("importance", 0),
                        "success": False,
                        "status": f"Error: {str(e)}",
                    }
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
                logging.info("Created memory: %s", content)
                return str(memory_id)
            else:
                logging.error("Failed to create memory: No ID returned")
                return None
        except Exception as e:
            logging.error("Error creating memory: %s", e)
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
            logging.info("Updated memory %s: %s", memory_id, content)
            return True
        except Exception as e:
            logging.error("Error updating memory %s: %s", memory_id, e)
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
            logging.info("Deleted memory %s", memory_id)
            return True
        except Exception as e:
            logging.error("Error deleting memory %s: %s", memory_id, e)
            return False
            
    async def _process_memories_background(
        self,
        combined_message: str,
        user: Any,
        user_id: str,
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
            # Get memories
            db_memories = Memories.get_memories_by_user_id(user_id)
            
            # Identify memories from the combined messages
            memory_operations = await self._identify_memories(
                combined_message, user_id, db_memories or []
            )
            
            # If no memory operations, emit completion status and return
            if not memory_operations:
                # Emit completion status if enabled
                if self.valves.show_status and event_emitter:
                    await self._safe_emit(
                        event_emitter,
                        {
                            "type": "status",
                            "data": {
                                "description": "🧠 0 Memories changed",
                                "done": True,
                            },
                        }
                    )
                    
                    # Create a background task to clear the status after delay
                    task = asyncio.create_task(
                        self._delayed_clear_status(event_emitter)
                    )
                    # Add task to set and set up callback to remove it when done
                    self.background_tasks.add(task)
                    task.add_done_callback(self.background_tasks.discard)
                return
                
            # Emit status update if enabled
            if self.valves.show_status and event_emitter:
                await self._safe_emit(
                    event_emitter,
                    {
                        "type": "status",
                        "data": {
                            "description": "💭 Processing memories (async)",
                            "done": False,
                        },
                    }
                )
            
            # Process the memory operations
            self.memory_operations = await self._process_memory_operations(
                memory_operations, user
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
                    content=citation_content
                )
            
            # Emit completion status based on operation counts
            if self.valves.show_status and event_emitter:
                if new_count > 0:
                    await self._safe_emit(
                        event_emitter,
                        {
                            "type": "status",
                            "data": {
                                "description": f"🧠 {new_count} Memories created",
                                "done": True,
                            },
                        }
                    )
                    
                    # Create a background task to clear the status after delay
                    task = asyncio.create_task(
                        self._delayed_clear_status(event_emitter)
                    )
                    # Add task to set and set up callback to remove it when done
                    self.background_tasks.add(task)
                    task.add_done_callback(self.background_tasks.discard)
                
                if update_count > 0:  # Changed from elif to if
                    await self._safe_emit(
                        event_emitter,
                        {
                            "type": "status",
                            "data": {
                                "description": f"🧠 {update_count} Memories updated",
                                "done": True,
                            },
                        }
                    )
                    
                    # Create a background task to clear the status after delay
                    task = asyncio.create_task(
                        self._delayed_clear_status(event_emitter)
                    )
                    # Add task to set and set up callback to remove it when done
                    self.background_tasks.add(task)
                    task.add_done_callback(self.background_tasks.discard)
                
                if delete_count > 0:  # Changed from elif to if
                    await self._safe_emit(
                        event_emitter,
                        {
                            "type": "status",
                            "data": {
                                "description": f"🧠 {delete_count} Memories deleted",
                                "done": True,
                            },
                        }
                    )
                    
                    # Create a background task to clear the status after delay
                    task = asyncio.create_task(
                        self._delayed_clear_status(event_emitter)
                    )
                    # Add task to set and set up callback to remove it when done
                    self.background_tasks.add(task)
                    task.add_done_callback(self.background_tasks.discard)
            
            # Clear the memory operations
            self.memory_operations = []
            
        except Exception as e:
            logging.error("Error in background memory processing: %s", e)
            # Emit error status
            if self.valves.show_status and event_emitter:
                await self._safe_emit(
                    event_emitter,
                    {
                        "type": "status",
                        "data": {
                            "description": "❌ Error processing memories",
                            "done": True,
                        },
                    }
                )

    async def _safe_emit(
        self,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
        data: Dict[str, Any]
    ) -> None:
        """
        Safely emit an event, handling missing emitter.
        
        Args:
            event_emitter: The event emitter function
            data: The data to emit
        """
        if not event_emitter:
            logging.debug("Event emitter not available")
            return
            
        try:
            await event_emitter(data)
        except Exception as e:
            logging.error(f"Error in event emitter: {e}")
            
    async def _send_citation(
        self,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
        url: str,
        title: str,
        content: str
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
            }
        )
    
    async def _delayed_clear_status(
        self,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
        delay_seconds: float = 2.0
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
                }
            )
        except Exception as e:
            logging.error(f"Error in delayed status clearing: {e}")

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
            combined_message = " [Next message] ".join([m.get("content", "") for m in recent_user_messages])
            if not combined_message:
                return body

            # Get user object
            user = Users.get_user_by_id(__user__["id"])
            if not user:
                return body

            # If background processing is enabled, create a background task
            if self.valves.background_processing:
                # Initial status update
                if self.valves.show_status and __event_emitter__:
                    await self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": "💭 Analyzing memories (async)",
                                "done": False,
                            },
                        }
                    )
                
                # Create and register background task
                task = asyncio.create_task(
                    self._process_memories_background(
                        combined_message,
                        user,
                        __user__["id"],
                        __event_emitter__
                    )
                )
                
                # Add task to set and set up callback to remove it when done
                self.background_tasks.add(task)
                task.add_done_callback(self.background_tasks.discard)
                
                # Return immediately without waiting for memory processing
                return body
            else:
                # Original synchronous processing
                db_memories = Memories.get_memories_by_user_id(__user__["id"])
                
                # Emit status update if enabled
                if self.valves.show_status and __event_emitter__:
                    await self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": "💭 Analyzing memories (sync)",
                                "done": False,
                            },
                        }
                    )

                # Identify memories from the combined messages
                memory_operations = await self._identify_memories(
                    combined_message, __user__["id"], db_memories or []
                )

                # If no memory operations, emit completion status and return body unchanged
                if not memory_operations:
                    # Emit completion status if enabled
                    if self.valves.show_status and __event_emitter__:
                        await self._safe_emit(
                            __event_emitter__,
                            {
                                "type": "status",
                                "data": {
                                    "description": "🧠 0 Memories changed",
                                    "done": True,
                                },
                            }
                        )
                        
                        # Create a background task to clear the status after delay
                        asyncio.create_task(
                            self._delayed_clear_status(__event_emitter__)
                        )
                        return body
                    
                # Emit status update if enabled
                if self.valves.show_status and __event_emitter__:
                    await self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": "💭 Processing memories",
                                "done": False,
                            },
                        }
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
                        content=citation_content
                    )

                # No need to modify the message body - we're just storing memories

        except Exception as e:
            logging.error("Error in inlet: %s", e)

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
        if not self.valves.enabled or (not self.memory_operations and self.valves.background_processing):
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
                if new_count > 0:
                    await self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": f"🧠 {new_count} Memories created",
                                "done": True,
                            },
                        }
                    )
                    
                    # Create a background task to clear the status after delay
                    asyncio.create_task(
                        self._delayed_clear_status(__event_emitter__)
                    )
                
                if update_count > 0:  # Changed from elif to if
                    await self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": f"🧠 {update_count} Memories updated",
                                "done": True,
                            },
                        }
                    )
                    
                    # Create a background task to clear the status after delay
                    asyncio.create_task(
                        self._delayed_clear_status(__event_emitter__)
                    )
                
                if delete_count > 0:  # Changed from elif to if
                    await self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": f"🧠 {delete_count} Memories deleted",
                                "done": True,
                            },
                        }
                    )
                    
                    # Create a background task to clear the status after delay
                    asyncio.create_task(
                        self._delayed_clear_status(__event_emitter__)
                    )

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
            logging.error("Error in outlet: %s", e)

        return body
