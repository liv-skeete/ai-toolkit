# Embedding Support Implementation Plan for AMM

## Overview

This plan outlines the implementation of vector embedding support for both the Memory Identification & Storage (MIS) and Memory Retrieval & Enhancement (MRE) modules in the Automatic Memory Manager. After examining the code, I've found that neither module currently has vector embedding support implemented. The MIS module uses text matching for memory resolution, and the MRE module uses LLM-based relevance scoring for memory retrieval. By implementing embedding support in both modules, we can:

1. Improve memory retrieval performance by reducing API calls
2. Enhance memory relevance accuracy through vector similarity search
3. Maintain consistency between the MIS and MRE modules
4. Provide a fallback mechanism to LLM-based scoring when needed
5. Improve memory resolution accuracy for updates and deletions

## Current State Analysis

### MIS Module (AMM_v5_MIS_Module.py)

The MIS module currently uses text matching for memory resolution with the following components:

1. `text_match_threshold` parameter for text similarity matching (line 141)
2. `_resolve_memory_id` method that uses text matching (lines 1173-1268)
3. `_extract_key_phrases` method for basic text analysis (lines 1271-1317)
4. Comments throughout the code indicating that text matching is an interim solution (lines 417-419, 425-426, 1207-1208, 1270)

The MIS module needs to be updated to use vector embeddings for memory resolution, which will improve accuracy when resolving memory IDs for UPDATE and DELETE operations.

### MRE Module (AMM_v5_MRE_Module.py)

The MRE module currently uses LLM-based relevance scoring with:

1. `relevance_threshold` parameter for minimum relevance score
2. `max_memories` parameter for maximum memories to include
3. `RELEVANCE_PROMPT` for LLM-based memory scoring
4. `get_relevant_memories` method that uses LLM to score memory relevance
5. `_format_memories_for_context` method to format memories for inclusion in context

The MRE module needs to be updated to use vector embeddings for memory retrieval, which will improve both performance and accuracy.

### Open WebUI Vector Database Integration (Reference/retrieval.py)

From examining the reference implementation in `retrieval.py`, we can see:

1. The vector database client is imported from `open_webui.retrieval.vector.connector` as `VECTOR_DB_CLIENT`
2. The embedding function is accessed via `request.app.state.EMBEDDING_FUNCTION`
3. There's an endpoint at `/api/memories/ef/{text}` that returns embeddings for a given text
4. The vector database supports operations like search, upsert, delete, etc.
5. The system supports different embedding models and providers (OpenAI, Ollama, etc.)

## Implementation Plan

### 1. MIS Module Updates

#### 1.1 Update Module Header

Update the version and changelog in the module header:

```python
"""
title: AMM_Memory_Identification_Storage
description: Memory Identification & Storage Module for Open WebUI - Identifies and stores memories from user messages
author: Claude
version: 5.6.9
date: 2025-03-12
changes:
- implemented vector embedding support for memory operations
- renamed text_match_threshold to vector_similarity_threshold
- added embedding generation and vector database integration
- updated memory resolution to use vector similarity
- maintained backward compatibility with text-based matching as fallback
- consolidated prior changes:
  - added special handling for foundational information types to prevent incorrect overwrites
  - increased text matching threshold to reduce false matches between unrelated memories
  - enhanced key phrase extraction to detect different types of foundational information
  - improved memory resolution logic to apply penalty for different information types
  ...
"""
```

#### 1.2 Update Valve Settings

Replace the `text_match_threshold` with `vector_similarity_threshold`:

```python
# Vector similarity threshold for memory matching
vector_similarity_threshold: float = Field(
    default=0.70,  # Lower than text matching as vector similarity works differently
    description="Threshold for vector similarity when resolving memory IDs (0.0-1.0)",
),
```

#### 1.3 Add Embedding Method

Add a method to get embeddings from the Open WebUI API:

```python
async def _get_embedding(self, content: str, user=None) -> Optional[List[float]]:
    """Get embedding using the existing /ef endpoint."""
    try:
        # Use path parameter format for the embedding endpoint
        async with self.session.get(
            f"/api/memories/ef/{content}"
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data["result"]
            logger.warning(f"Error getting embedding: {response.status}")
            return None
    except Exception as e:
        logger.error(f"Exception getting embedding: {e}")
        return None
```

#### 1.4 Add Vector Database Update Methods

Add methods to update and delete vector embeddings:

```python
async def _update_memory_embedding(self, memory_id: str, content: str, user: Any) -> None:
    """Update vector embedding for a memory in the background."""
    try:
        # Import vector DB client
        from open_webui.retrieval.vector.connector import VECTOR_DB_CLIENT
        
        # Get embedding from Open WebUI
        embedding = await self._get_embedding(content, user)
        
        if embedding:
            # Update vector database
            VECTOR_DB_CLIENT.upsert(
                collection_name=f"user-memory-{user.id}",
                items=[{
                    "id": memory_id,
                    "text": content,
                    "vector": embedding,
                    "metadata": {"updated_at": datetime.now().isoformat()},
                }],
            )
            logger.info(f"Updated embedding for memory {memory_id}")
        else:
            logger.warning(f"Could not generate embedding for memory {memory_id}")
    except Exception as e:
        logger.error(f"Error updating memory embedding: {e}")

async def _delete_memory_embedding(self, memory_id: str, user: Any) -> None:
    """Delete vector embedding for a memory in the background."""
    try:
        # Import vector DB client
        from open_webui.retrieval.vector.connector import VECTOR_DB_CLIENT
        
        # Delete from vector database
        VECTOR_DB_CLIENT.delete(
            collection_name=f"user-memory-{user.id}",
            ids=[memory_id]
        )
        logger.info(f"Deleted embedding for memory {memory_id}")
    except Exception as e:
        logger.error(f"Error deleting memory embedding: {e}")
```

#### 1.5 Update Memory Resolution Method

Update the `_resolve_memory_id` method to use vector similarity with fallback to text matching:

```python
async def _resolve_memory_id(
    self,
    operation: MemoryOperation,
    user: Any,
    all_memories: Optional[List[Any]] = None,
) -> Optional[str]:
    """
    Resolve memory ID using vector similarity with fallback to text matching.
    
    This method first tries direct ID matching, then vector similarity,
    and finally falls back to text matching if vector similarity fails.
    
    Args:
        operation: The memory operation to resolve
        user: The user object
        all_memories: Optional list of all memories for the user
        
    Returns:
        The resolved memory ID or None if no match is found
    """
    logger.info("Resolving memory ID for operation: %s", operation.operation)
    # First try direct ID match (existing behavior)
    if operation.id:
        logger.info("Attempting direct ID match with ID: %s", operation.id)
        existing_memory = Memories.get_memory_by_id(operation.id)
        if existing_memory and existing_memory.user_id == str(user.id):
            logger.info("Found direct ID match")
            return existing_memory.id
        else:
            logger.warning("Direct ID match failed")
    
    # If content is provided, try vector similarity matching
    if operation.content and all_memories:
        try:
            # Import vector DB client
            from open_webui.retrieval.vector.connector import VECTOR_DB_CLIENT
            
            # Get embedding for operation content
            embedding = await self._get_embedding(operation.content, user)
            
            if embedding:
                # Search for similar memories
                results = VECTOR_DB_CLIENT.search(
                    collection_name=f"user-memory-{user.id}",
                    vectors=[embedding],
                    limit=1,
                    min_score=self.valves.vector_similarity_threshold
                )
                
                if results and results[0] and len(results[0]) > 0:
                    memory_id = results[0][0]["id"]
                    score = results[0][0]["score"]
                    logger.info(f"Found memory match using vector similarity: {memory_id} (score: {score})")
                    return memory_id
                
                logger.info("No vector similarity match found above threshold")
            except Exception as e:
                logger.error(f"Error in vector similarity matching: {e}")
        
        # Fallback to text matching (existing behavior)
        logger.info("Falling back to text matching")
        best_match_id = None
        best_match_score = 0.85  # Use higher threshold for text matching
        
        # [Existing text matching code remains unchanged]
        
    logger.warning("Could not resolve memory ID")
    return None
```

#### 1.6 Update Memory Operation Execution

Update the `_execute_memory_operation` method to update vector embeddings:

```python
async def _execute_memory_operation(
    self,
    operation: MemoryOperation,
    user: Any,
    all_memories: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a memory operation with minimal logic.
    All matching decisions will be made by the Manager.
    """
    formatted_content = (operation.content or "").strip()
    logger.info("Executing memory operation: %s on content: %s", operation.operation, formatted_content[:50] + "..." if len(formatted_content) > 50 else formatted_content)
    try:
        if operation.operation == "NEW":
            # Always create new memory (Manager is responsible for avoiding duplicates)
            logger.info("Creating new memory for user %s", user.id)
            result = Memories.insert_new_memory(
                user_id=str(user.id), content=formatted_content
            )
            logger.info("New memory created with ID: %s", result.id if hasattr(result, "id") else "Unknown")
            
            # Queue embedding update as a background task
            self.task_queue.put_nowait((
                self._update_memory_embedding,
                (result.id, formatted_content, user),
                {}
            ))
            
            return {
                "operation": "NEW",
                "content": formatted_content,
                "success": True,
                "status": "Memory added successfully.",
            }

        elif operation.operation == "UPDATE":
            # Only update if explicit ID is provided
            resolved_id = await self._resolve_memory_id(operation, user, all_memories)
            if resolved_id:
                logger.info("Updating memory with ID: %s", resolved_id)
                result = Memories.update_memory_by_id(
                    resolved_id, content=formatted_content
                )
                
                # Queue embedding update as a background task
                self.task_queue.put_nowait((
                    self._update_memory_embedding,
                    (resolved_id, formatted_content, user),
                    {}
                ))
                
                return {
                    "operation": "UPDATE",
                    "content": formatted_content,
                    "success": True,
                    "status": f"Memory updated successfully (id: {resolved_id}).",
                }
            else:
                # Create new memory if no ID match (Manager is responsible for this decision)
                logger.info("No matching memory found for update, creating new memory")
                result = Memories.insert_new_memory(
                    user_id=str(user.id), content=formatted_content
                )
                logger.info("New memory created with ID: %s", result.id if hasattr(result, "id") else "Unknown")
                
                # Queue embedding update as a background task
                self.task_queue.put_nowait((
                    self._update_memory_embedding,
                    (result.id, formatted_content, user),
                    {}
                ))
                
                return {
                    "operation": "NEW",
                    "content": formatted_content,
                    "success": True,
                    "status": "No matching memory found; a new memory has been created.",
                }

        elif operation.operation == "DELETE":
            # Only delete if explicit ID is provided
            resolved_id = await self._resolve_memory_id(operation, user, all_memories)
            if resolved_id:
                logger.info("Deleting memory with ID: %s", resolved_id)
                deleted = Memories.delete_memory_by_id(resolved_id)
                
                # Delete from vector database as a background task
                self.task_queue.put_nowait((
                    self._delete_memory_embedding,
                    (resolved_id, user),
                    {}
                ))
                
                return {
                    "operation": "DELETE",
                    "content": formatted_content,
                    "success": True,
                    "status": "Memory deleted successfully.",
                }
            else:
                logger.warning("Could not resolve memory ID for deletion")
                return {
                    "operation": "DELETE",
                    "content": formatted_content,
                    "success": False,
                    "status": "Memory deletion failed (could not resolve memory ID).",
                }
    except Exception as e:
        logger.error("Error executing memory operation: %s", e)
        return {
            "operation": operation.operation,
            "content": formatted_content,
            "success": False,
            "status": f"Operation failed: {str(e)}",
        }
```

### 2. MRE Module Updates

#### 2.1 Update Module Header

Update the version and changelog in the module header:

```python
"""
title: AMM_Memory_Retrieval_Enhancement
description: Memory Retrieval & Enhancement Module for Open WebUI - Retrieves relevant memories and enhances prompts
author: Claude
version: 5.1.0
date: 2025-03-12
changes:
- implemented vector embedding support for memory retrieval
- replaced LLM-based relevance scoring with vector similarity search
- added fallback to LLM-based scoring when vector search fails
- improved performance by reducing API calls for memory retrieval
- maintained backward compatibility with existing memory format
- removed relevance markers from memory formatting to prevent duplication when recalled
- extracted Memory Retrieval & Enhancement functionality from AMM v4
- implemented LLM-first approach for memory relevance scoring
- simplified memory retrieval logic
- improved memory formatting for context
- implemented consistent error handling
- removed special case handling in favor of LLM-based decisions
- added configurable relevance threshold and maximum memories
"""
```

#### 2.2 Add Vector Search Settings to Valves

Add new valve parameters for vector search configuration:

```python
# Vector search settings
use_vector_search: bool = Field(
    default=True,
    description="Use vector search for memory retrieval (faster but may be less accurate)",
)
fallback_to_llm: bool = Field(
    default=True,
    description="Fall back to LLM-based scoring if vector search fails",
)
```

#### 2.3 Add Embedding Method

Add a method to get embeddings from the Open WebUI API:

```python
async def _get_embedding(self, content: str, user_id: str) -> Optional[List[float]]:
    """Get embedding using the existing /ef endpoint."""
    try:
        # Use path parameter format for the embedding endpoint
        async with self.session.get(
            f"/api/memories/ef/{content}"
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data["result"]
            logger.warning(f"Error getting embedding: {response.status}")
            return None
    except Exception as e:
        logger.error(f"Exception getting embedding: {e}")
        return None
```

#### 2.4 Implement Vector-Based Memory Retrieval

Add a new method for vector-based memory retrieval:

```python
async def _get_relevant_memories_vector(
    self, current_message: str, user_id: str, db_memories: List[Any]
) -> List[Dict[str, Any]]:
    """
    Get memories relevant to the current context using vector similarity search.
    
    Args:
        current_message: The current user message
        user_id: The user ID
        db_memories: List of memories from the database
        
    Returns:
        List of dictionaries containing relevant memories with their scores
    """
    try:
        # Import vector DB client
        from open_webui.retrieval.vector.connector import VECTOR_DB_CLIENT
        
        # Get embedding for current message
        embedding = await self._get_embedding(current_message, user_id)
        
        if not embedding:
            logger.warning("Could not generate embedding for current message")
            return []
        
        # Search for similar memories
        results = VECTOR_DB_CLIENT.search(
            collection_name=f"user-memory-{user_id}",
            vectors=[embedding],
            limit=self.valves.max_memories,
            min_score=self.valves.relevance_threshold
        )
        
        if not results or not results[0]:
            logger.info("No vector search results found")
            return []
        
        # Format results
        memory_results = []
        for result in results[0]:
            memory_results.append({
                "text": result["text"],
                "score": result["score"]
            })
        
        # Sort by score in descending order
        memory_results.sort(key=lambda x: x["score"], reverse=True)
        
        return memory_results
        
    except Exception as e:
        logger.error(f"Error in vector similarity search: {e}")
        raise
```

#### 2.5 Rename Existing Method and Update Main Method

Rename the existing `get_relevant_memories` method to `_get_relevant_memories_llm` and update the main method to use vector search with fallback to LLM:

```python
async def get_relevant_memories(
    self, current_message: str, user_id: str, db_memories: Optional[List[Any]] = None
) -> List[Dict[str, Any]]:
    """
    Get memories relevant to the current context using vector similarity search.
    Falls back to LLM-based scoring if vector search fails or is disabled.
    
    Args:
        current_message: The current user message
        user_id: The user ID
        db_memories: Optional list of memories from the database
        
    Returns:
        List of dictionaries containing relevant memories with their scores
    """
    logger.info(f"Getting relevant memories for message: {current_message[:50]}...")
    
    if not self.valves.enabled:
        logger.info("Module disabled, returning empty list")
        return []
        
    try:
        # Get memories from database if not provided
        if db_memories is None:
            logger.info(f"Fetching memories from database for user {user_id}")
            db_memories = Memories.get_memories_by_user_id(user_id=str(user_id))
            
        # If no memories, return empty list
        if not db_memories:
            logger.info("No memories found in database, returning empty list")
            return []
        
        # Try vector search first if enabled
        if self.valves.use_vector_search:
            try:
                logger.info("Using vector similarity search for memory retrieval")
                vector_results = await self._get_relevant_memories_vector(
                    current_message, user_id, db_memories
                )
                
                if vector_results:
                    logger.info(f"Vector search found {len(vector_results)} relevant memories")
                    return vector_results
                else:
                    logger.info("Vector search found no relevant memories")
                    
                    # Fall back to LLM-based scoring if enabled
                    if self.valves.fallback_to_llm:
                        logger.info("Falling back to LLM-based scoring")
                    else:
                        logger.info("Fallback to LLM disabled, returning empty list")
                        return []
            except Exception as e:
                logger.error(f"Error in vector similarity search: {e}")
                
                # Fall back to LLM-based scoring if enabled
                if self.valves.fallback_to_llm:
                    logger.info("Falling back to LLM-based scoring due to error")
                else:
                    logger.error("Fallback to LLM disabled, returning empty list")
                    return []
        
        # Use LLM-based scoring if vector search is disabled or failed
        if not self.valves.use_vector_search or self.valves.fallback_to_llm:
            logger.info("Using LLM-based scoring for memory retrieval")
            return await self._get_relevant_memories_llm(
                current_message, user_id, db_memories
            )
        
        return []
            
    except Exception as e:
        # Log the exception but return empty list
        logger.error(f"Error in get_relevant_memories: {e}")
        return []
```

#### 2.6 Update Relevance Threshold Description

Update the description of the `relevance_threshold` parameter to reflect that it's used for both vector similarity and LLM-based scoring:

```python
relevance_threshold: float = Field(
    default=0.5,
    description="Minimum relevance/similarity score (0.0-1.0) for memories to be included",
),
```

## Testing and Validation

### 1. MIS Module Testing

1. **Basic Functionality Test**:
   - Ensure the MIS module can resolve memory IDs using vector similarity
   - Verify that the fallback to text matching works when vector similarity fails
   - Test with different types of memory operations (NEW, UPDATE, DELETE)

2. **Performance Testing**:
   - Compare response times between vector-based and text-based resolution
   - Measure the impact on overall system performance

3. **Accuracy Testing**:
   - Compare the accuracy of memory resolution using vector similarity vs. text matching
   - Adjust the `vector_similarity_threshold` as needed to optimize accuracy

4. **Edge Cases**:
   - Test with empty memory database
   - Test with very large memory database
   - Test with similar but distinct memories

### 2. MRE Module Testing

1. **Basic Functionality Test**:
   - Ensure the MRE module can retrieve memories using vector search
   - Verify that the fallback to LLM-based scoring works when vector search fails
   - Test with different types of user queries (questions, statements, commands)
   - Verify that the retrieved memories are correctly formatted and included in the context

2. **Performance Testing**:
   - Compare response times between vector-based and LLM-based retrieval
   - Measure the impact on overall system performance
   - Test with different numbers of memories to evaluate scaling behavior
   - Benchmark memory usage and CPU/GPU utilization

3. **Accuracy Testing**:
   - Compare the relevance of memories retrieved by vector search vs. LLM-based scoring
   - Adjust the `vector_similarity_threshold` as needed to optimize accuracy
   - Test with different embedding models to find the best balance of speed and accuracy
   - Evaluate the quality of retrieved memories with different types of user queries

4. **Edge Cases**:
   - Test with empty memory database
   - Test with very large memory database (1000+ memories)
   - Test with unusual or complex user queries
   - Test with multilingual content
   - Test with specialized domain-specific content

5. **Integration Testing**:
   - Verify that the MRE module works correctly with the MIS module
   - Test the complete memory lifecycle (creation, retrieval, update, deletion)
   - Ensure that vector embeddings are properly updated when memories are modified
   - Test with different user accounts to verify isolation between users

## Implementation Notes

1. The vector search implementation relies on the existing vector database infrastructure in Open WebUI, specifically the `VECTOR_DB_CLIENT` from `open_webui.retrieval.vector.connector`.
2. The embedding function is accessed through the `/api/memories/ef/{text}` endpoint.
3. The fallback mechanism ensures robustness even if vector search fails, maintaining the current text-based and LLM-based functionality.
4. The implementation maintains backward compatibility with the existing memory format and doesn't require any database schema changes.
5. The code structure follows the existing patterns in both modules for consistency and maintainability.
6. The vector similarity threshold may need to be adjusted differently from the text matching threshold, as vector similarity scores have a different distribution.
7. Error handling is comprehensive to ensure that memory operations don't impact the user experience.
8. The implementation leverages the existing embedding infrastructure in Open WebUI, which supports different embedding models and providers.

## Future Improvements

1. Implement more sophisticated embedding models for better semantic understanding
2. Add support for hybrid retrieval (combining vector search with keyword search) similar to the `query_doc_with_hybrid_search` function in retrieval.py
3. Implement memory clustering for more efficient retrieval of related memories
4. Add support for memory reranking to improve relevance using a cross-encoder model
5. Implement caching of embeddings to reduce computation time for frequently accessed memories
6. Add support for incremental updates to embeddings when memories are modified
7. Implement a feedback mechanism to improve memory retrieval based on user interactions
8. Add support for different embedding models for different types of memories or users
9. Explore techniques for reducing the dimensionality of embeddings to improve storage efficiency
10. Implement a mechanism for periodically recomputing all embeddings when the embedding model is updated

## Conclusion

The implementation of vector embedding support for both the MIS and MRE modules represents a significant enhancement to the Automatic Memory Manager. By leveraging the existing vector database infrastructure in Open WebUI, we can improve both the performance and accuracy of memory operations while maintaining backward compatibility with the current system.

### Implementation Priorities

1. **MIS Module Updates**: Implement vector similarity for memory resolution first, as it affects the core memory operations
2. **MRE Module Updates**: Implement vector search for memory retrieval after the MIS module is updated
3. **Robust Error Handling**: Ensure comprehensive error handling to maintain system stability
4. **Performance Optimization**: Fine-tune the vector similarity threshold and other parameters
5. **Testing and Validation**: Thoroughly test the implementation with various scenarios

### Expected Benefits

1. **Reduced API Calls**: Vector search eliminates the need for LLM API calls for memory relevance scoring
2. **Improved Performance**: Vector similarity search is significantly faster than LLM-based scoring
3. **Enhanced Accuracy**: Vector embeddings can capture semantic relationships better than text matching
4. **Consistent Architecture**: Aligns both modules with a common approach to memory management
5. **Better Memory Resolution**: Improves the accuracy of memory updates and deletions

This implementation plan provides a clear roadmap for enhancing both modules with vector embedding support, building on the existing infrastructure and maintaining consistency with the overall system architecture.