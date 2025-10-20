# AMM v10 Unified Module Coding Plan

## Overview

This document outlines the plan for creating a new unified Automatic Memory Manager (AMM) module that combines the functionality of the existing Memory Retrieval & Enhancement (MRE v5) and Memory Identification & Storage (MIS v9) modules. The new module will eliminate race conditions and improve information flow while maintaining all existing functionality.

## Goals

1. Create a single module that handles the entire memory lifecycle
2. Eliminate race conditions between memory retrieval and storage
3. Maintain all existing functionality from both modules
4. Preserve existing valve toggles and configuration options
5. Improve information flow and reduce redundancies
6. Ensure clear separation between synchronous and asynchronous operations

## Module Structure

```
AMM_v10_Unified_Module.py
├── class AutomaticMemoryManager
│   ├── class Valves (combined from both modules)
│   ├── Prompts (RELEVANCE, IDENTIFICATION, INTEGRATION)
│   ├── __init__()
│   ├── close()
│   ├── update_valves()
│   ├── inlet() - Main entry point
│   ├── outlet() - Pass-through
│   ├── _retrieve_relevant_memories() - Synchronous
│   ├── _process_memories_background() - Asynchronous
│   ├── _identify_potential_memories() - Stage 1
│   ├── _integrate_potential_memories() - Stage 2
│   └── Helper methods (API calls, formatting, etc.)
```

## Valves Configuration

We'll combine valves from both modules and add a new toggle for memory retrieval:

```python
class Valves(BaseModel):
    # Global settings
    enabled: bool = Field(
        default=True,
        description="Enable automatic memory management",
    )
    show_status: bool = Field(
        default=True,
        description="Show memory processing status in chat"
    )
    priority: int = Field(
        default=10,
        description="Processing priority",
    )
    
    # API configuration (from both modules)
    api_provider: Literal["OpenAI API", "Ollama API"] = Field(...)
    # ... other API settings ...
    
    # Memory Retrieval settings
    memory_retrieval: bool = Field(
        default=True,
        description="Enable memory retrieval & enhancement",
    )
    
    # Memory Identification settings
    memory_identification: bool = Field(
        default=True,
        description="Enable memory identification",
    )
    memory_importance_threshold: float = Field(...)
    
    # Memory Integration settings
    memory_integration: bool = Field(
        default=True,
        description="Enable memory integration",
    )
    memory_relevance_threshold: float = Field(...)
    
    # Processing settings
    background_processing: bool = Field(
        default=True,
        description="Process memories in the background",
    )
    recent_messages_count: int = Field(...)
```

## Processing Flow

### Inlet Method (Synchronous Part)

```python
async def inlet(self, body, event_emitter, user):
    # Basic validation
    if not self.valves.enabled or not body or not user:
        return body
    
    # Get the last user message
    user_message = extract_last_user_message(body)
    if not user_message:
        return body
    
    # SYNCHRONOUS: Retrieve and enhance with relevant memories
    if self.valves.memory_retrieval:
        # Get existing memories
        db_memories = Memories.get_memories_by_user_id(user["id"])
        
        # Show status
        if self.valves.show_status:
            await self._show_status(event_emitter, "💭 Retrieving relevant memories...")
        
        # Get relevant memories
        relevant_memories = await self._retrieve_relevant_memories(
            user_message, user["id"], db_memories
        )
        
        # Update message context with relevant memories
        if relevant_memories:
            self._update_message_context(body, relevant_memories)
            
            # Show status and send citation
            if self.valves.show_status:
                await self._show_status(event_emitter, f"☑ Retrieved {len(relevant_memories)} relevant memories", done=True)
                await self._send_citation(event_emitter, "Memories Read", formatted_memories)
    
    # ASYNCHRONOUS: Start background task for memory identification and integration
    if (self.valves.memory_identification or self.valves.memory_integration) and self.valves.background_processing:
        # Show initial status
        if self.valves.show_status:
            await self._show_status(event_emitter, "💭 Analyzing memories (async)")
        
        # Create and register background task
        task = asyncio.create_task(
            self._process_memories_background(
                user_message, user, user["id"], event_emitter
            )
        )
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
    
    # Return enhanced message immediately
    return body
```

### Background Processing (Asynchronous Part)

```python
async def _process_memories_background(self, message, user, user_id, event_emitter):
    try:
        # Get existing memories
        db_memories = Memories.get_memories_by_user_id(user_id)
        
        # Stage 1: Identify potential memories
        potential_memories = []
        if self.valves.memory_identification:
            potential_memories = await self._identify_potential_memories(message)
            
        # Stage 2: Integrate with existing memories
        memory_operations = []
        if potential_memories and self.valves.memory_integration:
            memory_operations = await self._integrate_potential_memories(
                message, potential_memories, db_memories
            )
        elif potential_memories:
            # If integration is disabled, convert all to NEW operations
            memory_operations = [
                {
                    "operation": "NEW",
                    "content": mem.get("content", ""),
                    "importance": mem.get("importance", 0.0)
                }
                for mem in potential_memories
                if mem.get("importance", 0.0) >= self.valves.memory_importance_threshold
            ]
        
        # Process operations (create/update/delete)
        results = await self._process_memory_operations(memory_operations, user)
        
        # Show status updates and send citation
        if self.valves.show_status and event_emitter:
            # ... status updates for each operation type ...
            
    except Exception as e:
        logging.error(f"Error in background memory processing: {e}")
        # ... error handling ...
    finally:
        # Ensure status is cleared
        # ... cleanup code ...
```

## Code Integration Plan

### From MRE Module

1. **RELEVANCE_PROMPT**: Use as-is for memory retrieval
2. **_retrieve_relevant_memories**: Adapt from MRE's `_get_relevant_memories_llm`
3. **_format_memories_for_context**: Use for formatting memories for the message context
4. **_update_message_context**: Use for adding memories to the message context

### From MIS Module

1. **MEMORY_IDENTIFICATION_PROMPT**: Use as-is for Stage 1
2. **MEMORY_INTEGRATION_PROMPT**: Use as-is for Stage 2 (with our recent improvements)
3. **_identify_potential_memories**: Use for Stage 1 of background processing
4. **_integrate_potential_memories**: Use for Stage 2 of background processing
5. **_process_memory_operations**: Use for creating/updating/deleting memories

### Shared/Common Code

1. **API Calling**: Combine the API calling code from both modules
2. **JSON Parsing**: Use the more robust parsing from MIS
3. **Status Updates**: Use the status update code from MIS (more comprehensive)
4. **Background Tasks**: Use the background task management from MIS

## Optimizations and Improvements

1. **Eliminate Redundant API Calls**: 
   - Both modules currently make separate API calls for their stages
   - Combine when possible to reduce API usage

2. **Improved Memory Formatting**:
   - Use a consistent format for memories across all stages
   - Ensure memory IDs are handled consistently

3. **Better Error Handling**:
   - Add comprehensive error handling for all stages
   - Ensure background tasks don't fail silently

4. **Enhanced Logging**:
   - Consistent logging format across all operations
   - Clear distinction between stages in logs

## Testing Plan

1. **Basic Functionality**:
   - Test memory retrieval works correctly
   - Test memory identification works correctly
   - Test memory integration works correctly

2. **Race Condition Testing**:
   - Test with rapid message sequences
   - Ensure no circular references or race conditions

3. **Edge Cases**:
   - Test with empty messages
   - Test with very large messages
   - Test with various valve configurations

## Implementation Timeline

1. **Phase 1**: Create basic module structure and valves
2. **Phase 2**: Implement synchronous memory retrieval
3. **Phase 3**: Implement asynchronous memory identification and integration
4. **Phase 4**: Add error handling, logging, and status updates
5. **Phase 5**: Testing and refinement

## Conclusion

This unified module will provide a more robust and efficient memory management system by eliminating race conditions and improving information flow. By carefully integrating the best parts of both existing modules, we can maintain all current functionality while addressing the architectural issues we've identified.

## Implementation Notes and Lessons Learned

### What Went Wrong in the First Implementation Attempt

1. **Prompt Simplification**: The implementation drastically simplified the prompts instead of using the full, carefully crafted prompts from the MRE and MIS modules. The plan explicitly stated to "Use as-is" for the prompts, but simplified versions were created that lost critical functionality.

2. **Incorrect Module Structure**: The implementation created an "AutomaticMemoryManager" class but then added a "Function" class that inherits from it, which wasn't in the coding plan. Open WebUI expects a specific module structure, and this implementation broke this pattern.

3. **Valve Configuration**: While valves from both modules were combined, the implementation didn't properly preserve all the functionality and settings as required in the plan.

4. **Implementation Flow**: The implementation didn't follow the exact implementation flow specified in the coding plan, particularly for the inlet method and background processing, which are critical to eliminating race conditions.

5. **Missing Integration of Existing Code**: The plan clearly specified which parts to take from each module (MRE and MIS), but much of the code was rewritten instead of integrating the existing, tested functionality.

### Mitigation Strategies for Next Attempt

1. **Strict Adherence to Prompts**: Use the exact prompts from the existing modules without any simplification or modification. These prompts contain critical logic that has been refined over multiple versions.

2. **Follow Module Structure Exactly**: Implement the module structure exactly as specified in the coding plan, without adding any additional classes or inheritance patterns not specified.

3. **Preserve All Valve Parameters**: Ensure all valve parameters from both modules are preserved with their exact functionality and default values.

4. **Copy Implementation Flow**: Follow the processing flow exactly as specified in the plan, particularly for the inlet method and background processing.

5. **Integrate Existing Code**: Instead of rewriting functionality, copy and adapt code from the existing modules as specified in the plan. This ensures that all existing functionality is maintained.

6. **Incremental Implementation and Testing**: Implement the module in phases as outlined in the timeline, testing each phase thoroughly before moving to the next.

7. **Reference Existing Modules**: Continuously reference the existing MRE v5 and MIS v9 modules to ensure that all functionality is properly integrated.

8. **Understand Open WebUI Module System**: Ensure a thorough understanding of how Open WebUI modules are structured and how they interact with the system before making any changes.

By following these strategies, the next implementation attempt should successfully create a unified module that maintains all existing functionality while addressing the architectural issues identified in the plan.