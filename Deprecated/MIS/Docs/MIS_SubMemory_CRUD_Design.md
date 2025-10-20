# MIS Sub-Memory (Bullet) CRUD Design: Reliable LLM-Code Interop

## Purpose

This document defines the requirements and design for robust, reliable CRUD operations on sub-memories (bullets) within compound memories, ensuring the code can faithfully implement the LLM's intent. It also specifies the memory format and LLM output contract needed for seamless, deterministic execution.

---

## 1. Memory Format: Compound Memories with Sub-Memories

- **Compound Memory Block:**  
  - Each memory is stored as a block with a category header and a list of sub-memories (bullets).
  - Each block has a unique memory ID.
  - Example:
    ```
    User Profile: (ID: mem123)
    - Lives in Denver
    - Age is 30
    - Lives with Anna
    ```
- **Sub-Memory (Bullet):**  
  - Each bullet is an atomic, addressable sub-memory.
  - Bullets can be added or deleted independently.

---

## 2. LLM Output Contract for Sub-Memory Operations

### A. Referencing Sub-Memories: Hybrid Approach

- **For all sub-memory CRUD operations, the LLM must output:**
  - The parent memory ID (category block)
  - The bullet text (as it appears in the memory block)
  - The line number (index) of the bullet within the block (0-based or 1-based, must be specified)
- **Why hybrid?**
  - Index is robust to duplicate bullets and minor text changes.
  - Text provides human readability and a fallback if the index is out of sync.
  - Code can cross-validate both for maximum reliability.

### B. Adding a Sub-Memory

- LLM outputs:
  - Memory ID (category block)
  - New bullet text
  - (No index needed for addition)
- Example:
  ```
  {
    "memory_id": "mem123",
    "sub_memory": "Lives with Anna",
    "intent": "add"
  }
  ```

### C. Deleting a Sub-Memory

- LLM outputs:
  - Memory ID (category block)
  - Bullet text to remove
  - Line number (index) of bullet to remove
- Example:
  ```
  {
    "memory_id": "mem123",
    "sub_memory": "Lives with Anna",
    "index": 3,
    "intent": "delete"
  }
  ```

### D. Updating a Sub-Memory

- LLM outputs:
  - Memory ID (category block)
  - Old bullet text
  - New bullet text
  - Line number (index) of bullet to update
- Example:
  ```
  {
    "memory_id": "mem123",
    "old_sub_memory": "Lives in Denver",
    "new_sub_memory": "Lives in Austin",
    "index": 1,
    "intent": "update"
  }
  ```

### E. LLM Output Format

- All sub-memory operations must be explicit and reference both the memory ID and the bullet (by text and index).
- The LLM must be aware of the current memory format and present its intent in a way that is unambiguous for the code to execute.

---

## 3. Code Responsibilities

- **Parsing:**  
  - Parse LLM output for memory ID, bullet text, and index.
- **Matching:**  
  - Match memory ID to existing compound memory block.
  - Match sub-memory (bullet) by index if possible, cross-validate with text.
  - If index and text disagree, log a warning and prefer index if memory state is unchanged, otherwise prefer text.
- **CRUD Execution:**  
  - Add: Insert new bullet into the correct compound memory (or create new block if category does not exist).
  - Delete: Remove bullet at the specified index (if text matches), or search for exact text match if index is ambiguous.
  - Update: Replace bullet at the specified index with new text (if old text matches), or search for exact text match if index is ambiguous.
- **Validation:**  
  - Log and handle ambiguous or unmatched sub-memory operations (e.g., bullet not found).
- **Idempotency:**  
  - Ensure repeated add/delete/update operations do not corrupt memory state.

---

## 4. LLM Prompt Engineering Guidelines

- Always output memory ID, bullet text, and index for each sub-memory operation.
- For deletions/updates, use the exact bullet text and correct index as it appears in the memory block.
- For additions, ensure the bullet is not a duplicate of an existing bullet in the same category.
- Avoid ambiguous or partial references (e.g., "remove the Denver info" is not sufficient).

---

## 5. Example: Full LLM Output for a User Message

User message: "I moved to Austin and no longer live with Anna."

LLM output (semantic, not CRUD):
```
[
  {
    "memory_id": "mem123",
    "old_sub_memory": "Lives in Denver",
    "new_sub_memory": "Lives in Austin",
    "index": 1,
    "intent": "update"
  },
  {
    "memory_id": "mem123",
    "sub_memory": "Lives with Anna",
    "index": 3,
    "intent": "delete"
  }
]
```

---

## 6. Testing & Validation

- Unit tests for all add/delete/update scenarios, including edge cases (duplicates, ambiguous matches, last bullet removal).
- Integration tests with real LLM output and memory state.

---

## 7. Summary

- The LLM must be fully aware of the memory format and output explicit, unambiguous sub-memory operations using both memory ID and bullet (text + index).
- The code must faithfully and deterministically execute add, delete, and update operations on bullets, ensuring robust, human-like memory management.