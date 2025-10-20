# MIS Modular Implementation Plan

## Purpose

This document provides a step-by-step, modular implementation plan for restoring nuanced prompt logic and realigning LLM/code responsibilities in the MIS system. Changes are grouped logically for progressive coding, testing, and validation, minimizing risk and maximizing maintainability.

---

## Phase 1: Prompt Refactor – Nuanced Memory Analysis

**Objective:** Restore and expand nuanced logic in the LLM prompts, ensuring all semantic analysis, scoring, best match identification, and reasoning is handled by the LLM. The LLM must NOT output CRUD operation recommendations—only semantic data, and must be fully aware of the compound memory + sub-memory format and output contract.

### 1.1. Restore Implicit/Explicit Deletion Logic
- Add explicit rules and examples for:
  - Partial vs. full deletion (flagged as semantic intent, not CRUD)
  - Implicit negation/contradiction (e.g., "I don't like X anymore")
  - Anti-patterns for over-deletion
- **Test:** Prompt-only unit tests with sample user statements and expected LLM output (semantic, not CRUD).

### 1.2. Reinstate Expansion/Merge and Temporal Change Logic
- Add rules and examples for:
  - Merging new details into existing memories (semantic merge intent, not CRUD)
  - Handling temporal changes (e.g., "used to", "now", "previously")
  - Anti-patterns for overwriting vs. merging
- **Test:** Prompt-only unit tests for expansion, merge, and temporal scenarios.

### 1.3. Clarify Category Assignment and Merge/Split Recommendations
- Add guidance and examples for:
  - When to merge or split memories by category/topic (semantic recommendation, not CRUD)
  - Handling compound and split memories
- **Test:** Prompt-only unit tests for category merge/split cases.

### 1.4. Add Anti-patterns and Failure Examples
- Integrate anti-patterns and failure examples throughout prompts.
- **Test:** Prompt-only unit tests to ensure anti-patterns are avoided.

### 1.5. Enforce LLM Output Contract (with Sub-Memory CRUD Awareness)
- Ensure the prompt outputs, for each potential memory:
  - Content, category, importance
  - Relevance scores to all existing memories (with IDs)
  - Best match (ID, score)
  - Reasoning for each score
  - Explicit flags for deletion/negation (with target description)
  - For sub-memory (bullet) CRUD ops:
    - Always output the parent memory ID
    - For add: new bullet text
    - For delete/update: bullet text and line number (index) as it appears in the memory block
    - For update: both old and new bullet text, and index
  - **No CRUD operation recommendations**
- **Test:** Contract validation tests for prompt output, including sub-memory CRUD scenarios.

**Status:**  
:heavy_check_mark: **Completed**  
- Prompts/MIS_Identification_Prompt_v6.md and Prompts/MIS_Integration_Prompt_v6.md have been created with all requirements above.

---

## Phase 2: Code Refactor – Deterministic CRUD Operations

**Objective:** Ensure the code executes only deterministic CRUD operations based on LLM semantic output, removing any duplicated or conflicting semantic logic. The code should be logic-agnostic, stable, and only updated for structural reasons. The code must faithfully implement sub-memory (bullet) CRUD using the hybrid (ID + text + index) approach.

### 2.1. Remove Semantic/Category/Merge Logic from Code
- Refactor code to:
  - Remove or minimize any semantic similarity, category, or merge/split logic that should be handled by the LLM.
  - Ensure code only parses LLM semantic output (not CRUD ops).
  - Map LLM output to CRUD operations using explicit, minimal, and stable rules:
    - If explicit deletion/negation: 
      - If target is a specific sub-memory: UPDATE (remove bullet by ID + text + index)
      - If target is an entire memory: DELETE
    - Else:
      - If best match relevance >= memory_relevance_threshold and same category: add as sub-memory (bullet) to compound memory
      - If no memory of that category exists: create new compound memory and add as first bullet
      - If < threshold: create new compound memory and add as first bullet
- **Test:** Unit tests for CRUD operation execution, using mock LLM semantic output, including sub-memory add/delete/update.

### 2.2. Strengthen Output Parsing, Validation, and Error Handling
- Ensure code:
  - Strictly validates LLM output structure and required fields (semantic contract, not CRUD).
  - For sub-memory ops, cross-validate by both index and text; log and handle mismatches.
  - Handles ambiguous or incomplete LLM output with robust error handling and logging (e.g., fallback to CREATE).
- **Test:** Unit tests for error handling and fallback logic, including sub-memory edge cases.

---

## Phase 3: Integration & End-to-End Testing

**Objective:** Validate the full LLM-code pipeline for all nuanced memory operations, ensuring the LLM never outputs CRUD ops and the code applies deterministic mapping, including robust sub-memory CRUD.

### 3.1. Integration Tests for All Operation Types
- Test end-to-end flow for:
  - Implicit/explicit deletion (semantic intent only)
  - Expansion/merge (semantic intent only)
  - Temporal changes
  - Category merge/split
  - Sub-memory (bullet) add/delete/update, including edge cases (duplicates, ambiguous matches, last bullet removal)
  - Anti-pattern avoidance
- **Test:** Integration tests with real user scenarios and expected database state.

### 3.2. Regression Testing
- Ensure no regressions in basic memory CRUD operations or user experience.
- **Test:** Automated regression suite.

---

## Phase 4: Documentation & Team Training

**Objective:** Ensure maintainers and prompt engineers understand the new architecture and responsibilities.

### 4.1. Update Documentation
- Document:
  - Separation of duties (semantic LLM, CRUD code)
  - LLM output contract (semantic, not CRUD), including sub-memory CRUD format
  - Prompt engineering guidelines
  - Coding standards for CRUD execution

### 4.2. Team Training
- Walk through new workflow and responsibilities with team.

---

## Implementation Order & Rationale

- **Phase 1** (Prompt refactor) is independent and can be developed/tested in isolation.
- **Phase 2** (Code refactor) should follow, as it depends on the new prompt logic.
- **Phase 3** (Integration) ensures the new system works as intended.
- **Phase 4** (Documentation/training) ensures long-term maintainability.

Each phase is modular and can be merged/tested independently, minimizing risk and supporting progressive rollout.

---

## Implementation Readiness

All architectural, design, and contract details for robust, reliable, and evolvable MIS memory and sub-memory CRUD have been captured in the supporting documentation:
- MIS_Modular_Implementation_Steps.md (this plan)
- MIS_SubMemory_CRUD_Design.md (sub-memory CRUD contract)
- MIS_Stage1_Stage2_Logic_Brainstorm.md (pipeline and logic flow)
- MIS_Crud_Responsibility_Recommendation.md (CRUD responsibility and rationale)

**No further context is required for a successful implementation.**