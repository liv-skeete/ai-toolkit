# MIS LLM-Code Separation of Duties & Refactor Plan

## Purpose

This document defines the desired separation of responsibilities between the LLM (prompt logic) and the code (AMM_MIS_v13_Module.py) for the Memory Identification & Storage (MIS) system. It inventories current and missing functionality, recommends what to restore or move, and outlines a plan to implement these changes for robust, maintainable, and reliable memory operations.

---

## 1. Desired Separation of Duties

| Component | Responsibilities (Should Do) | Should NOT Do |
|-----------|------------------------------|--------------|
| **LLM (Prompts)** | - Analyze user input and existing memories<br>- Assign memory categories<br>- Score semantic similarity and relevance<br>- Decide on intended memory operations (NEW, UPDATE, DELETE, MERGE, REMOVE_ITEM, etc.)<br>- Provide reasoning and structured output for each operation<br>- Handle nuanced logic: implicit/explicit deletion, contradiction, expansion, temporal change, category assignment, merge/split recommendations | - Execute database CRUD operations<br>- Make low-level deterministic decisions about memory storage<br>- Handle ID mapping or database state |
| **Code (AMM_MIS_v13_Module.py)** | - Parse and validate LLM output<br>- Map LLM decisions to deterministic CRUD operations (create, update, delete, merge, remove item)<br>- Ensure data integrity and consistency<br>- Handle memory ID mapping, user validation, and error handling<br>- Provide logging and diagnostics<br>- Enforce anti-patterns and fallback logic if LLM output is ambiguous or incomplete | - Make semantic similarity or category decisions<br>- Infer user intent beyond LLM output<br>- Overwrite nuanced LLM decisions with hardcoded logic (except for safety/consistency) |

---

## 2. Functionality Inventory & Gaps

### A. LLM (Prompt) Functionality

| Functionality | Present in v5? | Present in Code? | Missing/Needs Restoration? | Recommendation |
|---------------|----------------|------------------|---------------------------|----------------|
| Memory category assignment | Yes | Yes (as fallback) | Needs more nuanced prompt logic | Restore/expand in prompt |
| Semantic similarity scoring | Yes (basic) | Yes (as fallback) | Prompt logic is less nuanced than earlier versions | Restore nuanced scoring/merge logic in prompt |
| Implicit/explicit deletion logic | Partial | Yes | Prompt lacks explicit rules/examples for partial deletion, negation, contradiction | Restore detailed deletion/negation logic in prompt |
| Expansion/merge logic | Partial | Yes | Prompt lacks explicit merge/expansion rules | Restore merge/expansion logic in prompt |
| Temporal change handling | Minimal | Yes | Prompt lacks temporal merge/expansion rules | Restore temporal logic in prompt |
| Anti-patterns/failure examples | Minimal | Yes (logging) | Prompt lacks anti-patterns | Add anti-patterns to prompt |
| Output structure for CRUD ops | Yes | Yes | Needs to be strictly enforced | Enforce strict output contract |

### B. Code Functionality

| Functionality | Present? | Should Remain? | Should Move to LLM? | Notes |
|---------------|----------|---------------|---------------------|-------|
| CRUD operation execution | Yes | Yes | No | Code should remain the executor of memory ops |
| Memory ID mapping/validation | Yes | Yes | No | Code responsibility |
| Fallback for ambiguous LLM output | Yes | Yes | No | Code should handle LLM ambiguity safely |
| Semantic similarity/category logic | Yes (as fallback) | No | Yes | Should move to LLM for nuanced decisions |
| Logging/diagnostics | Yes | Yes | No | Code responsibility |

---

## 3. What to Restore or Move

### Restore to LLM (Prompt):

- Nuanced logic for:
  - Implicit/explicit deletion (partial/full, negation, contradiction)
  - Expansion/merge (adding details, combining related info)
  - Temporal changes (preserving history, current vs. past)
  - Category assignment and merge/split recommendations
  - Anti-patterns and failure examples
- Richer examples and explicit directives for all nuanced operations
- Strict output contract: LLM must output structured, unambiguous operation objects (with reasoning, IDs, etc.)

### Move from Code to LLM:

- Any semantic similarity, category, or merge/split logic that is currently duplicated in code but is better handled by the LLM

### Keep in Code:

- All CRUD/database operations
- ID mapping, user validation, error handling
- Logging, diagnostics, and fallback for LLM ambiguity

---

## 4. Implementation Plan

### Step 1: Prompt Refactor

- Restore and expand nuanced logic in both Identification and Integration prompts:
  - Reintroduce lost directives, rules, and examples from earlier versions (see Docs/MIS_Prompt_Memory_Logic_Review.md)
  - Add explicit anti-patterns and failure examples
  - Clarify output structure and contract

### Step 2: Code Refactor

- Remove or minimize semantic similarity/category/merge logic from code that should be handled by LLM
- Ensure code strictly parses and executes LLM output, with robust error handling and logging
- Add fallback logic only for ambiguous or incomplete LLM output

### Step 3: Integration & Testing

- Test end-to-end flow with a variety of nuanced memory operations (deletion, negation, expansion, temporal, etc.)
- Validate that LLM makes correct decisions and code executes them deterministically
- Log and review any cases where LLM output is ambiguous or fails to match code expectations

### Step 4: Documentation & Training

- Document the separation of duties and output contract for future maintainers
- Provide prompt engineering guidelines for future LLM updates
- Train team on new workflow and responsibilities

---

## 5. Summary Table: Responsibilities

| Task | LLM (Prompt) | Code (Module) |
|------|--------------|--------------|
| Memory analysis, category, similarity | X |   |
| CRUD operation execution |   | X |
| Output structure (JSON ops) | X | X (parse/validate) |
| Error handling, logging |   | X |
| Anti-patterns, failure examples | X | X (logging) |

---

## 6. Next Steps

- Review and approve this plan
- Begin prompt and code refactor as outlined above