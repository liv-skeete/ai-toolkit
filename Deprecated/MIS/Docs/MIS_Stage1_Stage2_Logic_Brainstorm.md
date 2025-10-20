# MIS Stage 1 → Stage 2 Logic Flow: Brainstorm & Ideal Division of Labor

## 1. Current/Desired High-Level Flow

### Stage 1: Identification (LLM)
- Input: User message
- Output: List of potential memories, each with:
  - Content (raw memory candidate)
  - Category
  - Importance score (always present, even for low-importance info)
- **No CRUD or integration decisions made here.**

### Stage 2: Integration (LLM)
- Input: 
  - Potential memories from Stage 1
  - Existing memories (with IDs, content, categories)
- For each potential memory:
  - Compares to all existing memories
  - Computes a relevance score (0.0–0.9) for each comparison
  - Identifies the best match (highest relevance) for each potential memory
  - Optionally: recommends which existing memory is the best candidate for integration (but does NOT decide CRUD)
- Output: For each potential memory:
  - List of relevance scores to existing memories
  - Best match (memory ID)
  - Reasoning for each score

### Stage 3: CRUD Decision (Code)
- Input: 
  - LLM output from Stage 2 (relevance scores, best match recommendations)
  - System config (e.g., memory_relevance_threshold)
- For each potential memory:
  - If best match is in the same category and relevance >= threshold: integrate as a new sub-memory (bullet) in the compound memory
  - If no memory of the same category exists: create a new compound memory for that category and add the sub-memory as the first bullet
  - If explicit deletion/negation: decide between partial (UPDATE) or full (DELETE) based on LLM-provided context and code rules
- Output: Deterministic CRUD operation(s) to apply

---

## 2. Pain Points & Failure Modes

- LLM is unreliable at:
  - Applying strict thresholds (may recommend UPDATE when score is below threshold, or vice versa)
  - Making deterministic CRUD decisions (may hallucinate or misapply rules)
- Code is unreliable at:
  - Semantic similarity, nuanced category/intent matching (should not try to "outsmart" the LLM)
- Ambiguity in LLM output can lead to:
  - Inconsistent CRUD ops
  - Loss of information or incorrect merges/deletions

---

## 3. Ideal Division of Labor

| Step | LLM (Prompt) | Code (Module) |
|------|--------------|---------------|
| Stage 1: Extraction & Scoring | Extracts all potential memories, assigns category, importance | Receives and stores output |
| Stage 2: Semantic Comparison | Compares each potential memory to all existing, computes relevance scores, identifies best match (ID), provides reasoning | Receives scores, best match, and reasoning |
| Stage 3: CRUD Decision | (No action) | Applies deterministic rules:<br>- If best match relevance >= threshold and same category: add as sub-memory (bullet) to compound memory<br>- If no memory of that category exists: create new compound memory and add as first bullet<br>- If explicit deletion: DELETE/UPDATE as per context<br>- Always log and validate decisions |

---

## 4. Proposed Robust Process

### 4.1. LLM Output Contract (Stage 2)
- For each potential memory:
  - List of relevance scores to all existing memories (with IDs)
  - Best match (ID, score)
  - Reasoning for each score
  - Explicit flag if the memory is a deletion/negation candidate (with target description)
- **No CRUD operation recommendations.**

### 4.2. Code CRUD Logic
- For each potential memory:
  - If explicit deletion/negation:
    - If target is a specific item in a memory: UPDATE (remove item)
    - If target is an entire memory: DELETE
  - Else:
    - If best match is in the same category and relevance >= memory_relevance_threshold: add as sub-memory (bullet) to the compound memory
    - If no memory of that category exists: create a new compound memory for that category and add the sub-memory as the first bullet
    - If ambiguous or borderline (e.g., relevance == threshold): be strict with category alignment (do not merge across categories), but permissive with adding a new sub-memory (bullet) to the compound memory
- Always log:
  - LLM scores and best match
  - CRUD decision and reasoning
- If LLM output is ambiguous (e.g., multiple best matches, missing scores), fallback to CREATE and log a warning

---

## 5. Open Questions for Brainstorm

- Should the LLM ever recommend a CRUD operation, or should it only provide scores and best match?
- How should the code handle ambiguous or borderline cases (e.g., relevance == threshold)?
  - **Resolution:**  
    - Be strict with category alignment (never merge across categories).
    - Be permissive with creation of new sub-memories (bullets) within a compound memory of the same category.
    - If no memory of a category exists, create a new compound memory for that category.
    - Example:
      ```
      User Profile:
      - Lives in Denver
      - Age is 30
      - Lives with Anna
      ```
    - This ensures robust, human-like memory organization and avoids accidental cross-category merges.
- Should the code allow for configurable merge strategies (e.g., always merge if above threshold, or allow for stricter/looser policies)?
- How should partial deletions (removing a bullet/item) be flagged in LLM output for the code to act on deterministically?
- What is the best way to log and audit the full decision chain for debugging and future improvements?

---

## 6. Next Steps

- Review and refine this process with stakeholders
- Update prompt and code contracts to enforce this separation
- Implement and test with real-world scenarios