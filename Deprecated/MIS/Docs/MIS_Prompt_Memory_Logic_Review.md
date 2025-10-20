# MIS Prompt Memory Operation Logic Review

## Purpose

This document inventories and analyzes the evolution of nuanced memory operation logic in the MIS prompts (Identification and Integration, v1–v5), focusing on implicit/explicit deletions, expansions, negations, contradictions, and related operations. It also compares the latest prompt logic to the actual implementation in `AMM_MIS_v13_Module.py` to identify alignment gaps. The goal is to document lost or weakened logic and provide recommendations for restoring or improving nuanced memory handling in the latest prompts.

---

## Contents

- [Summary Table: Lost/Weakened Logic Inventory](#summary-table-lostweakened-logic-inventory)
- [Detailed Findings by Operation](#detailed-findings-by-operation)
- [Prompt-Code Alignment Gaps](#prompt-code-alignment-gaps)
- [Recommendations](#recommendations)
- [Appendix: Key Lost Directives and Examples](#appendix-key-lost-directives-and-examples)

---

## Summary Table: Lost/Weakened Logic Inventory

| Operation Type         | Present in (Prompt/Ver) | Missing/Weakened in v5 | Example/Directive Lost | Recommendation |
|------------------------|------------------------|------------------------|-----------------------|----------------|
| Implicit Deletion      | v1–v4, MIS_Prompt_v1/v2, Integration v1–v4 | v5 (less explicit) | "I don't like X anymore" triggers UPDATE/DELETE, anti-patterns for partial negation | Restore explicit rules and anti-patterns for implicit deletions |
| Explicit Deletion      | All versions           | v5 (handled, but less nuanced) | "Forget X", "Delete Y" with scope/partial vs. full | Clarify partial vs. full deletion, add more examples |
| Contradiction/Negation | v2–v4, Integration v1–v4, MIS_Prompt_v1/v2 | v5 (less detailed) | Contradictory preferences, temporal negations | Reinstate contradiction/negation rules and merge logic |
| Expansion              | v2–v4, Integration v1–v4, MIS_Prompt_v1/v2 | v5 (weaker) | "I also like Y", "I now prefer Z" | Add explicit expansion/merge rules and examples |
| Temporal Change        | v2–v4, Integration v1–v4, MIS_Prompt_v1/v2 | v5 (minimal) | "I used to like X but now Y" | Restore temporal merge/expansion logic |
| Category Merge/Split   | v2–v4, Integration v1–v4, MMC_Consolidation | v5 (category boundaries stricter, but merge/split logic less explicit) | Merging/splitting compound memories | Clarify when to merge/split, add examples |
| Anti-patterns          | v2–v4, Integration v1–v4, MIS_Prompt_v1/v2 | v5 (few anti-patterns) | Incorrect merges, loss of information | Add anti-patterns and failure examples |

---

## Detailed Findings by Operation

### 1. Implicit and Explicit Deletion

- **Earlier Prompts:**  
  - Explicitly distinguish between explicit deletion ("forget", "delete") and implicit deletion (negation, e.g., "I don't like X anymore").
  - Provide anti-patterns: e.g., do not delete unrelated information, do not replace entire memory unless fully contradicted.
  - Integration prompts (v1–v4) and MIS_Prompt_v1/v2:  
    - "If new information contradicts existing memory, the most recent statement takes priority unless explicitly stated otherwise."
    - "When the user indicates they no longer engage in something, modify only the relevant portion of an existing memory while preserving all other related content."
    - Examples for both full and partial deletions.

- **v5 Prompts:**  
  - Explicit deletion is handled via "Memory Deletion Command" and DELETE: [Target Description], but the distinction between partial and full deletion is less clear.
  - Implicit deletion/negation logic is less explicit; rules for when to UPDATE vs. DELETE are not as detailed.
  - Fewer anti-patterns and failure examples.

- **Lost/Weakened Logic:**  
  - Rules for partial vs. full deletion, and for implicit negation, are less explicit.
  - Anti-patterns for accidental over-deletion or loss of unrelated information are missing.

### 2. Contradiction/Negation

- **Earlier Prompts:**  
  - Contradictory statements (e.g., "I don't like X anymore") trigger UPDATE or DELETE, with rules for preserving non-contradicted information.
  - Temporal negations ("I used to like X but now Y") are handled with merge logic, preserving history.
  - Examples and anti-patterns for incorrect full replacement.

- **v5 Prompts:**  
  - Contradiction/negation logic is present but less detailed; temporal handling is minimal.
  - Fewer examples of how to merge contradictory or temporal information.

- **Lost/Weakened Logic:**  
  - Temporal contradiction handling and merge rules are less explicit.
  - Fewer examples for nuanced contradiction/negation.

### 3. Expansion (Adding Details)

- **Earlier Prompts:**  
  - "When the user adds new but related information, update the existing memory to include the new details while preserving all existing information."
  - Merge logic: "User likes X" + "I also like Y" → "User likes X and Y".
  - Anti-patterns for overwriting instead of merging.

- **v5 Prompts:**  
  - Expansion logic is present but less explicit; merge rules are not as detailed.
  - Fewer examples for multi-detail merges.

- **Lost/Weakened Logic:**  
  - Merge/expansion rules and anti-patterns are less explicit.
  - Fewer examples for correct expansion.

### 4. Temporal Change

- **Earlier Prompts:**  
  - Explicit rules for handling temporal changes: "I used to like X but now Y" → preserve both, with current preference prioritized.
  - Merge logic for temporal context.

- **v5 Prompts:**  
  - Temporal change handling is minimal; not much guidance on preserving history.

- **Lost/Weakened Logic:**  
  - Temporal merge/expansion logic is less explicit.
  - Fewer examples for temporal changes.

### 5. Category Handling and Merge/Split Logic

- **Earlier Prompts:**  
  - Guidance on when to merge or split memories by category and topic.
  - MMC_Consolidation_Prompt_v4: detailed merge/split strategies, standard format, and anti-patterns.

- **v5 Prompts:**  
  - Category boundaries are stricter, but merge/split logic is less explicit.
  - Fewer examples for compound or split memories.

- **Lost/Weakened Logic:**  
  - Merge/split logic and anti-patterns are less explicit.
  - Fewer examples for compound memory handling.

### 6. Anti-patterns and Failure Examples

- **Earlier Prompts:**  
  - Many anti-patterns and failure examples (e.g., incorrect merges, loss of information, combining unrelated memories).
  - Explicit warnings about what not to do.

- **v5 Prompts:**  
  - Fewer anti-patterns and failure examples.

- **Lost/Weakened Logic:**  
  - Anti-patterns and failure examples are largely missing.

---

## Prompt-Code Alignment Gaps

### Overview

The code in `AMM_MIS_v13_Module.py` implements nuanced memory operations, including:
- Explicit and implicit deletion (REMOVE_ITEM, DELETE, UPDATE)
- Expansion and merging (MERGE_UPDATE)
- Contradiction/negation handling
- Category preservation and merge logic
- Fallbacks for failed updates (e.g., fallback to CREATE)
- Logging and validation for memory operations

#### Key Alignment Observations

| Operation Type         | Code Logic Present | Prompt v5 Coverage | Alignment Gap |
|------------------------|-------------------|--------------------|--------------|
| Explicit Deletion      | Yes (REMOVE_ITEM, DELETE) | Yes (Memory Deletion Command) | Partial: prompt less clear on partial vs. full deletion, code supports both |
| Implicit Deletion      | Yes (contradiction, negation, item removal) | Weak | Code supports nuanced partial deletions, prompt lacks explicit rules/examples |
| Expansion              | Yes (MERGE_UPDATE, add bullet) | Weak | Code merges new details, prompt lacks merge/expansion rules/examples |
| Contradiction/Negation | Yes (contradiction triggers UPDATE/DELETE) | Weak | Code handles negation, prompt lacks explicit contradiction/negation rules |
| Temporal Change        | Yes (temporal merge, e.g., "previously", "now") | Minimal | Code supports, prompt lacks temporal merge rules/examples |
| Category Handling      | Yes (category match, merge, fallback) | Yes (strict boundaries) | Code allows merge within category, prompt stricter but less clear on merge/split |
| Anti-patterns          | Yes (logging, validation, fallback) | Minimal | Code prevents over-deletion/overwriting, prompt lacks anti-patterns |

#### Examples of Alignment Gaps

- **Partial Deletion:**  
  - Code: `REMOVE_ITEM` action removes a specific bullet/item from a memory, preserving the rest.
  - Prompt: v5 only mentions DELETE: [Target Description], but does not clarify partial vs. full deletion or provide examples for item-level removal.

- **Implicit Negation:**  
  - Code: Contradictory statements (e.g., "I don't like X anymore") trigger UPDATE or DELETE, with logic to preserve unrelated details.
  - Prompt: v5 lacks explicit rules or examples for implicit negation/contradiction.

- **Expansion/Merge:**  
  - Code: `MERGE_UPDATE` adds new bullet points to existing memories, preserving category and prior details.
  - Prompt: v5 does not provide explicit merge/expansion rules or examples.

- **Temporal Change:**  
  - Code: Handles temporal changes (e.g., "previously", "now") in merge logic.
  - Prompt: v5 lacks temporal merge rules/examples.

- **Anti-patterns:**  
  - Code: Logging and validation prevent accidental loss of information.
  - Prompt: v5 lacks anti-patterns and failure examples.

---

## Recommendations

1. **Restore Explicit Rules for Implicit/Explicit Deletion:**  
   - Add clear directives and examples for both partial and full deletion, including item-level removal and anti-patterns for over-deletion.

2. **Reinstate Contradiction/Negation Logic:**  
   - Add rules and examples for handling contradictory statements, including when to UPDATE vs. DELETE, and how to preserve unrelated information.

3. **Clarify Expansion/Merge Logic:**  
   - Add explicit rules and examples for merging new details into existing memories, including multi-detail merges and anti-patterns for overwriting.

4. **Restore Temporal Change Handling:**  
   - Add rules and examples for handling temporal changes, preserving both current and historical context.

5. **Clarify Category Merge/Split Logic:**  
   - Add guidance and examples for when to merge or split memories by category and topic, including compound and split memory handling.

6. **Add Anti-patterns and Failure Examples:**  
   - Include anti-patterns and failure examples to illustrate incorrect operations and prevent information loss.

7. **Align Prompts with Code Logic:**  
   - Ensure that all nuanced operations supported by the code (partial deletion, merge, temporal change, etc.) are explicitly described and exemplified in the prompts.

---

## Appendix: Key Lost Directives and Examples

### Example: Implicit Deletion (from earlier prompts)

> "When the user indicates they no longer engage in something (e.g., 'I'm not into board games anymore'), modify the existing memory to remove that aspect but keep other relevant details. Use UPDATE operation to remove specific details within a memory. Use DELETE operation only when the entire memory should be removed."

### Example: Contradiction/Negation

> "If new information contradicts existing memory, the most recent statement takes priority unless explicitly stated otherwise. When possible, merge related details rather than fully replacing prior context. Preserve existing information unless it has been explicitly superseded."

### Example: Expansion/Merge

> "When the user adds new but related information (e.g., 'I also train in Muay Thai'), update the existing memory to include the new details while preserving all existing information."

### Example: Temporal Change

> "When the user indicates a change over time (e.g., 'I used to like X but now I prefer Y'), create an update that clearly preserves the temporal relationship. Use phrasing like 'User previously liked X but now prefers Y' rather than just 'User prefers Y'."

---