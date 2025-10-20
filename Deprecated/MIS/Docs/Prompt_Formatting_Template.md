# Prompt Formatting Template

This document provides a standardized template for formatting prompts across all systems. Following these guidelines ensures consistency, readability, and maintainability.

---

## ⚠️ IMPORTANT JSON FORMATTING RULES

1. **ALWAYS use double braces {{}} for JSON examples that will be parsed by the system**
2. **NEVER use ```json code blocks for actual JSON examples that will be processed**
3. **Double braces are required to escape curly braces in the prompt context**

**CORRECT (will parse properly):**
```markdown
{{"content": "Example content", "importance": 0.8}}
```

**INCORRECT (will cause parsing errors):**
```markdown
```json
{
  "content": "Example content",
  "importance": 0.8
}
```
```

---

## Document Structure

### Title and Introduction
```markdown
# [System Name] ([Version])

❗️ [One-sentence description of the system's role]. [One or two sentences explaining the system's primary task].
```

**Example:**
```markdown
# Memory Identification System (Beta3)

❗️ You are a memory filtering system for an AI assistant. Your task is to analyze the user's message and identify any details that are worth remembering to personalize future interactions.
```

---

## Section Formatting

### Main Sections
Use level-2 headers (##) for main sections. Consider adding relevant emojis to section headers for visual distinction.

```markdown
## 🎯 [SECTION NAME]

[Section description or introduction]
```

**Example:**
```markdown
## 🎯 MEMORY CATEGORIES AND SCORING

Score each based on usefulness, specificity, and repeat relevance.
```

### Subsections
Use level-3 headers (###) for subsections. Include emojis and parameter information when applicable.

```markdown
### 🔖 [Subsection Name] → `[parameter information]`

[Subsection description]
```

**Example:**
```markdown
### 🔖 Direct Memory Commands → `importance: 1.0`

The user clearly asks you to remember, remind, or not forget something.
```

---

## Content Formatting

### Lists
Use bullet points with hyphens (-) rather than greater-than signs (>).

```markdown
- First item
- Second item
  - Nested item
```

### Modifiers and Parameters
Format modifiers and parameters with a "Modifiers:" heading followed by bullet points.

```markdown
**Modifiers:**
- +0.1 for [condition]
- -0.2 for [condition]
```

**Example:**
```markdown
**Modifiers:**
- +0.2 for strong statements ("love", "never")
- -0.2 for uncertainty ("maybe", "for now")
```

### Examples
Format examples with an "Examples:" heading followed by bullet points. Include the expected output using double braces for JSON.

```markdown
**Examples:**
- "[Example input]" → `{{"key": "value"}}`
```

**Example:**
```markdown
**Examples:**
- "I hate cilantro" → `{{"content": "User hates cilantro", "importance": 1.0, "category": "Preferences & Values"}}`
```

### Documentation Code Blocks
Use triple backticks with language specification ONLY for documentation purposes, never for actual JSON examples that will be parsed.

````markdown
```markdown
# Example of how to format a header
```
````

---

## Special Formatting

### JSON with Double Braces
When showing JSON examples that will be parsed by the system, ALWAYS use double braces to escape the curly braces.

```markdown
{{"key": "value", "another_key": "another_value"}}
```

### Section Separators
Use horizontal rules (---) to separate major sections.

```markdown
Content of first section

---

Content of next section
```

---

## Complete Example Section

Here's a complete example of a properly formatted section:

```markdown
### 🧍 Personal Details → `base: 0.7`

Describes user identity, relationships, location, or roles.

**Modifiers:**
- +0.2 if it includes names or places
- +0.1 for counts or timeline

**Examples:**
- "I live in Seattle with two dogs" → `{{"content": "User lives in Seattle with two dogs", "importance": 0.9, "category": "Personal Details"}}`

---
```

---

## Full Document Example

For a complete example of proper formatting, refer to the following files:
- MIS_Identification_Prompt_Beta3.md (for section separators and emoji usage)
- MIS_Identification_Prompt_Beta2.md (for example formatting and bullet point style)

---

## Best Practices

1. **Consistency**: Maintain consistent formatting throughout the document
2. **Visual Clarity**: Use emojis, headers, and section separators to improve readability
3. **Conciseness**: Keep descriptions clear and concise
4. **Examples**: Provide clear examples for each section or concept
5. **Spacing**: Use appropriate spacing between sections and elements
6. **JSON Formatting**: ALWAYS use double braces {{}} for JSON examples, NEVER use ```json code blocks

---

## Implementation Checklist

When creating a new prompt document, ensure it includes:

- [ ] Clear title with version information
- [ ] Concise introduction explaining the system's purpose
- [ ] Well-structured sections with appropriate headers
- [ ] Section separators (---) between major sections
- [ ] Properly formatted examples with bullet points
- [ ] Consistent use of emojis for visual distinction
- [ ] JSON examples using double braces {{}} (NOT ```json code blocks)
- [ ] Complete response format section with examples

---