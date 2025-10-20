# Standard Memory Categories

This document defines the standard memory categories used across all Automatic Memory Manager (AMM) modules. All memories must be formatted using these categories with the standardized format:

Category Name:
- Memory item 1
- Memory item 2
- Memory item 3

---

## Categories and Descriptions (Ordered by Importance & Permanence)

### 1. User Profile
**Description:** Core identity information about the user, including name, location, key relationships, and fundamental identity facts.

**Management Considerations:**
- High stickiness - rarely changes
- Strict replacement only on explicit correction
- Low volume expected
- High retrieval priority

**Examples:**
User Profile:
- Name is Tom
- Lives in Seattle, Washington
- Married to Sarah for 12 years
- Works as a software engineer at TechCorp
- Has two children: Emma (10) and James (7)
  - Emma enjoys soccer
  - James plays the violin

---

### 2. Health & Wellbeing
**Description:** Information about the user's health conditions, medications, symptoms, wellness routines, or mental health considerations. Safety-critical; use for health-specific info.

**Management Considerations:**
- High importance for safety
- Variable stickiness (allergies permanent, temporary conditions may resolve)
- Careful replacement needed
- Medium volume expected
- High retrieval priority for safety-critical information

**Examples:**
Health & Wellbeing:
- Takes medication for high blood pressure every morning
- Follows a gluten-free diet
- Exercises 3 times per week (running and weight training)
- Meditates for 10 minutes daily
- Has a mild peanut allergy

---

### 3. Preferences & Values
**Description:** User's likes, dislikes, priorities, beliefs, and communication style preferences that aren't direct instructions. Use for subjective, non-instructional preferences.

**Management Considerations:**
- Medium-high stickiness
- Replacement based on recency and explicitness
- Medium volume expected
- Allow natural evolution over time

**Examples:**
Preferences & Values:
- Enjoys dancing in the rain with Tiggly and Jeff
- Dislikes walking in the rain
- Prefers prompts formatted in markdown
- Values brevity over verbosity in explanations
- Believes in sustainable technology practices

---

### 4. Assistant Instructions
**Description:** Explicit rules and directives about how the assistant should behave, respond, or format information. Use for direct instructions, not general preferences.

**Management Considerations:**
- High stickiness - follow consistently until changed
- Strict replacement only by user command
- Low volume expected
- High retrieval priority

**Examples:**
Assistant Instructions:
- Should be addressed as Tom
- Use markdown formatting in all responses
- Provide code examples when explaining technical concepts
- Avoid using emojis in responses
- Always confirm before executing potentially destructive operations

---

### 5. Goals & Aspirations (Suggested Addition)
**Description:** Long-term objectives, life plans, and aspirations that guide the user's decisions and priorities.

**Management Considerations:**
- Medium stickiness - may evolve over time
- High value for personalization and planning
- Should be reviewed periodically for relevance

**Examples:**
Goals & Aspirations:
- Wants to run a marathon within the next two years
- Aspires to become a published author
- Plans to retire early and travel the world
- Hopes to learn to play the guitar
- Intends to start a family in the next five years

---

### 6. Projects & Tasks
**Description:** Information about specific ongoing projects, work items, technical details, and professional contexts. Use for goal-oriented, time-bounded activities.

**Management Considerations:**
- Relevance often tied to project lifecycle
- May become outdated upon completion
- Medium-high volume potential
- May benefit from time-decay or completion flags
- Consider project-based grouping or archival

**Examples:**
Projects & Tasks:
- Working on redesigning the company website, due June 15
- Needs to finish the quarterly report by Friday
- Leading the AI integration project with 5 team members
- Researching cloud migration options for legacy systems
- Planning to refactor the authentication module next sprint

---

### 7. Skills & Hobbies
**Description:** Information about the user's general abilities, interests, and non-work activities. Use for non-professional, recurring activities.

**Management Considerations:**
- Medium-high stickiness
- Similar to Preferences in update patterns
- Medium volume expected
- Allow natural evolution

**Examples:**
Skills & Hobbies:
- Plays piano at an intermediate level
- Enjoys landscape photography on weekends
- Has been learning Spanish for 2 years
- Collects vintage vinyl records
- Participates in a local hiking group

---

### 8. Contacts & Relationships (Optional)
**Description:** Named people, relationships, and social connections not core to identity (e.g., friends, colleagues, acquaintances).

**Management Considerations:**
- Medium stickiness
- Useful for social and contextual personalization
- May overlap with User Profile for core relationships

**Examples:**
Contacts & Relationships:
- Friend: Alex Johnson (met at university)
- Colleague: Priya Singh (works in marketing)
- Neighbor: Mrs. Chen (lives next door)
- Tennis partner: Mark
- Book club: "Seattle Readers" group

---

### 9. Events & Milestones (Optional)
**Description:** Significant life events, achievements, anniversaries, and milestones.

**Management Considerations:**
- Medium stickiness
- Useful for context, reminders, and personalization
- May overlap with Projects & Tasks or User Profile

**Examples:**
Events & Milestones:
- Graduated from Stanford University in 2012
- Completed first triathlon in 2021
- 10-year wedding anniversary in 2023
- Promoted to Senior Engineer in 2022
- Bought first home in 2018

---

### 10. Reminders
**Description:** Actionable items the user wants to remember, often time-sensitive. Use for transient, actionable items.

**Management Considerations:**
- Low stickiness - transient by nature
- Tied to completion or time expiration
- High turnover expected
- Needs clear completion mechanism
- Formatted as a bulleted list

**Examples:**
Reminders:
- Call mom on her birthday (May 15)
- Renew passport before international trip
- Pick up dry cleaning on Thursday
- Schedule dentist appointment
- Submit expense reports by end of month

---

### 11. Shopping List
**Description:** Items the user intends to purchase. Use for purchase-specific, high-turnover items.

**Management Considerations:**
- Low stickiness - transient by nature
- Tied to completion (purchase)
- High turnover expected
- Needs clear completion mechanism
- Formatted as a bulleted list

**Examples:**
Shopping List:
- Milk
- Bread
- Eggs
- Laundry detergent
- Batteries for smoke detector

---

### 12. Facts & Knowledge
**Description:** General knowledge points, specific facts remembered about topics, people, places, or things the user has shared. Use as a fallback for facts not fitting other categories.

**Management Considerations:**
- Variable stickiness (some facts timeless, others may become outdated)
- Replacement based on contradiction or recency
- High volume potential - may need consolidation
- May benefit from relevance-based pruning

**Examples:**
Facts & Knowledge:
- The user's favorite book is "Dune" by Frank Herbert
- User's dog: Rover is a 3-year-old golden retriever
- User attended Stanford University from 2008-2012
- User's favorite restaurant in Seattle is Canlis
- User is allergic to shellfish

---

### 13. Current Conversation Context
**Description:** Short-term details relevant to the immediate interaction that may not warrant long-term storage. Use for ephemeral, session-based info.

**Management Considerations:**
- Very low stickiness - ephemeral
- High turnover expected
- Aggressive FIFO or time decay (session-based)
- May not belong in long-term memory at all

**Examples:**
Current Conversation Context:
- User is currently troubleshooting a Python error
- User mentioned they're in a hurry and need quick answers
- User is accessing the system from a mobile device
- User is preparing for a meeting that starts in 15 minutes
- User is looking for a specific document they worked on yesterday

---

### 14. Miscellaneous
**Description:** Information that doesn't clearly fit into other categories but is still worth remembering. Use only as a last resort; encourage re-categorization.

**Management Considerations:**
- Variable stickiness
- Ideally low volume
- Periodic review recommended
- Potential re-categorization by MMC
- Possible aggressive pruning if volume grows

**Examples:**
Miscellaneous:
- User mentioned a strange dream about flying elephants
- User's lucky number is 42
- User once met a famous actor at an airport
- User has a collection of unusual paperweights
- User prefers window seats on airplanes

---

## Implementation Guidelines

1. **Format Consistency:** All memories must follow the `Category Name:\n- Item` format for consistent parsing across modules.

2. **Category Selection:** When categorizing memories, choose the most specific appropriate category. Use Miscellaneous only when no other category clearly applies.

3. **Updates vs. New:** When new information is identified that belongs to an existing category, it should be added as a new bullet point under that category header rather than creating a duplicate category header.

4. **Parsing Considerations:** All modules must be able to parse this standardized format, identifying category blocks and individual memory items within each category.

5. **Format Preservation:** When updating memories, modules must preserve the standardized format, maintaining proper formatting with category headers and bullet points.

---

## Category Review & Overlap Notes

- **Preferences & Values vs. Skills & Hobbies:** Clarify that Preferences are subjective likes/dislikes, while Skills & Hobbies are recurring, non-professional activities.
- **Assistant Instructions vs. Preferences:** Use Instructions for explicit rules, not general preferences.
- **Projects & Tasks vs. Goals & Aspirations:** Projects are ongoing, goal-oriented activities; Goals & Aspirations are long-term, high-level objectives.
- **Reminders vs. Shopping List:** Reminders are actionable, time-sensitive items; Shopping List is for purchase-specific items.
- **Facts & Knowledge:** Use as a fallback for facts not fitting other categories.
- **Miscellaneous:** Use only as a last resort; encourage re-categorization.
- **Contacts & Relationships, Events & Milestones:** Optional, but high-utility for social and contextual personalization.

---