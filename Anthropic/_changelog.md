# Changelog

## Version 2.2.22 (2025-10-16)

### Fixed

- **Image Size Limit for Pictshare Uploads**: Resolved an issue where base64 images larger than 5MB were incorrectly rejected even when the `EMBED_IMAGES_AS_URL` valve was enabled:
  - Modified the size validation logic in `_convert_data_url` to bypass the 5MB limit when `EMBED_IMAGES_AS_URL` is True
  - Large images will now proceed directly to pictshare upload, where they are converted to URLs and embedded without hitting Anthropic's base64 size limit
  - This change ensures that the `EMBED_IMAGES_AS_URL` feature works as intended for images of any size

## Version 2.2.21 (2025-10-15)

### Fixed

- **Granular Payload Logging Control**: Resolved an issue where payload logging and diagnostic logging were incorrectly interdependent:
  - Separated diagnostic logging for system blocks and user messages to be exclusively controlled by `verbose_logging` valve
  - General payload logging is now exclusively controlled by `log_outbound_payload` valve
  - This ensures that enabling one logging feature doesn't inadvertently trigger the other, providing the granular control needed for debugging
  - The "OUTBOUND USER MESSAGES" diagram now displays with `verbose_logging` instead of `log_outbound_payload`, matching the system block diagnostic logging behavior

## Version 2.2.20 (2025-10-15)

### Features

- **Comprehensive Payload Logging**: Enhanced outbound payload logging to provide complete visibility into both system blocks and user message content:
  - Extended `_log_payload` method to log user message blocks in addition to system blocks
  - Added visual grouping for user message content with clear role identification (user/assistant)
  - Implemented special tagging for User Memory blocks to distinguish [Prompt] vs non-[Prompt] memories
  - Non-[Prompt] User Memories relocated to user messages are now clearly visible in the logging
  - Cache information is displayed for all cached blocks, providing transparency into the cache chain behavior
  - The logging now provides a complete picture of what's being sent to the Anthropic API

## Version 2.2.18 (2025-10-14)

### Fixed

- **Transfer Encoding Error Resolution**: Fixed `TransferEncodingError: 400, message='Not enough data to satisfy transfer length header.'` that was occurring during streaming responses:
  - Increased read buffer size from 1MB to 4MB for better handling of large streaming responses
  - Improved connection timeout configuration with separate connect (30s) and read (60s) timeouts
  - Enhanced error handling for various aiohttp exceptions including ClientPayloadError, ClientResponseError, ServerDisconnectedError, and ClientConnectorError
  - Updated stream processing to use `iter_any()` for more reliable chunk processing
  - Added comprehensive exception handling in both streaming and non-streaming response handlers
  - Improved connection management with proper keep-alive settings

## Version 2.2.17 (2025-10-14)

### Performance Improvements

- **Regex Compilation Optimization**: Converted multiple regex compilations into a single class-level constant to eliminate redundant compilation overhead across different methods, improving efficiency.
- **Deep Copy Usage Optimization**: Eliminated all instances of `copy.deepcopy()` in the `_relocate_images_to_user_message` method, reducing unnecessary memory allocation for large data structures and improving performance with large conversation histories.
- **Verbose Logging Volume Optimization**: Added mechanisms to limit detailed logging output while maintaining diagnostic value:
  - Added configurable `max_log_blocks` valve to limit the number of system blocks logged in detail (default: 10)
  - Added configurable `max_log_block_preview` valve to control the maximum characters shown in system block previews (default: 100)
  - Implemented summary display for remaining blocks when logging is limited
  - Reduced potential log flooding while preserving the diagnostic value of verbose logging

## Version 2.2.16 (2025-10-14)

### Fixed

- **Block Categorization Logic**: Fixed an issue where unknown system blocks were being incorrectly categorized as static_core_blocks instead of other_volatile_blocks:
  - Modified the categorization logic in `_build_payload()` method to only treat explicit System Prompt and User Information blocks as static core
  - All other text blocks (including unknown ones) are now correctly placed in other_volatile_blocks
  - This prevents unknown blocks like memory error alerts from invalidating the entire cache

- **Prompt Memory Concatenation Issue**: Fixed a critical issue where new [Prompt] memories sent by upstream modules were being concatenated to the end of the System Prompt block instead of being processed as separate blocks:
  - Added `_split_combined_system_block()` method to detect and split concatenated System Prompt + memory blocks
  - Enhanced `_extract_system_blocks()` to process both list and string content types
  - Added detailed verbose logging to track system block extraction and splitting
  - Ensured proper categorization of split blocks for caching and payload assembly
  - This fix resolves the issue where the most recent [Prompt] memory would appear missing from the outbound payload and correctly separates all [Prompt] memories for independent caching

## Version 2.2.14 (2025-10-14)

### Improvements

- **Code Quality Enhancements**: Implemented comprehensive code improvements for better maintainability and performance:
  - Enhanced code documentation and comments throughout the pipeline
  - Added comprehensive type hints for improved code clarity and IDE support
  - Optimized regex patterns for better performance
  - Simplified complex conditional logic for improved readability
  - Improved variable naming for better code comprehension
  - Added more comprehensive error handling with detailed logging
  - Removed redundant code and eliminated code duplication

## Version 2.2.13 (2025-10-14)

### Fixed

- **Critical Syntax and Logic Errors**: Fixed multiple high-priority issues that could cause runtime errors:
  - Fixed `_extract_and_format_content` method to remove incorrect `await` calls since it's a synchronous method
  - Removed unnecessary `int()` casts in `_build_web_tools` method for values that are already integers
  - Fixed `await` calls outside of async functions in `_extract_and_format_content` method

## Version 2.2.12 (2025-10-14)

### Fixed

- **Prompt Memory Logic Restoration**: Restored [Prompt] memory identification and ordering logic in API payload assembly while keeping database fetching removed:
  - Restored helper methods: `_extract_user_memory_tag()`, `_is_prompt_memory()`, `_categorize_system_block()`
  - Updated `_apply_intelligent_caching()` method to use restored categorization logic
  - Updated `_build_payload()` method to correctly separate prompt blocks from other memory blocks
  - This ensures proper API payload ordering for [Prompt] memories that are now injected by an upstream module

## Version 2.2.11 (2025-10-12)

### Fixed

- **Orphaned Method Reference**: Fixed remaining references to removed `_is_prompt_memory()` method by implementing the logic directly using `_extract_user_memory_tag()`:
  - Updated `_categorize_system_block()` method
  - Updated `_build_payload()` method

### Removed

- **Prompt Memory Database Integration**: Removed all database-related functionality for prompt memories:
  - Removed valves: `PROMPT_DB_FETCH_ENABLED`, `PROMPT_MAX_ITEMS`, `PROMPT_MAX_CHARS`
  - Removed methods: `_fetch_prompt_memories_from_db()`, `_build_prompt_block()`, `_strip_leading_tags()`, `_extract_user_id_from_text()`, `_extract_user_id_from_system_blocks()`, `_is_prompt_memory()`
  - Removed database import: `get_db`, `Memory`
  - Cleaned up `_build_payload()` method to remove prompt memory DB fetch logic
  - This functionality is now handled exclusively by the MRS module

## Version 2.2.10 (2025-10-12)

### Features

- **Independent Outbound Payload Logging**: Added a new valve `log_outbound_payload` to control outbound payload logging separately from `verbose_logging`. This allows users to enable outbound payload logging without enabling verbose logging for other components.

## Version 2.2.9 (2025-10-02)

### Critical Fixes

#### Duplicate Image Blocks and Missing Image History (Lines 996-1053)

**Problem Identified:**
The pipeline was experiencing two interconnected bugs that broke Sol's visual coherence:
1. **Duplicate image blocks**: When Sol used the `![image](URL)URL` format, the image would appear twice in the API payload
2. **Missing earlier images**: Previously embedded images from earlier in the conversation would disappear from context

**Root Cause Analysis:**
The image relocation logic in `_build_payload()` had a fundamental architectural flaw. Open WebUI sends the complete conversation history with every API request, but the pipeline was processing ALL assistant messages in the conversation on EVERY turn:

```
Turn 1: Sol posts ![image](URL1)URL1
  - Pipeline strips markdown from Sol's message → relocates to user message ✓
  
Turn 2: User sends new message with full history
  - Pipeline sees Sol's Turn 1 message STILL has markdown in history
  - Strips it AGAIN → relocates to THIS user message → duplicate created ✗
  - Line 1031: last_user_message["content"] = [] → wipes out Turn 1 images ✗
```

This created a compounding problem:
- Each turn would re-process old assistant messages
- Each re-processing would create duplicate image blocks
- The content wipe would remove previously relocated images
- Sol would lose visual memory of images she had embedded

**The Fix:**

1. **Process only the most recent assistant message** (lines 1006-1036):
   - Added logic to find the index of the last assistant message
   - Changed from iterating all messages to processing only `conversation[last_assistant_idx]`
   - This ensures each assistant message is processed exactly once, when it's new

2. **Preserve existing user message content** (lines 1038-1048):
   - Changed from `last_user_message["content"] = []` (destructive)
   - To checking content type and preserving it:
     - If already a list → keep it
     - If string → process it with `_process_content()` first (parses markdown images)
     - If missing → create empty list
   - This ensures previously relocated images remain in context

**Impact:**
- ✅ **No duplicate image blocks**: Each markdown image processed exactly once
- ✅ **Visual history preserved**: Earlier image blocks remain in the last user message across turns
- ✅ **Sol's dual format works**: `![image](URL)URL` creates one image block + one text URL
- ✅ **Visual coherence maintained**: Sol can "see" images she embedded in previous turns

**Testing Results:**
- User embeds image with `![image](URL)` → creates image block ✓
- Plain text URL → stays as text (no automatic conversion) ✓
- Sol embeds `![image](URL)URL` → one image block + text URL ✓
- Next turn → Sol sees the image, no duplicates, earlier images preserved ✓

This fix took several days to identify because the bug only manifested across multiple conversation turns and required understanding the interaction between Open WebUI's conversation history management and the pipeline's image relocation logic.

## Version 2.2.8 (2025-10-02)

### Summary
This release focuses on resolving a critical visual coherence issue for the assistant, Sol. The core problem was that images embedded by the assistant in her own responses were being stripped and moved, causing her to lose awareness of her own actions and context. This was addressed through a combination of regex fixes and a new, robust architectural approach to image handling.

### Fixes
- **Corrected Image Stripping Regex:** Fixed a bug in the `_build_payload` function where the regex for stripping markdown images from assistant messages was incomplete. It now correctly removes the entire `![image](URL)` tag, including the closing parenthesis, preventing artifact characters from being left in the text.
- **Removed Greedy URL Parsing:** Eliminated the ambiguous and error-prone automatic conversion of plain text URLs into image blocks within the `_process_content` function. This was the root cause of several issues, including duplicate image blocks and merged URLs.

### Architectural Changes
- **Single Source of Truth for Images:** The pipeline now operates on a clear principle: an image block is created **if and only if** the input is explicit markdown (`![image](URL)`) or a structured image type from the frontend. Plain text URLs are always treated as plain text. This aligns the assistant's perception with the user's and removes system fragility.
- **New Assistant Prompting Strategy:** To support this new architecture, a prompting strategy was developed (`![image](URL)URL`) that allows the assistant to embed a visual image while simultaneously creating a plain text URL anchor. This anchor remains after processing, allowing her to maintain positional awareness and a coherent sense of agency across conversational turns.

## 2.2.7 (2025-09-30)

### Added
- **Web Tools Status Emissions**: Added real-time status updates in chat UI for web tool usage:
  - `🌐 Web tools available: web_search, web_fetch` - Shows which tools are enabled
  - `🔍 Searching: [query]` - Shows when web_search starts
  - `✅ Found X search results` - Shows search completion with result count
  - `🌐 Fetching: [url]` - Shows when web_fetch starts
  - `✅ Fetched: [url]` - Shows successful fetch
  - `❌ Web fetch failed: [error_code]` - Shows fetch errors with specific error codes

### Removed
- **Citation System Integration**: Removed complex Open WebUI citation system integration that was buggy and difficult to maintain
  - Removed `_send_citation()` method (~40 lines)
  - Removed all citation buffer tracking code
  - Removed citation processing from streaming and non-streaming responses
  - Citations functionality replaced with simpler, more reliable status emissions
  - Reduced code complexity by ~100 lines total

### Fixed
- **Large Document Support**: Fixed "Chunk too big" error when fetching large documents
  - Increased aiohttp max_line_size to 10MB for handling large SSE responses
  - Increased max_field_size to 10MB for large HTTP fields
  - Added 1MB read buffer for improved performance
  - Web_fetch now properly handles documents up to 10MB without errors

### Changed
- Web tool execution is now fully transparent via status messages instead of citations
- Cleaner, more maintainable code with better user experience

## Version 2.2.6 (2025-09-30)

### Features

- **Web Fetch Error Logging**: Added comprehensive error detection and logging for web_fetch_tool_result errors in non-streaming responses. Errors now emit both server logs and user-visible status messages, improving debugging and user feedback.

### Fixes

- **Critical Indentation Error**: Fixed incorrect indentation in cache logging that could cause runtime issues (line 283).
- **Code Duplication**: Refactored duplicate "find or create last user message" logic into a shared helper method `_get_or_create_last_user_message()`, reducing code by ~20 lines and improving maintainability.

### Performance

- **Library Modernization**: Replaced deprecated `requests` library with `httpx` for synchronous model fetching, ensuring consistency with async code and better performance.

### Cleanup

- **Removed Dead Code**: Eliminated unused imports (`requests`, `asyncio`, `Literal`), unused instance variables (`request_id`), and unused helper methods (`_run_blocking`, `_extract_leading_tags`).
- **Removed Unused Variables**: Cleaned up unused `tool_name` variable assignments in streaming event handlers.

## Version 2.2.5 (2025-09-27)

### Fixes

- **Assistant Image Handling**: Implemented a fix to correctly process markdown images generated in assistant turns. The pipeline now relocates these images to the last user message as valid Anthropic image blocks, making them visible to the model in subsequent turns and preventing API errors.

## Version 2.2.4 (2025-09-25)

### Features

- Image Base64-to-URL Offloading: Added valves `EMBED_IMAGES_AS_URL` and `PICTSHARE_URL`. The pipeline now uploads `data:image/*;base64,...` content to pictshare and embeds the returned URL, dramatically reducing prompt size and enabling multi-image chats without hitting token limits.
- Stable URL Preservation for Caching: Once converted, the image remains a URL across subsequent turns, keeping the chat prefix byte-identical and preserving Anthropic prompt cache hits.

### Fixes

- Data URL Parsing: Corrected media type extraction so `media_type` is a string (e.g., `image/png`) instead of a list, preventing "Unsupported media type" errors.
- Event Loop Safety: Removed `run_until_complete` from within an active event loop by using a synchronous pictshare upload path, resolving "this event loop is already running" errors.

## Version 2.2.3 (2025-09-25)

### Features

- **Image URL Detection**: The pipeline now automatically detects and processes image URLs pasted as plain text in user messages. It uses a regular expression to identify URLs ending in common image formats and transforms them into the appropriate structured format for the Anthropic API.

### Fixes

- **Role-Specific URL Processing**: Resolved an issue where the new image URL detection logic was incorrectly applied to assistant messages, causing API errors. The functionality is now correctly restricted to `user` messages only, ensuring that image content is not sent in assistant turns.

## Version 2.2.2 (2025-09-20)

### Fixes

- **Prompt Memory Fetching**: Replaced the inefficient and unreliable memory fetching logic with a direct database query. The new implementation uses `startswith("[Prompt]")` to accurately and efficiently retrieve all `[Prompt]` tagged memories, resolving the issue of older memories being excluded and ensuring scalability.

## Version 2.2.1 (2025-09-19)

### Features

- **Web Search Citations**: Implemented citation handling for the web search tool. Citations are now parsed from the API response and emitted to the platform's citation UI in a non-blocking manner for both streaming and non-streaming responses, consistent with the platform's citation standards.

## Version 2.2.1 (2025-09-18)

### Fixes

- Web Search citations: ensure <details> block collapses correctly in streaming by separating it from assistant text, starting it on first web_search tool_use, and closing before the first text tokens. Non-stream already prepends the block.

## Version 2.2.0 (2025-09-17)
- Added Anthropic server web tools integration with minimal complexity and prompt-caching compatibility:
  - Auto-inject web_search (web_search_20250305) and web_fetch (web_fetch_20250910) only for supported models.
  - Valves: ENABLE_WEB_SEARCH, ENABLE_WEB_FETCH, WEB_MAX_USES (shared), WEB_ALLOWED_DOMAINS, WEB_BLOCKED_DOMAINS, WEB_USER_LOCATION, WEB_FETCH_CITATIONS_ENABLED, WEB_FETCH_MAX_CONTENT_TOKENS.
  - De-duplication with user-provided tools; user tools take precedence.
  - Header auto-add for web fetch beta (web-fetch-2025-09-10) when fetch tool is present.
  - Kept existing prompt cache structure intact: tools participate in cached prefix via system breakpoint; 5m chat-history remains unchanged.
- Extended server tool passthrough to include allowed/blocked domains, user_location, citations, max_content_tokens, cache_control.
- Added concise verbose logs for injected web tools.

## Version 2.1.2 (2025-09-16)

### Features

*   **Streaming Refusal Handling:** The pipeline now detects `stop_reason: "refusal"` in streaming responses and relays a clear error message to the user. This prevents the application from hanging on a terminated stream and informs the user that their prompt may have violated an API policy.

## Version 2.1.1 (2025-09-14)

### Fixes

*   **Prompt Memory Reliability:** Fixed a critical bug where the pipeline would fail to include all `[Prompt]` tagged memories if some were already present in the inbound request. The logic now reliably fetches all prompt memories from the database and merges them, ensuring the assistant always has its complete set of prompt memories.
*   **Cache Stability:** Changed the ordering of `[Prompt]` memories from `updated_at` to the more stable `created_at` timestamp. This guarantees a consistent order for the memories, improving the reliability and hit rate of the 1-hour system cache.

## Version 2.1.0 (2025-09-12)

### Features

*   **Anthropic Thinking Block Integration:** Implemented handling for Anthropic's `thinking` content blocks to improve user experience with thinking-enabled models.
    *   **Non-Streaming:** `thinking` blocks are now parsed from the API response and prepended to the final text output, wrapped in `<thinking>` tags for UI rendering.
    *   **Streaming:** The pipeline now correctly streams `thinking` content by emitting `<thinking>` start and end tags around the real-time `thinking_delta` chunks, allowing the UI to render the thinking process as it happens.
*   **Cache Preservation:** All changes were implemented as view-only transformations, ensuring that the underlying data structure for API requests and prompt caching remains unaffected.

### Fixes

*   **Streaming Logic:** Refactored the streaming handler to be a simple, stateless processor, resolving issues where thinking chunks were being buffered incorrectly or creating rendering errors in the UI.

### Logging

*   **Concise Logging:** Adjusted logging to be more focused.
    *   Removed verbose per-event logging in streaming mode.
    *   Replaced the full inbound API payload log in non-streaming mode with a simple, timestamped confirmation message.