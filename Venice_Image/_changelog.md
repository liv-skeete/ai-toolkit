# Changelog


## --- Venice Image v2 ---

### v2.3.1 (2025-10-09)
- Fixed asyncio event loop issues in synchronous methods that were causing runtime errors
- Replaced asyncio.get_event_loop().time() with time.time() for performance timing
- Updated logging to handle cases where timing may not be available
- Ensured compatibility with thread pool executors used for blocking operations

### v2.3.0 (2025-10-09)
- Comprehensive code quality improvements with full type annotation coverage
- Enhanced error handling with detailed error messages and graceful API failure recovery
- Strengthened input validation with parameter range checks (steps 1-50, cfg_scale 0-20)
- Added comprehensive inline documentation with docstrings and usage examples
- Externalized all hardcoded values to valve configuration system
- Implemented structured logging throughout with performance metrics
- Completed security review with no identified vulnerabilities
- Verified performance optimization with efficient async operations
- Confirmed code organization follows best practices

### v2.1.2 (2025-10-02)
- Reverted image URL format to `![image]({image_url}){image_url}` for improved assistant compatibility.
- Enforced newline separation for streamed images to fix UI rendering issues.

### v2.1.1 (2025-10-02)
- Fixed image streaming to emit images as each completes instead of batching at the end
- Fixed image URL formatting (removed duplicate URL appending)
- Fixed error message formatting to properly display in chat UI
- All image outputs now stream directly to the assistant message as they're generated

### v2.1.0 (2025-10-01)
- Removed obsolete `MODELS_CACHE_HOURS` valve (no longer used in assistant-driven workflow)
- Removed obsolete `UPSCALE_KEYWORDS` valve (no longer used with assistant-driven workflow)
- Removed unused `_get_json_blocking` method (never called)
- Removed obsolete `_upscale_intent_and_factor` method (assistant now specifies scale)
- Removed unused imports: `json`, `datetime`, `timedelta`
- Fixed typo in `TIMEOUT_SECONDS` description
- Added timeout to source image URL fetches
- Added `_validate_and_clean_b64` helper to reduce code duplication
- Added error handling for valve type conversions
- Code cleanup to align with final assistant-driven workflow

### v2.0.9 (2025-09-30)
- Fixed a bug where the regex for finding the `<details>` block was too greedy, causing issues when multiple `<details>` blocks were present in the message.

### v2.0.8 (2025-09-30)
- Appended the image URL to the response body for improved traceability.

### v2.0.7 (2025-09-30)
- Updated the assistant's guide with correct parameter ranges from the documentation.
- Implemented a final, robust regex-based parsing logic to correctly handle multi-line prompts.
- Added `source_image_url` to the edit and upscale examples in the assistant's guide.

### v2.0.6 (2025-09-30)
- Implemented a correct and robust multi-line prompt parsing logic based on the documentation.

### v2.0.5 (2025-09-30)
- Fixed a critical bug where the parsed prompt was not being passed to the image generation function.

### v2.0.4 (2025-09-30)
- Implemented robust multi-line prompt parsing.

### v2.0.3 (2025-09-30)
- Fixed a bug where multi-line prompts were not parsed correctly.

### v2.0.2 (2025-09-30)
- Changed outlet to preserve the `<details>` block for easier testing.

### v2.0.1 (2025-09-30)
- Removed unused style API call.

### v2.0.0 (2025-09-28)
- Reworked module to accept input directly from the assistant instead of user messages.
- Removed UserValves and message parsing logic.
- Simplified the `pipe` method to directly handle prompts and images.


## --- Venice Image v1 ---

### v1.1.5 (2025-10-09)
- Improved code quality and refactored for better maintainability
- Added proper input validation and error handling
- Enhanced logging and documentation

### v1.1.4 (2025-09-25)
- Added `EMBED_IMAGES_AS_URL` valve to upload images to pictshare and embed URLs instead of base64.
- Added `PICTSHARE_URL` valve for configuring the pictshare service endpoint.

### v1.1.3 (2025-08-30)
- Constrained OUTPUT_FORMAT to enum ['png', 'jpeg', 'webp'] to enable dropdown in UI.

### v1.1.2 (2025-08-30)
- Moved UPSCALE_FACTOR to admin Valves; user setting no longer overrides scale.

### v1.1.1 (2025-08-30)
- Fix: Respect per-user UserValves values passed via __user__.valves with user-first fallback resolution.

### v1.1.0 (2025-08-30)
- Refactored valves into UserValves (user-facing) and Valves (admin/advanced).

### v1.0.0 (2025-08-19)
- Initial release with Venice API integration for generate, edit, upscale, and styles.