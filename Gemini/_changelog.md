# Changelog

## v1.3.4 (2025-10-12)

**Fixed**
- Resolved "Unclosed client session" error by implementing proper resource management for the Google GenAI client
- Added graceful shutdown mechanism for aiohttp transport used by the Google GenAI library

**Enhanced**
- Improved logging configuration with module-level setup
- Added caching for Google GenAI client to improve performance
- Bumped version to 1.3.4

## v1.3.3 (2025-10-12)

**Initial Release**
- First stable version with model listing and chat capabilities
- Support for Google Generative AI and Vertex AI
- Multimodal input (text and images)
- Safety settings and retry logic
- Streaming and non-streaming response handling