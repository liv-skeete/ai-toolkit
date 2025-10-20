# AI Toolkit

A comprehensive collection of modular components for Open WebUI, providing essential utilities for conversational AI systems. From context management to multimodal processing, these tools integrate seamlessly into production workflows.

## Overview

Building sophisticated AI applications requires more than just language models—it requires thoughtful infrastructure. This toolkit provides 11 carefully engineered components that handle the unglamorous but essential work: managing conversation context, tracking user state, fetching external information, generating images, and coordinating with external APIs.

Each component is designed as a standalone filter or utility that can be imported independently or work as part of a larger system.

## Components

### 🎯 Core Context & State Management
- **Context_Manager**: Maintains conversation context across sessions with intelligent windowing
- **User_Info**: Tracks and persists user metadata, preferences, and interaction patterns
- **Date_Time**: Context-aware temporal utilities for scheduling and time-based logic

### 🌐 External Integrations
- **Anthropic**: Direct integration with Claude API for alternative model routing
- **Gemini**: Google's multimodal models integration
- **Rss_News**: Fetch and inject real-time news/information into conversations

### 🖼️ Multimodal Processing
- **Venice_Image**: Image generation, editing, and upscaling via Venice.ai API

### 🛠️ Utilities & Infrastructure
- **Utilities**: Common helper functions, validators, and data transformers
- **Reference**: Knowledge base utilities and citation management
- **Pending**: Task tracking and deferred execution helpers

## Author

**Liv Skeete** | [liv@di.st](mailto:liv@di.st)

Part of a comprehensive AI infrastructure suite. See also: [ai-memory-architecture](https://github.com/liv-skeete/ai-memory-architecture) and [smart-model-router](https://github.com/liv-skeete/smart-model-router).

## License

MIT License - See LICENSE file for details
