"""
title: Dynamic Anthropic Pipeline
description: Anthropic API integration with dynamic model updates and intelligent caching
author: Cody
version: 2.1.2
date: 2025-09-07
"""

import os
import json
import re
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Union, Dict, Optional, AsyncGenerator, Any, Literal
from pydantic import BaseModel, Field
from open_webui.internal.db import get_db
from open_webui.models.memories import Memory
import aiohttp

# Configure logging
logger = logging.getLogger("anthropic")
logger.propagate = False
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class Pipe:
    """Anthropic API pipeline with dynamic model selection and intelligent caching."""

    # Constants
    API_VERSION = "2023-06-01"
    BASE_URL = "https://api.anthropic.com/v1"
    MESSAGES_URL = f"{BASE_URL}/messages"
    MODELS_URL = f"{BASE_URL}/models"

    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    MAX_IMAGE_SIZE_MB = 5
    MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
    REQUEST_TIMEOUT = 300

    # Cache TTL options
    CACHE_TTL_1H = "1h"
    CACHE_TTL_5M = "5m"
    CACHE_TYPE_EPHEMERAL = "ephemeral"

    class Valves(BaseModel):
        """Configuration valves for the pipeline."""

        # Logging Configuration
        verbose_logging: bool = Field(
            default=False,
            description="Enable detailed diagnostic logging",
        )

        # API Configuration
        ANTHROPIC_API_KEY: str = Field(
            default=os.getenv("ANTHROPIC_API_KEY", ""),
            description="Your Anthropic API key",
        )

        # Model Configuration
        MODELS_CACHE_HOURS: int = Field(
            default=1,
            ge=0,
            le=24,
            description="Hours to cache model list (0 to disable caching)",
        )
        DEFAULT_MAX_TOKENS: int = Field(
            default=int(os.getenv("ANTHROPIC_DEFAULT_MAX_TOKENS", "8000")),
            ge=1,
            description="Default max_tokens when not provided in request",
        )

        # Thinking Mode Configuration
        THINKING_BUDGET_RATIO: float = Field(
            default=float(os.getenv("ANTHROPIC_THINKING_BUDGET_RATIO", "0.3")),
            gt=0.0,
            lt=1.0,
            description="Fraction of max_tokens allocated to thinking budget (0-1)",
        )
        STREAMING_ENABLED: bool = Field(
            default=os.getenv("ANTHROPIC_STREAMING_ENABLED", "true").lower() == "true",
            description="Enable streaming responses; when false, forces non-streaming even if requested by body.",
        )
        # Caching Controls
        ENABLE_PROMPT_CACHE: bool = Field(
            default=True,
            description="Enable 1h cache for static system blocks (System Prompt, User Information, [Prompt] memories)",
        )
        ENABLE_CHAT_CACHE: bool = Field(
            default=True,
            description="Enable single 5m cache breakpoint for stable chat history (end of previous assistant turn)",
        )

        # Prompt Memory DB Fetch
        PROMPT_DB_FETCH_ENABLED: bool = Field(
            default=True,
            description="Fetch [Prompt] memories directly from DB using user ID extracted from system 'User Information' block",
        )
        PROMPT_MAX_ITEMS: int = Field(
            default=3,
            ge=0,
            description="Maximum number of [Prompt] memories to include from DB (0 to disable)",
        )
        PROMPT_MAX_CHARS: int = Field(
            default=8000,
            ge=0,
            description="Maximum characters per [Prompt] memory block when injecting into system[]",
        )

    def __init__(self):
        """Initialize the Anthropic pipeline."""
        self.type = "manifold"
        self.id = "anthropic"
        self.valves = self.Valves()
        self.request_id = None

        # Model cache
        self._cached_models: Optional[List[dict]] = None
        self._models_cache_time: Optional[datetime] = None

        logger.info("Dynamic Anthropic Pipeline initialized")
        self._log_verbose(f"Cache duration: {self.valves.MODELS_CACHE_HOURS} hours")
        self._log_verbose(
            f"API Key Loaded: {'*' * 8 if self.valves.ANTHROPIC_API_KEY else 'Not Loaded'}"
        )

    def update_valves(self, new_valves: Dict[str, Any]) -> None:
        """Update valve settings."""
        self._log_verbose("Updating module configuration")
        for key, value in new_valves.items():
            if hasattr(self.valves, key):
                setattr(self.valves, key, value)
                self._log_verbose(f"  └─ Set {key} to {value}")

    # ========== Logging Methods ==========

    def _log_verbose(self, message: str) -> None:
        """Log message only in verbose mode."""
        if self.valves.verbose_logging:
            logger.info(message)

    def _log_payload(self, direction: str, payload: Any) -> None:
        """Log request/response payloads in verbose mode."""
        if not self.valves.verbose_logging:
            return

        try:
            # Convert payload to JSON string for sanitization
            payload_str = (
                json.dumps(payload, indent=2)
                if not isinstance(payload, str)
                else payload
            )

            # Sanitize large base64 data strings using regex for conciseness
            def replacer(match):
                # For "data": "..." blocks, keep surrounding quotes
                prefix = match.group(1)
                data = match.group(2)
                return f'{prefix}"{data[:20]}... ({len(data)} chars)"'

            # Regex to find "data": "base64_string"
            payload_str = re.sub(
                r'("data":\s*)"([^"]+)"',
                replacer,
                payload_str,
                flags=re.DOTALL,
            )

            logger.info(f"{direction} PAYLOAD (sanitized):\n---\n{payload_str}\n---")
        except Exception as e:
            logger.warning(f"Failed to log {direction.lower()} payload: {e}")

    def _log_cache_usage(self, usage: Dict[str, Any], is_stream: bool = False) -> None:
        """Log cache usage statistics."""
        if not usage:
            return

        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_created = usage.get("cache_creation_input_tokens", 0)
        input_tokens = usage.get("input_tokens", 0)

        # Extract cache creation breakdown if available
        cache_creation = usage.get("cache_creation", {})
        cache_5m = cache_creation.get("ephemeral_5m_input_tokens", 0)
        cache_1h = cache_creation.get("ephemeral_1h_input_tokens", 0)

        stream_label = " (stream)" if is_stream else ""
        base_msg = f"CACHE USAGE{stream_label}: read={cache_read}, created={cache_created}, input={input_tokens}"

        # Add TTL breakdown if available
        if cache_creation and (cache_5m > 0 or cache_1h > 0):
            base_msg += f" [5m={cache_5m}, 1h={cache_1h}]"

        logger.info(base_msg)

        # Always log cache hits with the special emoji
        if cache_read > 0:
            logger.info(f"🎯 CACHE HIT: {cache_read} tokens read from cache")

    # ========== Model Management ==========

    def pipes(self) -> List[dict]:
        """Return available models."""
        self._log_verbose("pipes() method called - retrieving model list")
        return self._get_models()

    def _get_models(self) -> List[dict]:
        """Get models with caching."""
        # Check cache validity
        if self._is_cache_valid():
            self._log_verbose(
                f"Using cached models (count: {len(self._cached_models)})"
            )
            return self._cached_models

        # Fetch fresh models
        self._log_verbose("Fetching fresh models from API")
        api_models = self._fetch_models_from_api()

        if api_models:
            transformed_models = self._transform_api_models(api_models)
            if transformed_models:
                self._cached_models = transformed_models
                self._models_cache_time = datetime.now()
                logger.info(
                    f"Successfully refreshed {len(transformed_models)} models from API"
                )
                return transformed_models

        # API fetch failed - return existing cache if available, otherwise empty list
        logger.warning("API unavailable or returned no models")
        if self._cached_models:
            self._log_verbose(
                f"Using existing cached models (count: {len(self._cached_models)})"
            )
            return self._cached_models
        return []

    def _is_cache_valid(self) -> bool:
        """Check if model cache is still valid."""
        if self.valves.MODELS_CACHE_HOURS <= 0:
            return False

        if not self._cached_models or not self._models_cache_time:
            return False

        cache_age = datetime.now() - self._models_cache_time
        return cache_age < timedelta(hours=self.valves.MODELS_CACHE_HOURS)

    def _fetch_models_from_api(self) -> List[dict]:
        """Fetch available models from Anthropic API."""
        api_key = self.valves.ANTHROPIC_API_KEY
        if not api_key or not api_key.strip():
            logger.warning("No API key available for fetching models")
            return []
        self._log_verbose(f"Using API Key: {'*' * 8}")

        headers = {
            "x-api-key": api_key,
            "anthropic-version": self.API_VERSION,
        }

        try:
            response = requests.get(self.MODELS_URL, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                self._log_verbose(f"Fetched {len(models)} models from API")
                return models
            else:
                logger.warning(f"API returned {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch models from API: {e}")

        return []

    def _transform_api_models(self, api_models: List[dict]) -> List[dict]:
        """Transform Anthropic API models to Open WebUI format."""
        models = []

        for model in api_models:
            if model.get("type") != "model":
                continue

            model_id = model.get("id", "")
            if not model_id:
                continue

            # Determine vision support
            supports_vision = self._check_vision_support(model)
            context_length = model.get("context_length", 200000)

            # Base model
            models.append(
                {
                    "id": model_id,
                    "name": model_id,
                    "context_length": context_length,
                    "supports_vision": supports_vision,
                }
            )

            # Add thinking variant if supported
            if self._supports_thinking(model_id):
                models.append(
                    {
                        "id": f"{model_id}-thinking",
                        "name": f"{model_id}-thinking",
                        "context_length": context_length,
                        "supports_vision": supports_vision,
                    }
                )

        self._log_verbose(
            f"Transformed {len(api_models)} API models into {len(models)} variants"
        )
        return models

    def _check_vision_support(self, model: dict) -> bool:
        """Check if model supports vision/image inputs."""
        input_modalities = model.get("input_modalities", [])
        if isinstance(input_modalities, list):
            if any(m in ("image", "vision") for m in input_modalities):
                return True
        return "vision" in model.get("tags", [])

    def _supports_thinking(self, model_id: str) -> bool:
        """Check if model supports thinking mode."""
        thinking_patterns = ["sonnet-4-", "opus-4-1-", "opus-4-", "3-7-sonnet-"]
        return any(pattern in model_id for pattern in thinking_patterns)

    # ========== Content Processing ==========

    def _process_content(self, content: Union[str, Dict, List[dict]]) -> List[dict]:
        """Process and normalize content to Anthropic format."""
        if content is None:
            return []

        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        if isinstance(content, dict):
            content = [content]
        elif not isinstance(content, list):
            return []

        processed = []
        for item in content:
            processed_item = self._process_content_item(item)
            if processed_item:
                processed.append(processed_item)

        return processed

    def _process_content_item(self, item: Union[str, dict]) -> Optional[dict]:
        """Process a single content item."""
        if isinstance(item, str):
            return {"type": "text", "text": item}

        if not isinstance(item, dict):
            return None

        item_type = item.get("type")

        # Pass through Anthropic native types
        if item_type in [
            "text",
            "image",
            "tool_use",
            "tool_result",
            "thinking",
            "redacted_thinking",
        ]:
            return item

        # Convert OpenAI image format
        if item_type == "image_url":
            return self._convert_image_url(item)

        # Handle unknown types with text field
        if "text" in item and item_type is None:
            return {"type": "text", "text": item["text"]}

        return None

    def _convert_image_url(self, item: dict) -> Optional[dict]:
        """Convert OpenAI image_url format to Anthropic format."""
        url = item.get("image_url", {}).get("url", "")
        if not isinstance(url, str):
            return None

        # Handle base64 data URLs
        if url.startswith("data:image"):
            return self._convert_data_url(url)

        # Handle remote URLs
        return {"type": "image", "source": {"type": "url", "url": url}}

    def _convert_data_url(self, url: str) -> Optional[dict]:
        """Convert data URL to Anthropic image format."""
        try:
            mime_type, base64_data = url.split(",", 1)
            header = mime_type[5:] if mime_type.startswith("data:") else mime_type
            media_type = header.split(";")[0]
        except Exception as e:
            raise ValueError(f"Invalid data URL for image: {e}")

        if media_type not in self.SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"Unsupported media type: {media_type}")

        if len(base64_data) * 3 / 4 > self.MAX_IMAGE_SIZE_BYTES:
            raise ValueError(f"Image size exceeds {self.MAX_IMAGE_SIZE_MB}MB limit")

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64_data,
            },
        }

    # ========== Tool Processing ==========

    def _convert_tools(self, tools: List[dict]) -> List[dict]:
        """Convert OpenAI tools to Anthropic format."""
        if not tools:
            return []

        anthropic_tools = []
        for tool in tools:
            converted = self._convert_single_tool(tool)
            if converted:
                anthropic_tools.append(converted)

        return anthropic_tools

    def _convert_single_tool(self, tool: dict) -> Optional[dict]:
        """Convert a single tool to Anthropic format."""
        if not isinstance(tool, dict):
            return None

        # Pass through Anthropic-shaped tools
        if "input_schema" in tool and "name" in tool:
            return {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "input_schema": tool.get("input_schema"),
            }

        # Pass through server-defined tools
        if tool.get("type") and tool.get("type") != "function" and "name" in tool:
            return {k: tool[k] for k in ("type", "name", "max_uses") if k in tool}

        # Convert OpenAI function tools
        if tool.get("type") == "function":
            function = tool.get("function", {})
            return {
                "name": function.get("name"),
                "description": function.get("description"),
                "input_schema": function.get("parameters", {}),
            }

        return None

    # ========== Caching Logic ==========

    def _apply_intelligent_caching(self, system_blocks: List[dict]) -> List[dict]:
        """
        Apply caching to system[]:
        - Static core (System Prompt, User Information, [Prompt]) receives a single 1h ephemeral checkpoint on its tail block.
        - All other system blocks are treated as volatile (no cache).
        - The 5m chat-history checkpoint is applied in _build_payload() on the previous assistant turn, not here.
        """
        static_blocks = []  # System Prompt, User Information, [Prompt] memories
        volatile_blocks = []  # All other system content (e.g., Time Context), uncached

        for block in system_blocks:
            if block.get("type") != "text":
                volatile_blocks.append(block)
                continue

            text = block.get("text", "")
            category = self._categorize_system_block(text)

            if category == "static":
                static_blocks.append(block)
            else:  # volatile
                volatile_blocks.append(block)

        # Rebuild in optimal order with cache controls
        reorganized: List[dict] = []

        # Add static content with 1h cache
        if static_blocks:
            reorganized.extend(static_blocks)
            if getattr(self.valves, "ENABLE_PROMPT_CACHE", True):
                reorganized[-1]["cache_control"] = {
                    "type": self.CACHE_TYPE_EPHEMERAL,
                    "ttl": self.CACHE_TTL_1H,
                }

        # Add volatile content (no cache)
        reorganized.extend(volatile_blocks)

        self._log_verbose(
            f"Cache strategy: {len(static_blocks)} static (1h), {len(volatile_blocks)} volatile"
        )

        return reorganized

    # Helper: extract the second bracketed tag from a "User Memory:" header
    def _extract_user_memory_tag(self, text: str) -> Optional[str]:
        """
        Parse 'User Memory:' header to extract the primary tag immediately after the id.
        Expected formats:
          - 'User Memory: [<id>] [<Tag>] ...'
        Returns the tag (e.g., 'Prompt', 'Session', 'Static') or None if not matched.
        """
        if not isinstance(text, str) or not text.startswith("User Memory:"):
            return None
        m = re.match(r"^User Memory:\s*\[[^\]]+\]\s*\[([^\]]+)\]", text)
        if m:
            return m.group(1).strip()
        return None

    def _is_prompt_memory(self, text: str) -> bool:
        """True only when the 'User Memory:' primary tag is exactly [Prompt]."""
        return self._extract_user_memory_tag(text) == "Prompt"

    def _categorize_system_block(self, text: str) -> str:
        """Categorize a system block for caching strategy."""
        # Check for volatile content
        if text.startswith("Time Context:"):
            return "volatile"

        # Check for static content - only core system components
        if text.startswith("System Prompt:") or text.startswith("User Information:"):
            return "static"

        # Memories: Prompt = static, others = volatile (uncached)
        if text.startswith("User Memory:"):
            if self._is_prompt_memory(text):
                return "static"
            return "volatile"

        # Default to volatile for unknown content
        return "volatile"

    def _extract_user_id_from_text(self, text: str) -> Optional[str]:
        """
        Extract a user ID from a 'User Information:' text block.
        Expected pattern example:
          'User Information: ID: f19a0a64-d2de-4281-bf56-e63b6bdfa758 | User: Philip | Role: admin'
        """
        if not isinstance(text, str) or not text.startswith("User Information:"):
            return None
        m = re.search(r"ID:\s*([0-9a-fA-F\-]{8,})", text)
        return m.group(1) if m else None

    def _extract_user_id_from_system_blocks(
        self, system_blocks: List[dict]
    ) -> Optional[str]:
        """Scan system blocks to find the 'User Information:' block and return the extracted user ID."""
        if not system_blocks:
            return None
        for b in system_blocks:
            if b.get("type") == "text":
                uid = self._extract_user_id_from_text(b.get("text", ""))
                if uid:
                    return uid
        return None

    def _extract_leading_tags(self, content: str) -> List[str]:
        """
        Extract leading bracketed tags from the beginning of a content string.
        Example: "[Prompt] [Personal] Foo" -> ["Prompt", "Personal"]
        """
        tags: List[str] = []
        if not isinstance(content, str):
            return tags
        s = content.lstrip()
        while s.startswith("["):
            end = s.find("]")
            if end == -1:
                break
            tag = s[1:end].strip()
            if tag:
                tags.append(tag)
            s = s[end + 1 :].lstrip()
        return tags

    def _strip_leading_tags(self, content: str) -> str:
        """Remove all leading bracketed tags from a content string."""
        if not isinstance(content, str):
            return ""
        s = content.lstrip()
        while s.startswith("["):
            end = s.find("]")
            if end == -1:
                break
            s = s[end + 1 :].lstrip()
        return s

    def _build_prompt_block(self, mem_id: str, content: str) -> dict:
        """
        Build an Anthropic-compatible text block for a [Prompt] memory.
        Ensures the line conforms to: 'User Memory: [shortId] [Prompt] ...'
        """
        short_id = str(mem_id)[:8] if mem_id else "no-id"
        body = self._strip_leading_tags(content or "").strip()
        text = f"User Memory: [{short_id}] [Prompt] {body}".strip()
        return {"type": "text", "text": text}

    def _fetch_prompt_memories_from_db(self, user_id: str) -> List[dict]:
        """
        Query the primary DB for this user's [Prompt] memories (leading tag),
        return as Anthropic text blocks suitable for system[] injection.
        """
        try:
            if not user_id or not self.valves.PROMPT_DB_FETCH_ENABLED:
                return []

            max_items = max(0, int(self.valves.PROMPT_MAX_ITEMS))
            if max_items == 0:
                return []

            blocks: List[dict] = []
            with get_db() as db:
                # Pull a reasonable recent window to filter in Python by leading tag
                candidates = (
                    db.query(Memory)
                    .filter(Memory.user_id == user_id)
                    .order_by(Memory.created_at.desc())
                    .limit(200)
                    .all()
                )

            for m in candidates or []:
                content = getattr(m, "content", "") or ""
                tags = self._extract_leading_tags(content)
                if tags and tags[0].lower() == "prompt":
                    # Trim excessively long content per valve
                    trimmed = content
                    max_chars = max(0, int(self.valves.PROMPT_MAX_CHARS))
                    if max_chars and len(trimmed) > max_chars:
                        trimmed = trimmed[:max_chars]
                    blocks.append(
                        self._build_prompt_block(getattr(m, "id", ""), trimmed)
                    )

                    if len(blocks) >= max_items:
                        break

            return blocks
        except Exception as e:
            logger.warning(
                f"Failed to fetch [Prompt] memories from DB: {e}", exc_info=False
            )
            return []


    # ========== Main Request Processing ==========

    async def pipe(self, body: Dict) -> Union[str, AsyncGenerator[str, None]]:
        """Process request through the Anthropic pipeline."""
        # Log inbound payload in verbose mode
        self._log_payload("INBOUND", body)

        api_key = self.valves.ANTHROPIC_API_KEY
        if not api_key or not api_key.strip():
            return {"content": "Error: ANTHROPIC_API_KEY is required", "format": "text"}

        try:
            # Extract model name and check for thinking mode
            model_name = body["model"].split("/")[-1].split(".")[-1]
            is_thinking_mode = "-thinking" in body["model"]
            if is_thinking_mode:
                model_name = model_name.replace("-thinking", "")

            # Process messages
            messages = self._normalize_messages(body["messages"])

            # Extract system blocks for caching
            system_blocks = self._extract_system_blocks(messages)


            # Build payload
            payload = self._build_payload(
                model_name=model_name,
                messages=messages,
                system_blocks=system_blocks,
                body=body,
                is_thinking_mode=is_thinking_mode,
            )

            # Build headers
            headers = self._build_headers(body.get("betas"))

            # Log outbound payload in verbose mode
            self._log_payload("OUTBOUND", payload)

            # Execute request
            if payload["stream"]:
                return self._stream_response(headers, payload)
            else:
                return await self._non_stream_response(headers, payload)

        except Exception as e:
            logger.error(f"Pipe error: {e}")
            return f"Error: {str(e)}"

    def _normalize_messages(self, messages: List) -> List[dict]:
        """Normalize messages to Anthropic format."""
        normalized = []

        for message in messages:
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", message.get("text", ""))
                normalized.append(
                    {"role": role, "content": self._process_content(content)}
                )
            elif isinstance(message, str):
                normalized.append(
                    {"role": "user", "content": self._process_content(message)}
                )

        return normalized

    def _extract_system_blocks(self, messages: List[dict]) -> List[dict]:
        """Extract all system message blocks."""
        blocks = []
        for message in messages:
            if message["role"] == "system":
                content = message.get("content")
                if isinstance(content, list):
                    blocks.extend(content)
        return blocks

    def _build_payload(
        self,
        model_name: str,
        messages: List[dict],
        system_blocks: List[dict],
        body: dict,
        is_thinking_mode: bool,
    ) -> dict:
        """Build the API request payload."""
        # Extract conversation messages
        conversation = [m for m in messages if m["role"] in ["user", "assistant"]]

        # Partition system blocks:
        # - static_core: System Prompt, User Information
        # - prompt_blocks: "User Memory:" blocks tagged [Prompt]
        # - time_context_blocks: Time Context (volatile, uncached)
        # - non_prompt_memory_blocks: other "User Memory:" blocks (non-Prompt) relocated to the tail of the last user message (volatile, uncached)
        # - other_volatile_blocks: non-text/unknown (volatile, uncached)
        static_core_blocks: List[dict] = []
        prompt_blocks: List[dict] = []
        time_context_blocks: List[dict] = []
        non_prompt_memory_blocks: List[dict] = []
        other_volatile_blocks: List[dict] = []

        for block in system_blocks or []:
            if block.get("type") != "text":
                other_volatile_blocks.append(block)
                continue

            text = block.get("text", "")
            if text.startswith("User Memory:"):
                if self._is_prompt_memory(text):
                    prompt_blocks.append(block)
                else:
                    non_prompt_memory_blocks.append(block)
                continue

            if text.startswith("Time Context:"):
                time_context_blocks.append(block)
                continue

            # Treat remaining text (e.g., System Prompt, User Information) as static core
            static_core_blocks.append(block)

        # Apply single chat-history checkpoint (5m) at the end of the previous assistant turn
        if self.valves.ENABLE_CHAT_CACHE:
            if isinstance(conversation, list) and len(conversation) >= 2:
                # Search from the second-to-last message backward for the last assistant turn
                for idx in range(len(conversation) - 2, -1, -1):
                    if conversation[idx].get("role") == "assistant":
                        blocks = conversation[idx].get("content")
                        if isinstance(blocks, list):
                            for j in range(len(blocks) - 1, -1, -1):
                                b = blocks[j]
                                if isinstance(b, dict) and b.get("type") == "text":
                                    b["cache_control"] = {
                                        "type": self.CACHE_TYPE_EPHEMERAL,
                                        "ttl": self.CACHE_TTL_5M,
                                    }
                                    break
                        break

        # Relocate Time Context and non-[Prompt] memories to the tail of the last user message (uncached, discrete blocks)
        relocated_tail: List[dict] = []
        if isinstance(time_context_blocks, list):
            relocated_tail.extend(time_context_blocks)
        if isinstance(non_prompt_memory_blocks, list):
            relocated_tail.extend(non_prompt_memory_blocks)
        if relocated_tail:
            # Find the last user message; create one if absent
            last_user = None
            for m in reversed(conversation):
                if m.get("role") == "user":
                    last_user = m
                    break
            if last_user is None:
                last_user = {"role": "user", "content": []}
                conversation.append(last_user)

            if not isinstance(last_user.get("content"), list):
                last_user["content"] = []

            # Append each block discretely, stripping any cache_control
            for blk in relocated_tail:
                if isinstance(blk, dict):
                    blk.pop("cache_control", None)
                    if blk.get("type") == "text" and isinstance(blk.get("text"), str):
                        last_user["content"].append(blk)

        # Ensure [Prompt] memories from DB are included, merging with any from payload
        if self.valves.PROMPT_DB_FETCH_ENABLED:
            try:
                user_id = self._extract_user_id_from_system_blocks(system_blocks or [])
                if user_id:
                    # Fetch canonical list from DB, ordered by created_at
                    db_prompt_blocks = self._fetch_prompt_memories_from_db(user_id)
                    
                    # Create a final list, starting with the DB's canonical order
                    final_prompt_blocks = list(db_prompt_blocks)
                    existing_texts = {b.get("text") for b in final_prompt_blocks}

                    # Add any unique memories from the initial payload (maintaining DB order)
                    for b in prompt_blocks:
                        if b.get("text") not in existing_texts:
                            final_prompt_blocks.append(b)
                    
                    prompt_blocks = final_prompt_blocks
            except Exception as e:
                logger.warning(f"Prompt DB fetch skipped due to error: {e}")

        # Rebuild system blocks: static core, then [Prompt], then other volatile
        # (Time Context and non-[Prompt] memories were relocated to the user tail and must not appear here)
        reordered_system_blocks = (
            static_core_blocks + prompt_blocks + other_volatile_blocks
        )

        # Base payload
        payload = {
            "model": model_name,
            "messages": conversation,
            "max_tokens": int(body.get("max_tokens", self.valves.DEFAULT_MAX_TOKENS)),
            "stream": (bool(body.get("stream", True)) and self.valves.STREAMING_ENABLED),
        }
        if body.get("stream", True) and not self.valves.STREAMING_ENABLED:
            logger.info(
                "Streaming requested but disabled by STREAMING_ENABLED valve; using non-streaming mode."
            )

        # Add system messages with caching (1h TTL applied to last static block)
        if reordered_system_blocks:
            payload["system"] = self._apply_intelligent_caching(reordered_system_blocks)

        # Add sampling parameters
        self._add_sampling_params(payload, body, is_thinking_mode)

        # Add tools
        if "tools" in body:
            payload["tools"] = self._convert_tools(body["tools"])

        # Add tool choice
        if "tool_choice" in body:
            tool_choice = body["tool_choice"]
            if isinstance(tool_choice, str):
                tool_choice = {"type": tool_choice}
            payload["tool_choice"] = tool_choice

        # Configure thinking mode
        if is_thinking_mode:
            self._configure_thinking(payload, body)

        # Validate constraints
        self._validate_payload(payload, is_thinking_mode)

        return payload

    def _add_sampling_params(
        self, payload: dict, body: dict, is_thinking_mode: bool
    ) -> None:
        """Add sampling parameters to payload."""
        if not is_thinking_mode:
            if body.get("temperature") is not None:
                payload["temperature"] = float(body["temperature"])
            if body.get("top_k") is not None:
                payload["top_k"] = int(body["top_k"])

        if body.get("top_p") is not None:
            top_p = float(body["top_p"])
            if is_thinking_mode:
                top_p = max(0.95, min(1.0, top_p))
            else:
                top_p = max(0.0, min(1.0, top_p))
            payload["top_p"] = top_p

    def _configure_thinking(self, payload: dict, body: dict) -> None:
        """Configure thinking mode parameters."""
        max_tokens = payload["max_tokens"]

        # Check for interleaved thinking
        betas = body.get("betas", [])
        if isinstance(betas, str):
            betas = [betas]
        interleaved = "interleaved-thinking-2025-05-14" in betas

        # Calculate thinking budget
        explicit_budget = body.get("thinking_budget_tokens")
        if explicit_budget is not None:
            budget = int(explicit_budget)
        else:
            ratio = self.valves.THINKING_BUDGET_RATIO
            ratio = max(0.0, min(ratio, 0.99))
            budget = max(1, int(max_tokens * ratio))
            if not interleaved and max_tokens > 1024:
                budget = max(1024, budget)

        # Apply budget
        if max_tokens > 1:
            has_tools = bool(payload.get("tools"))
            if not (interleaved and has_tools) and budget >= max_tokens:
                budget = max(1, max_tokens - 1)

            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
        else:
            logger.warning("Thinking requested but max_tokens <= 1; disabling")

    def _validate_payload(self, payload: dict, is_thinking_mode: bool) -> None:
        """Validate payload constraints."""
        # Validate tool choice with thinking mode
        if is_thinking_mode and payload.get("tools") and "tool_choice" in payload:
            tc = payload.get("tool_choice", {})
            tc_type = tc.get("type") if isinstance(tc, dict) else None
            if tc_type not in {"auto", "none"}:
                raise ValueError(
                    "tool_choice must be 'auto' or 'none' when thinking is enabled"
                )

        # Force streaming for very large outputs (only if streaming is enabled)
        if not payload["stream"] and payload["max_tokens"] > 21333:
            if getattr(self.valves, "STREAMING_ENABLED", True):
                logger.warning(
                    "max_tokens > 21333 requires streaming; forcing stream=True"
                )
                payload["stream"] = True
            else:
                logger.warning(
                    "max_tokens > 21333 requires streaming, but streaming is disabled; capping max_tokens to 21333"
                )
                payload["max_tokens"] = 21333

    def _build_headers(self, betas: Optional[Union[str, List[str]]]) -> dict:
        """Build request headers."""
        api_key = self.valves.ANTHROPIC_API_KEY
        headers = {
            "x-api-key": api_key if api_key else "",
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }

        if betas:
            if isinstance(betas, list):
                headers["anthropic-beta"] = ",".join(betas)
            else:
                headers["anthropic-beta"] = str(betas)

        return headers

    # ========== Response Handling ==========

    async def _non_stream_response(self, headers: dict, payload: dict) -> str:
        """Handle non-streaming response."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.MESSAGES_URL,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT),
            ) as response:
                if response.status != 200:
                    return f"Error: HTTP {response.status}: {await response.text()}"

                result = await response.json()
                self._log_verbose("INBOUND API response received")

                # Log cache usage
                usage = result.get("usage", {})
                self._log_cache_usage(usage)

                # Extract and format content
                content_blocks = result.get("content", [])
                return self._extract_and_format_content(content_blocks)

    def _extract_and_format_content(self, content_blocks: List[dict]) -> str:
        """Extract and format thinking and text content from response blocks."""
        thinking_parts = []
        text_parts = []

        if not isinstance(content_blocks, list):
            return ""

        for block in content_blocks:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")
            if block_type == "thinking" and "thinking" in block:
                thinking_parts.append(block["thinking"])
            elif block_type == "text" and "text" in block:
                text_parts.append(block["text"])

        full_response = ""
        if thinking_parts:
            full_response += f"<thinking>{''.join(thinking_parts)}</thinking>\n"
        
        full_response += "".join(text_parts)
        return full_response

    async def _stream_response(
        self, headers: dict, payload: dict
    ) -> AsyncGenerator[str, None]:
        """Handle streaming response."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.MESSAGES_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT),
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        yield f"Error: HTTP {response.status}: {error_text}"
                        return

                    async for chunk in self._process_stream(response):
                        yield chunk

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"Stream Error: {str(e)}"

    async def _process_stream(self, response) -> AsyncGenerator[str, None]:
        """Process streaming response chunks with simple start/end tags for thinking."""
        buffer = ""
        in_thinking_block = False

        async for chunk in response.content:
            try:
                text = chunk.decode("utf-8", errors="ignore")
            except Exception:
                continue

            buffer += text

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line or not line.startswith("data:"):
                    continue

                data_str = line[6:].strip()

                try:
                    data = json.loads(data_str)
                    event_type = data.get("type")

                    if event_type == "message_start":
                        usage = data.get("message", {}).get("usage", {})
                        self._log_cache_usage(usage, is_stream=True)

                    elif event_type == "content_block_start":
                        block_type = data.get("content_block", {}).get("type")
                        if block_type == "thinking":
                            in_thinking_block = True
                            yield "<thinking>"

                    elif event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")
                        elif delta.get("type") == "thinking_delta":
                            yield delta.get("thinking", "")

                    elif event_type == "content_block_stop":
                        if in_thinking_block:
                            yield "</thinking>"
                            in_thinking_block = False
                    
                    elif event_type == "message_delta":
                        # Check for refusal
                        delta = data.get("delta", {})
                        if delta.get("stop_reason") == "refusal":
                            logger.warning("Stream refused by API policy")
                            yield "Error: The response was refused for violating Anthropic's policy. Please adjust your prompt and try again."
                            return

                        usage = data.get("usage", {})
                        if usage and usage.get("completion_tokens", 0) % 8 == 0:
                            yield ""

                    elif event_type == "error":
                        error = data.get("error", {})
                        logger.error(f"Stream error from API: {error}")
                        yield f"Error: {error.get('message', 'Unknown error from API')}"

                except json.JSONDecodeError:
                    self._log_verbose(f"Failed to decode stream data: {data_str!r}")

