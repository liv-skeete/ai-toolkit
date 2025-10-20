"""
title: Dynamic Anthropic Pipeline
description: Anthropic API integration with dynamic model updates and intelligent caching
author: Cody
version: 2.2.13
date: 2025-10-14
"""

import os
import json
import re
import logging
from datetime import datetime, timedelta
from typing import (
    List,
    Union,
    Dict,
    Optional,
    AsyncGenerator,
    Any,
    Callable,
    Awaitable,
)
from pydantic import BaseModel, Field
import aiohttp
import base64
import httpx
import copy

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

    # Web tools configuration
    WEB_SEARCH_TOOL_TYPE = "web_search_20250305"
    WEB_FETCH_TOOL_TYPE = "web_fetch_20250910"
    WEB_TOOLS_SUPPORTED_PATTERNS = [
        "claude-opus-4-1-",
        "claude-opus-4-",
        "claude-sonnet-4-",
        "claude-3-7-sonnet-",
        "claude-3-5-sonnet-",
        "claude-3-5-haiku-",
    ]

    class Valves(BaseModel):
        """Configuration valves for the pipeline."""

        # Logging Configuration
        show_status: bool = Field(
            default=True, description="Show web search status in chat"
        )
        verbose_logging: bool = Field(
            default=False,
            description="Enable detailed diagnostic logging",
        )
        log_outbound_payload: bool = Field(
            default=False,
            description="Enable logging of outbound payloads",
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

        # Web Tools Valves
        ENABLE_WEB_SEARCH: bool = Field(
            default=False,
            description="Enable Anthropic server web search tool (auto-injected when model supports)",
        )
        ENABLE_WEB_FETCH: bool = Field(
            default=False,
            description="Enable Anthropic server web fetch tool (auto-injected when model supports)",
        )
        WEB_FETCH_CITATIONS_ENABLED: bool = Field(
            default=True,
            description="Enable citations for fetched content",
        )
        WEB_MAX_USES: int = Field(
            default=5,
            ge=0,
            description="Shared max_uses for web_search and web_fetch; omit when <= 0",
        )
        WEB_USER_LOCATION: Optional[dict] = Field(
            default=None,
            description="Approximate user location for search localization",
        )
        WEB_FETCH_MAX_CONTENT_TOKENS: int = Field(
            default=50000,
            ge=0,
            description="Max content tokens included from fetched documents (approximate)",
        )
        # Pictshare Configuration
        PICTSHARE_URL: Optional[str] = Field(
            default="http://pictshare:8080",
            description="URL for pictshare service (internal or public)",
        )
        EMBED_IMAGES_AS_URL: bool = Field(
            default=False,
            description="Upload images to pictshare and embed as URLs instead of base64",
        )

    def __init__(self):
        """Initialize the Anthropic pipeline."""
        self.type = "manifold"
        self.id = "anthropic"
        self.valves = self.Valves()
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
        """Log request/response payloads based on direction and valve settings."""
        # Check if logging is enabled for this direction
        if direction == "INBOUND" and not self.valves.verbose_logging:
            return
        elif direction == "OUTBOUND" and not self.valves.log_outbound_payload:
            return

        try:
            # Convert payload to JSON string for sanitization
            payload_str = (
                json.dumps(payload, indent=2)
                if not isinstance(payload, str)
                else payload
            )

            # Sanitize image data in markdown format `![image](data:image/...`
            if "![image](data:image" in payload_str:
                parts = payload_str.split("base64,")
                sanitized_parts = []
                for i, part in enumerate(parts):
                    if i == 0:
                        sanitized_parts.append(part)
                        continue

                    # Find the end of the base64 string
                    end_of_data = -1
                    for char in [")", '"', "}"]:
                        pos = part.find(char)
                        if pos != -1:
                            if end_of_data == -1 or pos < end_of_data:
                                end_of_data = pos

                    if end_of_data != -1:
                        original_data = part[:end_of_data]
                        sanitized_data = (
                            f"{original_data[:20]}... ({len(original_data)} chars)"
                        )
                        sanitized_parts.append(f"{sanitized_data}{part[end_of_data:]}")
                    else:
                        sanitized_parts.append(part)

                payload_str = "base64,".join(sanitized_parts)

            logger.info(f"{direction} PAYLOAD (sanitized):\n---\n{payload_str}\n---")
        except Exception as e:
            logger.warning(f"Failed to log {direction.lower()} payload: {e}")

    def _log_cache_usage(
        self, usage: Dict[str, Any], is_stream: bool = False, **kwargs
    ) -> None:
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
        """Fetch available models from Anthropic API (synchronous for pipes() compatibility)."""
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
            # Use httpx for sync requests (more modern and consistent with async code)
            with httpx.Client(timeout=10) as client:
                response = client.get(self.MODELS_URL, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("data", [])
                    self._log_verbose(f"Fetched {len(models)} models from API")
                    return models
                else:
                    logger.warning(f"API returned {response.status_code}: {response.text}")
        except httpx.RequestError as e:
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

    # Web tools capability checks (supported models only; no discovery)
    def _supports_web_search_capability(self, model_id: str) -> bool:
        return any(p in model_id for p in self.WEB_TOOLS_SUPPORTED_PATTERNS)

    def _supports_web_fetch_capability(self, model_id: str) -> bool:
        return any(p in model_id for p in self.WEB_TOOLS_SUPPORTED_PATTERNS)

    # ========== Content Processing ==========

    def _process_content(self, content: Union[str, Dict, List[dict]]) -> List[dict]:
        """Process and normalize content to Anthropic format."""
        if content is None:
            return []

        if isinstance(content, str):
            # Parse markdown images: ![alt](URL)
            # This matches ONLY explicit markdown syntax, not plain URLs
            image_pattern = re.compile(
                r"!\[[^\]]*\]\((data:image/[^;]+;base64,[^)]+|(?:https?://[^\s)]+\.(?:png|jpg|jpeg|gif|webp)))\)"
            )
            
            parts = image_pattern.split(content)
            processed_content = []
            
            # Split gives us: [text_before, url1, text_between, url2, text_after, ...]
            for i, part in enumerate(parts):
                if i % 2 == 1:  # This is a captured URL from the regex
                    if part.startswith("data:image"):
                        converted = self._convert_data_url(part)
                        if isinstance(converted, list):
                            processed_content.extend(converted)
                        elif converted:
                            processed_content.append(converted)
                    else:
                        processed_content.append(
                            {"type": "image", "source": {"type": "url", "url": part}}
                        )
                elif part:  # This is text (before, between, or after images)
                    processed_content.append({"type": "text", "text": part})
            
            return processed_content if processed_content else [{"type": "text", "text": content}]

        if isinstance(content, dict):
            content = [content]
        elif not isinstance(content, list):
            return []

        processed = []
        for item in content:
            processed_item = self._process_content_item(item)
            if processed_item:
                if isinstance(processed_item, list):
                    processed.extend(processed_item)
                else:
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

    def _convert_image_url(self, item: dict) -> Union[Optional[dict], List[dict]]:
        """Convert OpenAI image_url format to Anthropic format."""
        url = item.get("image_url", {}).get("url", "")
        if not isinstance(url, str):
            return None

        # Handle base64 data URLs
        if url.startswith("data:image"):
            return self._convert_data_url(url)

        # Handle remote URLs
        return {"type": "image", "source": {"type": "url", "url": url}}

    def _convert_data_url(self, url: str) -> Union[Optional[dict], List[dict]]:
        """Convert data URL to Anthropic image format, optionally uploading to pictshare."""
        try:
            mime_type, base64_data = url.split(",", 1)
            header = mime_type[5:] if mime_type.startswith("data:") else mime_type
            media_type = header.split(";", 1)[0]
        except Exception as e:
            raise ValueError(f"Invalid data URL for image: {e}")

        if media_type not in self.SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"Unsupported media type: {media_type}")

        if len(base64_data) * 3 / 4 > self.MAX_IMAGE_SIZE_BYTES:
            raise ValueError(f"Image size exceeds {self.MAX_IMAGE_SIZE_MB}MB limit")

        # If embedding as URL is enabled, upload to pictshare (synchronously; avoid nested event loop)
        if self.valves.EMBED_IMAGES_AS_URL:
            try:
                image_url = self._upload_to_pictshare_blocking(base64_data)
            except Exception as e:
                logger.warning(f"Pictshare upload raised, falling back to base64: {e}")
                image_url = None
            if image_url:
                return [
                    {"type": "image", "source": {"type": "url", "url": image_url}},
                    {"type": "text", "text": f"Image uploaded: {image_url}"},
                ]
            else:
                logger.warning("Pictshare upload failed, falling back to base64 embedding.")

        # Fallback to base64 if pictshare is disabled or fails
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
            return {
                k: tool[k]
                for k in (
                    "type",
                    "name",
                    "max_uses",
                    "allowed_domains",
                    "blocked_domains",
                    "user_location",
                    "citations",
                    "max_content_tokens",
                    "cache_control",
                )
                if k in tool
            }

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



    def _upload_to_pictshare_blocking(self, b64_data: str) -> Optional[str]:
        pictshare_url = (self.valves.PICTSHARE_URL or "").strip()
        if not pictshare_url:
            logger.warning("PICTSHARE_URL is not configured.")
            return None
        url = f"{pictshare_url}/api/upload.php"
        logger.info(f"Uploading to pictshare: {url}")
        try:
            with httpx.Client(timeout=60) as client:
                try:
                    image_bytes = base64.b64decode(b64_data, validate=True)
                except (ValueError, TypeError):
                    logger.error("Invalid base64 data for pictshare upload.")
                    return None
                files = {"file": ("image.png", image_bytes, "image/png")}
                response = client.post(url, files=files)
                response.raise_for_status()
                res_json = response.json()
                if res_json.get("status") == "ok" and "url" in res_json:
                    logger.info(f"Pictshare upload successful: {res_json['url']}")
                    return res_json["url"]
                else:
                    err = res_json.get("error", "Unknown error")
                    logger.error(f"Pictshare API error: {err}")
                    return None
        except httpx.RequestError as e:
            logger.error(f"HTTP error uploading to pictshare: {e}")
            return None
        except Exception as e:
            logger.error(f"Error during pictshare upload: {e}", exc_info=True)
            return None

    # ========== Main Request Processing ==========
    async def _emit_event(self, payload: dict, **kwargs) -> None:
        """A low-level wrapper to safely emit events to the WebSocket."""
        emitter = kwargs.get("__event_emitter__") or kwargs.get("event_emitter")
        if emitter:
            try:
                await emitter(payload)
            except Exception as e:
                logger.warning(f"Failed to emit event: {e}", exc_info=True)

    async def _emit_status(
        self, description: str, done: bool = False, hidden: bool = False, **kwargs
    ) -> None:
        """
        Handles sending all status messages.
        """
        if not self.valves.show_status:
            return

        payload = {
            "type": "status",
            "data": {"description": description, "done": done, "hidden": hidden},
        }
        await self._emit_event(payload, **kwargs)

    async def _emit_cache_hit_status(self, tokens: int, **kwargs) -> None:
        """Emit a status message for a cache hit."""
        if not self.valves.show_status or tokens <= 0:
            return
        message = f"🎯 {tokens} Tokens read from cache"
        await self._emit_status(message, done=True, hidden=False, **kwargs)


    async def pipe(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __user__: Optional[dict[str, Any]] = None,
    ) -> Union[str, AsyncGenerator[str, None], dict]:
        """Process request through the Anthropic pipeline."""
        kwargs = {
            "__event_emitter__": __event_emitter__,
            "__user__": __user__,
        }
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

            # Build headers (auto-include web fetch beta if required)
            incoming_betas = body.get("betas", [])
            if isinstance(incoming_betas, str):
                betas_list = [incoming_betas]
            elif isinstance(incoming_betas, list):
                betas_list = [str(b) for b in incoming_betas]
            else:
                betas_list = []

            need_fetch_beta = False
            try:
                tools_for_header = payload.get("tools", [])
                if isinstance(tools_for_header, list):
                    for t in tools_for_header:
                        if (
                            isinstance(t, dict)
                            and t.get("type") == self.WEB_FETCH_TOOL_TYPE
                        ):
                            need_fetch_beta = True
                            break
            except Exception:
                need_fetch_beta = False

            if need_fetch_beta and "web-fetch-2025-09-10" not in betas_list:
                betas_list.append("web-fetch-2025-09-10")

            headers = self._build_headers(betas_list if betas_list else None)
            
            # Log outbound payload in verbose mode
            self._log_payload("OUTBOUND", payload)
            
            # Execute request
            if payload["stream"]:
                return self._stream_response(headers, payload, **kwargs)
            else:
                return await self._non_stream_response(headers, payload, **kwargs)

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

                # Only process content for image URLs if the role is 'user'
                if role == "user":
                    processed_content = self._process_content(content)
                else:
                    # For other roles, just ensure it's in the right list format
                    if isinstance(content, str):
                        processed_content = [{"type": "text", "text": content}]
                    elif isinstance(content, dict):
                        processed_content = [content]
                    else:
                        processed_content = content

                normalized.append({"role": role, "content": processed_content})
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

    def _get_or_create_last_user_message(self, conversation: List[dict]) -> dict:
        """Find or create the last user message in the conversation."""
        for message in reversed(conversation):
            if message.get("role") == "user":
                return message
        
        # Create new user message if none exists
        new_user = {"role": "user", "content": []}
        conversation.append(new_user)
        return new_user
    
    def _build_payload(
        self,
        model_name: str,
        messages: List[dict],
        system_blocks: List[dict],
        body: dict,
        is_thinking_mode: bool,
    ) -> dict:
        """Build the API request payload."""
        
        # Extract conversation messages - create shallow copy list but we'll deep copy messages we modify
        conversation = [m for m in messages if m["role"] in ["user", "assistant"]]

        # Venice/Assistant Image Handling:
        # Find all markdown images in assistant messages, move them to the corresponding user turn.
        # This is required because Anthropic does not allow image blocks in assistant messages.
        # We scan ALL assistant messages and check if their images have already been relocated
        image_pattern = re.compile(
            r"\s?!\[[^\]]*\]\((data:image/[^;]+;base64,[^)]+|(?:https?://[^\s)]+\.(?:png|jpg|jpeg|gif|webp)))\)\s?"
        )

        # Process ALL assistant messages, checking each one
        for asst_idx in range(len(conversation)):
            if conversation[asst_idx].get("role") != "assistant":
                continue
                
            original_message = conversation[asst_idx]
            images_in_this_message = []
            
            # Find all markdown images in this assistant message
            if isinstance(original_message.get("content"), list):
                for block in original_message["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        for match in image_pattern.finditer(text):
                            images_in_this_message.append(match.group(1))
            
            if not images_in_this_message:
                continue  # No images in this assistant message
            
            # Check if these images have ALREADY been relocated to a subsequent user message
            already_relocated = set()
            for user_idx in range(asst_idx + 1, len(conversation)):
                if conversation[user_idx].get("role") == "user":
                    user_content = conversation[user_idx].get("content", [])
                    if isinstance(user_content, list):
                        for block in user_content:
                            if isinstance(block, dict) and block.get("type") == "image":
                                source = block.get("source", {})
                                url = source.get("url", "")
                                if url in images_in_this_message:
                                    already_relocated.add(url)
            
            # Only relocate images that haven't been relocated yet
            images_to_relocate = [img for img in images_in_this_message if img not in already_relocated]
            
            if images_to_relocate:
                # Deep copy this message since we're going to modify it
                message = copy.deepcopy(original_message)
                conversation[asst_idx] = message
                new_content = []
                
                # Strip markdown images from text blocks
                if isinstance(message.get("content"), list):
                    for block in message["content"]:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            # Remove all markdown images from this text block
                            clean_text = image_pattern.sub("", text).strip()
                            if clean_text:
                                new_content.append({"type": "text", "text": clean_text})
                        else:
                            new_content.append(block)  # Keep non-text blocks
                
                # If content becomes empty after stripping images, add a placeholder
                if not new_content:
                    new_content.append({"type": "text", "text": "(Image content moved to user message)"})
                message["content"] = new_content

                # Find the next user message after this assistant message
                target_user_idx = -1
                for idx in range(asst_idx + 1, len(conversation)):
                    if conversation[idx].get("role") == "user":
                        target_user_idx = idx
                        break
                
                if target_user_idx == -1:
                    # No user message after this assistant, create one
                    target_user_message = {"role": "user", "content": []}
                    conversation.append(target_user_message)
                else:
                    # Deep copy the target user message
                    target_user_message = copy.deepcopy(conversation[target_user_idx])
                    conversation[target_user_idx] = target_user_message
                    # Ensure content is a list
                    if not isinstance(target_user_message.get("content"), list):
                        if isinstance(target_user_message.get("content"), str):
                            target_user_message["content"] = self._process_content(target_user_message["content"])
                        else:
                            target_user_message["content"] = []
                
                # Relocate images (works for both new and existing user messages)
                for url in images_to_relocate:
                    if url.startswith("data:image"):
                        converted = self._convert_data_url(url)
                        if isinstance(converted, list):
                            target_user_message["content"].extend(converted)
                        elif converted:
                            target_user_message["content"].append(converted)
                    else:
                        target_user_message["content"].append(
                            {"type": "image", "source": {"type": "url", "url": url}}
                        )
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
            last_user = self._get_or_create_last_user_message(conversation)
            if not isinstance(last_user.get("content"), list):
                last_user["content"] = []

            # Append each block discretely, stripping any cache_control
            for blk in relocated_tail:
                if isinstance(blk, dict):
                    blk.pop("cache_control", None)
                    if blk.get("type") == "text" and isinstance(blk.get("text"), str):
                        last_user["content"].append(blk)


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
            "stream": (
                bool(body.get("stream", True)) and self.valves.STREAMING_ENABLED
            ),
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

        # Add tools (merge user-provided with auto web tools)
        user_tools = body.get("tools") if isinstance(body.get("tools"), list) else []
        auto_tools = self._build_web_tools(model_name)
        merged_tools = self._merge_tools(user_tools, auto_tools)
        if merged_tools:
            if self.valves.verbose_logging and auto_tools:
                try:
                    injected = [f"{t.get('type')}:{t.get('name')}" for t in auto_tools]
                    self._log_verbose(f"Injecting web tools: {injected}")
                except Exception:
                    pass
            payload["tools"] = self._convert_tools(merged_tools)

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

    async def _non_stream_response(self, headers: dict, payload: dict, **kwargs) -> str:
        """Handle non-streaming response."""
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=100, limit_per_host=30),
            read_bufsize=2**20  # 1MB read buffer
        ) as session:
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
                self._log_cache_usage(usage, **kwargs)

                # Emit cache hit status
                cache_read = usage.get("cache_read_input_tokens", 0)
                await self._emit_cache_hit_status(cache_read, **kwargs)

                # Extract and format content
                content_blocks = result.get("content", [])
                full = self._extract_and_format_content(content_blocks, **kwargs)
                # Emit final status when response is ready
                await self._emit_status("👍 Response received", done=True, **kwargs)
                return full

    def _extract_and_format_content(
        self, content_blocks: List[dict], **kwargs
    ) -> str:
        """Extract and format thinking and text content from response blocks."""
        thinking_parts = []
        text_parts = []

        if not isinstance(content_blocks, list):
            return ""

        for block in content_blocks:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")
            
            # Log tool invocations and results
            if block_type == "tool_use":
                tool_name = block.get("name", "unknown")
                logger.info(f"🔧 Tool invoked (non-streaming): {tool_name}")
                # Emit status for web_fetch
                if tool_name == "web_fetch":
                    tool_input = block.get("input", {})
                    url = tool_input.get("url", "unknown")
                    self._emit_status(
                        f"🌐 Fetching: {url}",
                        done=False,
                        hidden=False,
                        **kwargs
                    )
            
            # Log web_fetch_tool_result (both success and error)
            elif block_type == "web_fetch_tool_result":
                content = block.get("content", {})
                tool_use_id = block.get("tool_use_id", "unknown")
                
                if isinstance(content, dict):
                    content_type = content.get("type")
                    if content_type == "web_fetch_tool_error":
                        error_code = content.get("error_code", "unknown")
                        logger.warning(
                            f"❌ Web fetch ERROR (non-streaming) for tool_use_id {tool_use_id}: {error_code}"
                        )
                        # Emit detailed error status to user
                        self._emit_status(
                            f"❌ Web fetch failed: {error_code}",
                            done=True,
                            hidden=False,
                            **kwargs
                        )
                    elif content_type == "web_fetch_result":
                        url = content.get("url", "unknown")
                        logger.info(
                            f"✅ Web fetch SUCCESS (non-streaming) for tool_use_id {tool_use_id}: {url}"
                        )
                        # Emit success status to user
                        self._emit_status(
                            f"✅ Fetched: {url}",
                            done=True,
                            hidden=False,
                            **kwargs
                        )
            
            if block_type == "thinking" and "thinking" in block:
                thinking_parts.append(block["thinking"])
            elif block_type == "text" and "text" in block:
                text_parts.append(block["text"])

        full_response = ""
        if thinking_parts:
            full_response += f"<thinking>{''.join(thinking_parts)}</thinking>\n"

        full_response += "".join(text_parts)
        return full_response

    async def _stream_response(self, headers: dict, payload: dict, **kwargs) -> AsyncGenerator[str, None]:
        """Handle streaming response."""
        try:
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=100, limit_per_host=30),
                read_bufsize=2**20  # 1MB read buffer
            ) as session:
                async with session.post(
                    self.MESSAGES_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT),
                    max_line_size=10*1024*1024,  # 10MB max line size for large web_fetch responses
                    max_field_size=10*1024*1024  # 10MB max field size
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        yield f"Error: HTTP {response.status}: {error_text}"
                        return

                    async for chunk in self._process_stream(response, **kwargs):
                        yield chunk

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"Stream Error: {str(e)}"

    async def _process_stream(
        self, response, **kwargs
    ) -> AsyncGenerator[str, None]:
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
                        self._log_cache_usage(usage, is_stream=True, **kwargs)

                        # Emit cache hit status
                        cache_read = usage.get("cache_read_input_tokens", 0)
                        await self._emit_cache_hit_status(cache_read, **kwargs)

                    elif event_type == "content_block_start":
                        block_type = data.get("content_block", {}).get("type")
                        if block_type == "thinking":
                            in_thinking_block = True
                            yield "<thinking>"
                        elif block_type == "tool_use":
                            tool_name = data.get("content_block", {}).get("name", "unknown")
                            logger.info(f"🔧 Tool invoked: {tool_name}")
                            # Emit status for web tools
                            if tool_name == "web_fetch":
                                tool_input = data.get("content_block", {}).get("input", {})
                                url = tool_input.get("url", "unknown")
                                await self._emit_status(
                                    f"🌐 Fetching: {url}",
                                    done=False,
                                    hidden=False,
                                    **kwargs
                                )
                            elif tool_name == "web_search":
                                tool_input = data.get("content_block", {}).get("input", {})
                                query = tool_input.get("query", "unknown")
                                await self._emit_status(
                                    f"🔍 Searching: {query}",
                                    done=False,
                                    hidden=False,
                                    **kwargs
                                )
                        elif block_type == "web_search_tool_result":
                            content = data.get("content_block", {}).get("content", [])
                            if content and isinstance(content, list):
                                result_count = sum(1 for r in content if isinstance(r, dict) and r.get("type") == "web_search_result")
                                await self._emit_status(
                                    f"✅ Found {result_count} search results",
                                    done=True,
                                    hidden=False,
                                    **kwargs
                                )
                        elif block_type == "web_fetch_tool_result":
                            # web_fetch_tool_result is a content_block type, not a separate event
                            content_block = data.get("content_block", {})
                            content = content_block.get("content", {})
                            tool_use_id = content_block.get("tool_use_id", "unknown")
                            
                            if isinstance(content, dict):
                                content_type = content.get("type")
                                if content_type == "web_fetch_tool_error":
                                    error_code = content.get("error_code", "unknown")
                                    logger.warning(
                                        f"❌ Web fetch ERROR for tool_use_id {tool_use_id}: {error_code}"
                                    )
                                    # Emit detailed error status to user
                                    await self._emit_status(
                                        f"❌ Web fetch failed: {error_code}",
                                        done=True,
                                        hidden=False,
                                        **kwargs
                                    )
                                elif content_type == "web_fetch_result":
                                    url = content.get("url", "unknown")
                                    logger.info(
                                        f"✅ Web fetch SUCCESS for tool_use_id {tool_use_id}: {url}"
                                    )
                                    # Emit success status to user
                                    await self._emit_status(
                                        f"✅ Fetched: {url}",
                                        done=True,
                                        hidden=False,
                                        **kwargs
                                    )

                    elif event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        delta_type = delta.get("type")

                        if delta_type == "text_delta":
                            yield delta.get("text", "")
                        elif delta_type == "thinking_delta":
                            yield delta.get("thinking", "")

                    elif event_type == "content_block_stop":
                        if in_thinking_block:
                            yield "</thinking>"
                            in_thinking_block = False
                        
                        # Log tool completion
                        block = data.get("content_block", {})
                        if block.get("type") == "tool_use":
                            tool_name = block.get("name", "unknown")
                            logger.info(f"🔧 Tool completed: {tool_name}")

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

                    elif event_type == "message_stop":
                        # Emit final status at end of streaming message
                        await self._emit_status("💦 Stream received", done=True, **kwargs)

                    elif event_type == "error":
                        error = data.get("error", {})
                        logger.error(f"Stream error from API: {error}")
                        yield f"Error: {error.get('message', 'Unknown error from API')}"

                except json.JSONDecodeError:
                    self._log_verbose(f"Failed to decode stream data: {data_str!r}")

    def _build_web_tools(self, model_name: str) -> List[dict]:
        """Assemble server web tools from valves for supported models."""
        tools: List[dict] = []

        # Shared config
        max_uses_val = max(0, getattr(self.valves, "WEB_MAX_USES", 0))
        user_location = getattr(self.valves, "WEB_USER_LOCATION", None)

        # Web Search
        if getattr(
            self.valves, "ENABLE_WEB_SEARCH", False
        ) and self._supports_web_search_capability(model_name):
            search_tool: Dict[str, Any] = {
                "type": self.WEB_SEARCH_TOOL_TYPE,
                "name": "web_search",
            }
            if max_uses_val > 0:
                search_tool["max_uses"] = max_uses_val
            if isinstance(user_location, dict) and user_location:
                search_tool["user_location"] = user_location
            tools.append(search_tool)

        # Web Fetch
        if getattr(
            self.valves, "ENABLE_WEB_FETCH", False
        ) and self._supports_web_fetch_capability(model_name):
            fetch_tool: Dict[str, Any] = {
                "type": self.WEB_FETCH_TOOL_TYPE,
                "name": "web_fetch",
            }
            if max_uses_val > 0:
                fetch_tool["max_uses"] = max_uses_val

            # Citations and content token cap
            citations_enabled = bool(
                getattr(self.valves, "WEB_FETCH_CITATIONS_ENABLED", True)
            )
            fetch_tool["citations"] = {"enabled": citations_enabled}

            max_content_tokens = max(
                0, getattr(self.valves, "WEB_FETCH_MAX_CONTENT_TOKENS", 0)
            )
            if max_content_tokens > 0:
                fetch_tool["max_content_tokens"] = max_content_tokens

            # Add 1-hour cache control to cache all web tools
            fetch_tool["cache_control"] = {
                "type": self.CACHE_TYPE_EPHEMERAL,
                "ttl": self.CACHE_TTL_1H,
            }

            tools.append(fetch_tool)

        return tools

    def _merge_tools(
        self, user_tools: List[dict], auto_tools: List[dict]
    ) -> List[dict]:
        """
        Merge user-provided and auto-generated tools.
        De-duplicate by (type, name), preferring user tools.
        """
        merged: List[dict] = []
        seen = set()

        # Add user tools first
        for t in user_tools or []:
            if isinstance(t, dict):
                key = (t.get("type"), t.get("name"))
                merged.append(t)
                seen.add(key)

        # Add auto tools not provided by user
        for t in auto_tools or []:
            if isinstance(t, dict):
                key = (t.get("type"), t.get("name"))
                if key not in seen:
                    merged.append(t)
                    seen.add(key)

        tool_list = [f"{t.get('type')}:{t.get('name')}" for t in merged if isinstance(t, dict)]
        logger.info(f"Merged tools: {tool_list}")

        return merged
