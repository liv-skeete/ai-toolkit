"""
title: Dynamic Anthropic
description: Supports Anthropic API with dynamic model updates
author: Cody
version: 1.2.0
date: 2025-09-07
changes:
- v1.2.0: Implemented intelligent caching system that automatically reorganizes system messages
  to minimize cache invalidation and stay within Anthropic's 4 breakpoint limit.
  Removed CACHE_1H_PREFIXES and CACHE_5M_PREFIXES valves - now uses hardcoded smart logic.
- v1.1.1: Trimmed debug-only options: removed raw_stream and return_blocks paths
- Removed cache_control handling from content and tool schemas (unused by Open WebUI)
- Kept thinking budget support for '-thinking' variants, with existing valves
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Union, Dict, Optional, AsyncGenerator
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message
import aiohttp


# Set up a dedicated logger for this module
logger = logging.getLogger("anthropic")
logger.propagate = False
logger.setLevel(logging.INFO)

# Configure handler if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class Pipe:
    API_VERSION = "2023-06-01"
    MODEL_URL = "https://api.anthropic.com/v1/messages"
    MODELS_API_URL = "https://api.anthropic.com/v1/models"
    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    MAX_IMAGE_SIZE = 5 * 1024 * 1024
    REQUEST_TIMEOUT = 300

    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(
            default=os.getenv("ANTHROPIC_API_KEY", ""),
            description="Your Anthropic API key",
        )
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
        THINKING_BUDGET_RATIO: float = Field(
            default=float(os.getenv("ANTHROPIC_THINKING_BUDGET_RATIO", "0.3")),
            gt=0.0,
            lt=1.0,
            description="Fraction of max_tokens allocated to thinking budget (0-1, must be <1)",
        )

        # User message caching (still configurable)
        CACHE_USER_MESSAGE: bool = Field(default=False)
        CACHE_USER_MESSAGE_TTL: str = Field(default="5m")

    def __init__(self):
        self.logger = logger
        self.type = "manifold"
        self.id = "anthropic"
        self.valves = self.Valves()
        self.request_id = None

        self._cached_models: Optional[List[dict]] = None
        self._models_cache_time: Optional[datetime] = None

        self.logger.info("🚀 Dynamic Anthropic Pipeline initialized")
        self.logger.info(f"⏰ Cache duration: {self.valves.MODELS_CACHE_HOURS} hours")


    def get_anthropic_models(self) -> List[dict]:
        """Get models with caching, relying solely on the API."""
        self.logger.info("🔧 get_anthropic_models() called")

        # Check cache validity
        cache_duration_hours = self.valves.MODELS_CACHE_HOURS
        if cache_duration_hours > 0 and self._cached_models and self._models_cache_time:
            cache_age = datetime.now() - self._models_cache_time
            if cache_age < timedelta(hours=cache_duration_hours):
                self.logger.info(
                    f"💾 Using cached models (age: {cache_age.total_seconds():.0f}s)"
                )
                return self._cached_models
            else:
                self.logger.info("⏰ Cache expired, fetching fresh models from API.")

        # Try fetching from API
        self.logger.info("🌐 Attempting to fetch models from API...")
        api_models = self._fetch_models_from_api()
        if api_models:
            transformed_models = self._transform_api_models(api_models)
            if transformed_models:
                self._cached_models = transformed_models
                self._models_cache_time = datetime.now()
                self.logger.info(
                    f"✅ Successfully refreshed {len(transformed_models)} models from API"
                )
                return transformed_models

        # If API fails, return empty list and clear cache
        self.logger.warning(
            "⚠️ API unavailable or returned no models. Returning empty list."
        )
        self._cached_models = None
        self._models_cache_time = None
        return []

    def _fetch_models_from_api(self) -> List[dict]:
        """Fetch available models from Anthropic API."""
        api_key = self.valves.ANTHROPIC_API_KEY
        if not api_key or not api_key.strip():
            self.logger.warning("🔐 No API key available for fetching models.")
            return []

        self.logger.info(f"📡 Fetching models from {self.MODELS_API_URL}...")
        headers = {
            "x-api-key": api_key,
            "anthropic-version": self.API_VERSION,
        }

        try:
            response = requests.get(self.MODELS_API_URL, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                self.logger.info(
                    f"✅ Successfully fetched {len(models)} models from API."
                )
                return models
            else:
                self.logger.warning(
                    f"❌ API returned {response.status_code}: {response.text}"
                )
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"❌ Failed to fetch models from API: {e}")

        return []

    def _transform_api_models(self, api_models: List[dict]) -> List[dict]:
        """Transform Anthropic API models to Open WebUI format."""
        models = []
        for model in api_models:
            if model.get("type") == "model":
                model_id = model.get("id", "")

                # Determine vision support using input_modalities if present; fallback to tags
                supports_vision = False
                input_modalities = model.get("input_modalities")
                if isinstance(input_modalities, list) and any(
                    m in ("image", "vision") for m in input_modalities
                ):
                    supports_vision = True
                else:
                    supports_vision = "vision" in model.get("tags", [])

                context_len = model.get("context_length", 200000)

                # Base model
                models.append(
                    {
                        "id": model_id,
                        "name": model_id,
                        "context_length": context_len,
                        "supports_vision": supports_vision,
                    }
                )

                # Add thinking variant for models that support it (heuristic)
                mid = model_id
                thinking_capable = (
                    ("sonnet-4-" in mid)
                    or ("opus-4-1-" in mid)
                    or ("opus-4-" in mid)
                    or ("3-7-sonnet-" in mid)
                )
                if thinking_capable:
                    models.append(
                        {
                            "id": f"{model_id}-thinking",
                            "name": f"{model_id}-thinking",
                            "context_length": context_len,
                            "supports_vision": supports_vision,
                        }
                    )

        self.logger.info(
            f"✅ Transformed {len(api_models)} API models into {len(models)} total variants."
        )
        return models

    def pipes(self) -> List[dict]:
        """Return available models."""
        self.logger.info("🔌 pipes() method called - retrieving model list")
        return self.get_anthropic_models()

    def _process_content(self, content: Union[str, Dict, List[dict]]) -> List[dict]:
        # Normalize to list of content-block dicts, supporting strings and dicts
        if content is None:
            return []
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, dict):
            items = [content]
        else:
            items = content

        processed_content: List[dict] = []
        for item in items:
            # Convert plain strings to text blocks
            if isinstance(item, str):
                processed_content.append({"type": "text", "text": item})
                continue
            if not isinstance(item, dict):
                # Skip unsupported types
                continue

            item_type = item.get("type")
            if item_type in [
                "text",
                "image",
                "tool_use",
                "tool_result",
                "thinking",
                "redacted_thinking",
            ]:
                processed_content.append(item)
                continue

            if item_type == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if not isinstance(url, str):
                    continue
                # Data URL (base64)
                if url.startswith("data:image"):
                    try:
                        mime_type, base64_data = url.split(",", 1)
                        header = (
                            mime_type[5:]
                            if mime_type.startswith("data:")
                            else mime_type
                        )
                        media_type = header.split(";")[0]
                    except Exception as e:
                        raise ValueError(f"Invalid data URL for image: {e}")

                    if media_type not in self.SUPPORTED_IMAGE_TYPES:
                        raise ValueError(f"Unsupported media type: {media_type}")
                    if len(base64_data) * 3 / 4 > self.MAX_IMAGE_SIZE:
                        raise ValueError(
                            f"Image size exceeds {self.MAX_IMAGE_SIZE/(1024*1024)}MB limit."
                        )

                    image_block = {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_data,
                        },
                    }
                    processed_content.append(image_block)
                else:
                    # Remote URL passthrough as Anthropic image block with URL source
                    image_block = {
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    }
                    processed_content.append(image_block)
                continue

            # Unknown type: attempt to coerce if 'text' field present, otherwise skip
            if "text" in item and item_type is None:
                block = {"type": "text", "text": item["text"]}
                processed_content.append(block)
                continue

            # If we reach here, skip unrecognized content item
            continue

        return processed_content

    def _convert_openai_tools_to_anthropic(self, tools: List[dict]) -> List[dict]:
        """
        Normalize tools into Anthropic's tools schema.
        - Pass through Anthropic-shaped tools (name, input_schema, optional description)
        - Convert OpenAI function tools into Anthropic shape
        - Pass through server-defined tools (e.g., web_search_20250305) with name/max_uses
        """
        anthropic_tools = []
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue

            # Pass-through for Anthropic-shaped tool definitions
            if "input_schema" in tool and "name" in tool:
                t = {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "input_schema": tool.get("input_schema"),
                }
                anthropic_tools.append(t)
                continue

            # Pass-through for server-defined tools (e.g., web_search_20250305)
            if tool.get("type") and tool.get("type") != "function" and "name" in tool and "input_schema" not in tool:
                t = {k: tool[k] for k in ("type", "name", "max_uses") if k in tool}
                anthropic_tools.append(t)
                continue

            # Convert from OpenAI function tools
            if tool.get("type") == "function":
                function = tool.get("function", {}) or {}
                t = {
                    "name": function.get("name"),
                    "description": function.get("description"),
                    "input_schema": function.get("parameters", {}) or {},
                }
                anthropic_tools.append(t)

        return anthropic_tools
    def _apply_intelligent_caching(self, system_blocks: List[dict]) -> List[dict]:
        """
        Apply intelligent caching that:
        1. Groups blocks to minimize breakpoints
        2. Differentiates between session and static memories
        3. Places volatile content at the end to minimize invalidation
        4. Respects the 4 breakpoint limit
        
        Hardcoded logic (no valves):
        - System Prompt:, User Information, non-[Session] memories: 1h cache
        - [Session] memories: 5m cache
        - Time Context: no cache (always last)
        """
        static_blocks = []  # Sol-core, User Info, non-[Session] memories
        session_memory_blocks = []  # [Session] memories
        volatile_blocks = []  # Time Context and other non-cacheable content
        
        for block in system_blocks:
            if block.get("type") != "text":
                volatile_blocks.append(block)
                continue
                
            txt = block.get("text", "")
            
            # Categorize blocks based on content
            # IMPORTANT: Check Sol-core and User Info FIRST, before [Session]
            # because System Prompt: contains "[Session]" in its instructions
            if "Time Context:" in txt:
                # Time context should never be cached
                volatile_blocks.append(block)
            elif txt.startswith("System Prompt:") or txt.startswith("User Information:"):
                # Core static content gets 1h cache
                # Use startswith to avoid false matches
                static_blocks.append(block)
            elif "User Memory:" in txt and "[Session]" in txt:
                # Session memories get 5m cache
                session_memory_blocks.append(block)
            elif "User Memory:" in txt:
                # Non-session memories (like [Reminder], [Lifestyle], [Preference]) get 1h cache
                static_blocks.append(block)
            else:
                # Unknown content - no cache
                volatile_blocks.append(block)
        
        # Rebuild system content in optimal order
        reorganized_system = []
        
        # 1. Add static content first (most cacheable)
        if static_blocks:
            reorganized_system.extend(static_blocks)
            # Always add cache control to the LAST static block
            reorganized_system[-1]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
        
        # 2. Add session memory blocks (medium volatility)
        if session_memory_blocks:
            reorganized_system.extend(session_memory_blocks)
            # Always add cache control to the LAST session block
            reorganized_system[-1]["cache_control"] = {"type": "ephemeral", "ttl": "5m"}
        
        # 3. Add volatile blocks last (no cache)
        reorganized_system.extend(volatile_blocks)
        
        # Log the caching strategy for debugging
        self.logger.info(f"Cache strategy: {len(static_blocks)} static (1h), {len(session_memory_blocks)} session (5m), {len(volatile_blocks)} volatile (no cache)")
        
        return reorganized_system


    async def pipe(self, body: Dict) -> Union[str, AsyncGenerator[str, None]]:
        # Log the inbound payload for debugging
        try:
            body_str = json.dumps(body, indent=2)
            self.logger.info(f"INBOUND PAYLOAD:\n---\n{body_str}\n---")
        except Exception as e:
            self.logger.warning(f"Failed to log inbound payload: {e}")

        if not self.valves.ANTHROPIC_API_KEY:
            return {"content": "Error: ANTHROPIC_API_KEY is required", "format": "text"}

        try:
            model_name = body["model"].split("/")[-1].split(".")[-1]
            is_thinking_mode = "-thinking" in body["model"]
            if is_thinking_mode:
                model_name = model_name.replace("-thinking", "")

            # Normalize messages to ensure each is a dict with role and Anthropic-formatted content
            normalized_messages: List[dict] = []
            for m in body["messages"]:
                if isinstance(m, dict):
                    role = m.get("role", "user")
                    content_value = m.get("content", m.get("text", ""))
                    normalized_messages.append(
                        {"role": role, "content": self._process_content(content_value)}
                    )
                elif isinstance(m, str):
                    normalized_messages.append(
                        {"role": "user", "content": self._process_content(m)}
                    )
                else:
                    # Skip unsupported message item types
                    continue

            # Collect all system blocks for intelligent caching
            system_blocks = []
            for message in normalized_messages:
                if message["role"] == "system":
                    content = message.get("content")
                    if isinstance(content, list):
                        system_blocks.extend(content)
            # User Message Caching
            if self.valves.CACHE_USER_MESSAGE:
                last_user_message = next(
                    (
                        m
                        for m in reversed(normalized_messages)
                        if m["role"] == "user"
                    ),
                    None,
                )
                if last_user_message and isinstance(last_user_message.get("content"), list):
                    # Find the last text block in the content and add cache_control to it
                    for block in reversed(last_user_message["content"]):
                        if block.get("type") == "text":
                            block["cache_control"] = {
                                "type": "ephemeral",
                                "ttl": self.valves.CACHE_USER_MESSAGE_TTL,
                            }
                            break

            # Assemble Payload
            conversation_messages = [
                message
                for message in normalized_messages
                if message["role"] in ["user", "assistant"]
            ]

            payload = {
                "model": model_name,
                "messages": conversation_messages,
                "max_tokens": int(
                    body.get("max_tokens", self.valves.DEFAULT_MAX_TOKENS)
                ),
                "stream": body.get("stream", True),
            }

            # Apply intelligent caching and use reorganized system blocks
            if system_blocks:
                payload["system"] = self._apply_intelligent_caching(system_blocks)

            # Sampling params
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
            # Tools and tool choice
            has_tools = False
            if "tools" in body:
                payload["tools"] = self._convert_openai_tools_to_anthropic(
                    body["tools"]
                )
                has_tools = bool(payload["tools"])
            if "tool_choice" in body:
                payload["tool_choice"] = body["tool_choice"]
                # Normalize to documented schema: {"type": "..."} when provided as a bare string
                if isinstance(payload["tool_choice"], str):
                    payload["tool_choice"] = {"type": payload["tool_choice"]}
            # Beta headers (e.g., interleaved thinking)
            betas = body.get("betas")
            interleaved_enabled = False
            if isinstance(betas, list):
                interleaved_enabled = "interleaved-thinking-2025-05-14" in betas
            elif isinstance(betas, str):
                interleaved_enabled = "interleaved-thinking-2025-05-14" in betas
            # Configure extended thinking
            if is_thinking_mode:
                max_tokens = int(body.get("max_tokens", self.valves.DEFAULT_MAX_TOKENS))
                ratio = self.valves.THINKING_BUDGET_RATIO
                # Clamp ratio into (0,1)
                ratio = max(0.0, min(ratio, 0.99))
                explicit_budget = body.get("thinking_budget_tokens")
                if explicit_budget is not None:
                    budget = int(explicit_budget)
                    if max_tokens > 1024 and budget < 1024 and not interleaved_enabled:
                        self.logger.info(
                            "Thinking budget < 1024 tokens; consider increasing for complex tasks."
                        )
                else:
                    budget = max(1, int(max_tokens * ratio))
                    if not interleaved_enabled and max_tokens > 1024:
                        budget = max(1024, budget)
                if max_tokens <= 1:
                    self.logger.warning(
                        "Thinking requested but max_tokens <= 1; disabling thinking for this request."
                    )
                else:
                    if not (interleaved_enabled and has_tools):
                        if budget >= max_tokens:
                            budget = max(1, max_tokens - 1)
                    payload["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": int(budget),
                    }
            # Enforce tool_choice constraints when thinking is enabled
            if is_thinking_mode and has_tools and "tool_choice" in payload:
                tc = payload.get("tool_choice")
                tc_type = tc.get("type") if isinstance(tc, dict) else None
                if tc_type not in {"auto", "none"}:
                    return "Error: tool_choice must be 'auto' or 'none' when thinking is enabled"
            # Streaming required for very large outputs
            if not payload["stream"] and payload["max_tokens"] > 21333:
                self.logger.warning(
                    "max_tokens > 21333 with stream disabled; forcing stream=True per Anthropic requirements."
                )
                payload["stream"] = True
            headers = {
                "x-api-key": self.valves.ANTHROPIC_API_KEY,
                "anthropic-version": self.API_VERSION,
                "content-type": "application/json",
            }
            if betas:
                headers["anthropic-beta"] = (
                    ",".join(betas) if isinstance(betas, list) else str(betas)
                )

            # Log the outbound payload for debugging
            try:
                log_payload = {k: v for k, v in payload.items() if k != "messages"}
                log_payload["messages"] = [
                    {
                        "role": m.get("role"),
                        "content": m.get("content"),
                    }
                    for m in payload.get("messages", [])
                ]
                payload_str = json.dumps(log_payload, indent=2)
                self.logger.info(
                    f"OUTBOUND PAYLOAD:\n---\n{payload_str}\n---"
                )
            except Exception as e:
                self.logger.warning(f"Failed to log outbound payload: {e}")

            # Streaming
            if payload["stream"]:
                return self._stream_response(
                    self.MODEL_URL, headers, payload
                )
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.MODEL_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT),
                ) as response:
                    if response.status != 200:
                        return f"Error: HTTP {response.status}: {await response.text()}"
                    result = await response.json()

                    # Log prompt caching usage
                    usage = result.get("usage", {})
                    try:
                        self.logger.info(
                            f"CACHE USAGE: read={usage.get('cache_read_input_tokens', 0)}, "
                            f"created={usage.get('cache_creation_input_tokens', 0)}, "
                            f"input={usage.get('input_tokens', 0)}"
                        )
                        if (
                            usage.get("cache_read_input_tokens", 0) > 0
                            or usage.get("cache_creation_input_tokens", 0) > 0
                        ):
                            self.logger.info(
                                f"CACHE HIT: read={usage.get('cache_read_input_tokens', 0)}, "
                                f"created={usage.get('cache_creation_input_tokens', 0)}, "
                                f"input={usage.get('input_tokens', 0)}"
                            )
                    except Exception:
                        pass


                    # Robust content parsing
                    content = result.get("content", "")
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, list):
                        full_text = ""
                        for block in content:
                            if isinstance(block, dict) and "text" in block:
                                full_text += block["text"]
                        return full_text

                    self.logger.warning(
                        "Could not extract text from non-streaming response."
                    )
                    return ""

        except Exception as e:
            self.logger.error(f"Pipe error: {e}")
            return f"Error: {str(e)}"

    async def _stream_response(
        self, url: str, headers: dict, payload: dict
    ) -> AsyncGenerator[str, None]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        yield f"Error: HTTP {response.status}: {error_text}"
                        return

                    buffer = ""
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
                                    try:
                                        self.logger.info(
                                            f"CACHE USAGE (stream): read={usage.get('cache_read_input_tokens', 0)}, "
                                            f"created={usage.get('cache_creation_input_tokens', 0)}, "
                                            f"input={usage.get('input_tokens', 0)}"
                                        )
                                        if (
                                            usage.get("cache_read_input_tokens", 0) > 0
                                            or usage.get("cache_creation_input_tokens", 0) > 0
                                        ):
                                            self.logger.info(
                                                f"CACHE HIT (stream): read={usage.get('cache_read_input_tokens', 0)}, "
                                                f"created={usage.get('cache_creation_input_tokens', 0)}, "
                                                f"input={usage.get('input_tokens', 0)}"
                                            )
                                    except Exception:
                                        pass
                                elif event_type == "content_block_delta":
                                    delta = data.get("delta", {})
                                    delta_type = delta.get("type")
                                    if delta_type == "text_delta":
                                        yield delta.get("text", "")
                                    elif delta_type == "thinking_delta":
                                        yield delta.get("thinking", "")
                                    elif delta_type == "input_json_delta":
                                        # Debug-only for fine-grained tool streaming; do not emit in non-raw mode
                                        pj = delta.get("partial_json", "")
                                        self.logger.debug(f"input_json_delta len={len(pj)}")
                                elif event_type == "error":
                                    error_details = data.get("error", {})
                                    self.logger.error(f"Stream error from API: {error_details}")
                                    yield f"Error: {error_details.get('message', 'Unknown error from API')}"
                                # Other event types are ignored in non-raw mode (tool_use, signature_delta, etc.)
                            except json.JSONDecodeError:
                                self.logger.warning(f"Failed to decode stream data: {data_str!r}")
                                continue
        except Exception as e:
            self.logger.error(f"Stream error: {e}", exc_info=True)
            yield f"Stream Error: {str(e)}"
