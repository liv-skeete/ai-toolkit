"""
title: Context Manager
description: 1. Truncate chat context with token limit and max turns; preserves critical MRS system memory. 2. Log chat turn data with 'log_type'.
author: Cody
version: 1.0
date: 2025-08-19
changes:
"""

import tiktoken
from pydantic import BaseModel, Field
from typing import Optional, Callable, Any, Awaitable
import time
import logging
import json

# Standalone logger configuration aligned with Utilities/Date_Time.py
logger = logging.getLogger("context_manager")
logger.propagate = False
logger.setLevel(logging.INFO)

# Configure handler once
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0, description="Priority level")
        show_status: bool = Field(
            default=True, description="Emit status events to the UI"
        )
        max_turns: int = Field(
            default=20,
            description="Number of conversation turns to retain. Set '0' for unlimited",
        )
        token_limit: int = Field(
            default=16000,
            description="Number of token limit to retain. Set '0' for unlimited",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.limit_exceeded = False
        self.encoding = tiktoken.get_encoding("o200k_base")
        self.input_tokens = 0
        self.output_tokens = 0
        self.user = None
        self.model_base = None
        self.model_name = None
        self.start_time = None
        self.elapsed_time = None
        self.input_message_count = None

        # Internal tracking for trimming diagnostics
        self._turns_trim_info = None
        self._tokens_trim_info = None

    def _is_pinned_system(self, msg: dict) -> bool:
        """
        Return True if this is the MRS-injected system memory message that should be preserved.
        """
        try:
            if msg.get("role") != "system":
                return False
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.startswith("User Information (sorted by relevance):")
            return False
        except Exception:
            return False

    def _extract_user_identifier(self, __user__: Optional[dict]) -> Optional[str]:
        """
        Safely extract a user identifier for logging: prefer email, fall back to username or id.
        Works with dict-like objects or attribute-bearing objects.
        """
        if not __user__:
            return None
        try:
            if isinstance(__user__, dict):
                return __user__.get("email") or __user__.get("username") or __user__.get("id")
            return getattr(__user__, "email", None) or getattr(__user__, "username", None) or getattr(__user__, "id", None)
        except Exception:
            return None

    def _infer_model_fields(self, body: dict, __model__: Optional[dict]) -> tuple[Optional[str], Optional[str]]:
        """
        Infer model_base (id) and model_name with preference for SMR breadcrumb.
        Returns (model_base, model_name).
        """
        model_id: Optional[str] = None
        model_name: Optional[str] = None
        try:
            meta = body.get("metadata") or {}
            if isinstance(meta, dict):
                router = meta.get("router") or {}
                if isinstance(router, dict):
                    # Prefer SMR breadcrumb
                    model_id = router.get("model_id") or model_id
                    model_name = router.get("model") or model_name  # route key as name if available
            # Fallbacks
            if not model_id:
                if isinstance(body, dict):
                    model_id = body.get("model") or model_id
                if not model_id and isinstance(__model__, dict):
                    model_id = __model__.get("id")
            if not model_name:
                if isinstance(__model__, dict):
                    model_name = __model__.get("name")
                if not model_name and model_id:
                    model_name = str(model_id)
        except Exception:
            if isinstance(body, dict):
                model_id = body.get("model", None)
            if isinstance(__model__, dict):
                model_id = model_id or __model__.get("id")
                model_name = __model__.get("name") or model_name
            if not model_name and model_id:
                model_name = str(model_id)
        return model_id, model_name

    def log_chat_turn(self):
        """
        Log data for a single chat turn
        """
        if all(
            [
                self.user,
                self.model_base,
                self.model_name,
                self.input_tokens is not None,
                self.output_tokens is not None,
                self.elapsed_time is not None,
                self.input_message_count is not None,
            ]
        ):
            log_data = {
                "log_type": "chat_turn",
                "user": self.user,
                "model_base": self.model_base,
                "model_name": self.model_name,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "elapsed_seconds": round(self.elapsed_time, 0),
                "input_message_count": self.input_message_count,
            }
            # print(json.dumps(log_data))
            logger.info(json.dumps(log_data))

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __model__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        """Truncate chat context with token and turn limits; preserves critical MRS system memory."""
        messages = body["messages"]
        chat_messages = messages[:]
        self.limit_exceeded = False
        # Reset trim diagnostics
        self._turns_trim_info = None
        self._tokens_trim_info = None

        # Pre-trim diagnostics
        user_ident = self._extract_user_identifier(__user__)
        try:
            pre_count = len(chat_messages)
            pre_tokens = sum(self.count_text_tokens(m) for m in chat_messages)
        except Exception:
            pre_count = len(chat_messages)
            pre_tokens = 0
        logger.info(
            f"Context trimming started for user {user_ident or 'unknown'}: "
            f"messages={pre_count}, tokens={pre_tokens}, "
            f"max_turns={self.valves.max_turns}, token_limit={self.valves.token_limit}"
        )

        # Apply trims
        chat_messages = self.truncate_turns(chat_messages)
        chat_messages = self.truncate_tokens(chat_messages)

        # Post-trim diagnostics
        try:
            post_count = len(chat_messages)
            post_tokens = sum(self.count_text_tokens(m) for m in chat_messages)
        except Exception:
            post_count = len(chat_messages)
            post_tokens = 0

        # Emit detailed trim logs if applied
        if isinstance(self._turns_trim_info, dict):
            info = self._turns_trim_info
            logger.info(
                "Turn trim: "
                f"current_turns={info.get('current_turns')}, "
                f"max_turns={info.get('max_turns')}, "
                f"removed_messages={info.get('removed')}, "
                f"removed_turns={info.get('removed_turns')}, "
                f"pinned_preserved={info.get('pinned_preserved')}"
            )
        if isinstance(self._tokens_trim_info, dict):
            info = self._tokens_trim_info
            if (info.get("dropped") or 0) > 0:
                logger.info(
                    "Token trim: "
                    f"budget={info.get('budget')}, "
                    f"pinned_tokens={info.get('pinned_tokens')}, "
                    f"kept={info.get('kept')}, "
                    f"dropped={info.get('dropped')}"
                )


        logger.info(
            f"Context trimming complete: messages={post_count}, tokens={post_tokens}, "
            f"limit_exceeded={self.limit_exceeded}"
        )
        await self.show_exceeded_status(__event_emitter__, len(chat_messages))
        self.init_log_data(user_ident, chat_messages)
        body["messages"] = chat_messages
        return body

    def truncate_turns(self, messages: list) -> list:
        """
        Trim by conversation turns while preserving pinned system messages (e.g., MRS memory block).
        Only user/assistant roles count towards turn calculation; system messages are excluded.
        """
        result = messages
        self._turns_trim_info = None
        if self.valves.max_turns > 0 and messages:
            # Count only user/assistant messages
            ua_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]
            current_turns = max(0, (len(ua_msgs) - 1) // 2)
            if current_turns > self.valves.max_turns:
                sent_msg_count = self.valves.max_turns * 2 + 1
                cut_start = len(messages) - sent_msg_count
                if cut_start < 0:
                    cut_start = 0
                # Preserve pinned system messages that appear before the cut
                pinned_prefix = [m for m in messages[:cut_start] if self._is_pinned_system(m)]
                tail = messages[cut_start:]
                # Messages removed are those before cut that were not pinned and not included in tail
                removed = cut_start - len(pinned_prefix)
                # Avoid duplicates if any pinned also exist in tail
                result = pinned_prefix + tail
                self.limit_exceeded = True
                self._turns_trim_info = {
                    "current_turns": current_turns,
                    "max_turns": self.valves.max_turns,
                    "removed": max(0, removed),
                    "removed_turns": max(0, current_turns - self.valves.max_turns),
                    "pinned_preserved": len(pinned_prefix),
                }
        return result

    def truncate_tokens(self, messages: list) -> list:
        """
        Trim by token budget while always preserving:
        - Pinned system memory from MRS
        - The latest user message (never dropped)
        """
        filter_messages = messages
        self._tokens_trim_info = None
        if self.valves.token_limit > 0 and messages:
            budget = self.valves.token_limit
            # Collect pinned system messages and reserve their token budget
            pinned = [m for m in messages if self._is_pinned_system(m)]
            pinned_ids = set(id(m) for m in pinned)
            pinned_tokens = sum(self.count_text_tokens(m) for m in pinned)

            non_pinned_result = []
            current_toks = pinned_tokens

            for msg in reversed(messages):
                if id(msg) in pinned_ids:
                    continue  # already reserved
                toks = self.count_text_tokens(msg)
                role = msg.get("role", "")
                # Latest user message should not be dropped even if it exceeds the limit
                if (current_toks + toks > budget) and (role != "user"):
                    self.limit_exceeded = True
                    break
                non_pinned_result.insert(0, msg)
                current_toks += toks

            # Final ordering: pinned (front) + retained non-pinned in original order
            filter_messages = pinned + non_pinned_result

            dropped = max(0, len(messages) - len(filter_messages))
            self._tokens_trim_info = {
                "budget": budget,
                "pinned_tokens": pinned_tokens,
                "kept": len(filter_messages),
                "dropped": dropped,
            }
        return filter_messages

    def init_log_data(self, user_email: str, messages: list):
        self.user = user_email
        self.input_tokens = 0
        self.start_time = time.time()
        for msg in messages:
            self.input_tokens += self.count_text_tokens(msg)
        self.input_message_count = len(messages)

    def outlet(
        self,
        body: dict,
        __model__: Optional[dict] = None,
    ) -> dict:
        self.output_tokens = self.count_text_tokens(body["messages"][-1])
        model_base, model_name = self._infer_model_fields(body, __model__)
        self.model_base = model_base
        self.model_name = model_name
        end_time = time.time()
        if self.start_time:
            self.elapsed_time = end_time - self.start_time
        return body

    async def show_exceeded_status(
        self, __event_emitter__: Callable[[Any], Awaitable[None]], message_count: int
    ) -> None:
        if self.limit_exceeded and getattr(self.valves, "show_status", True):
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"🔥 Context limit reached - keeping last {message_count} messages",
                        "done": True,
                    },
                }
            )

    def count_text_tokens(self, msg: dict) -> int:
        content = msg.get("content", "")
        total_tokens = 0

        if isinstance(content, list):
            # Handle multi-modal content
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    total_tokens += len(self.encoding.encode(text))
        elif isinstance(content, str):
            # Handle text-only content
            total_tokens = len(self.encoding.encode(content))
        else:
            # Handle unexpected content types
            total_tokens = 0

        return total_tokens
